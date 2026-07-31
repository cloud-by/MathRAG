"""不信任客户端路径的 PDF 流式受控存储。"""

from __future__ import annotations

import asyncio
import hashlib
import os
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PureWindowsPath
from typing import Awaitable, Callable, Protocol
from uuid import UUID, uuid4

from app.core.errors import AppError
from app.modules.ingestion.errors import (
    DocumentMimeError,
    DocumentPathError,
    DocumentPdfInvalidError,
    DocumentStorageError,
    DocumentTooLargeError,
)
from app.modules.ingestion.extractors import extract_pdf_text


PDF_MIME_TYPE = "application/pdf"
PDF_MAGIC = b"%PDF-"
UPLOAD_CHUNK_BYTES = 1024 * 1024


class UploadReadable(Protocol):
    """UploadFile 及测试替身所需的最小读取协议。"""

    filename: str | None
    content_type: str | None

    def read(self, size: int = -1) -> Awaitable[bytes]: ...


@dataclass(frozen=True)
class StoredUpload:
    """可安全跨越事务边界的上传结果，不携带绝对路径。"""

    relative_path: str
    size_bytes: int
    sha256: str
    original_name: str
    mime_type: str


def validate_original_name(filename: str | None) -> str:
    """验证客户端文件名仅为元数据，并返回去除首尾空白后的名称。"""
    if not isinstance(filename, str):
        raise DocumentPathError()
    normalized = filename.strip()
    windows_path = PureWindowsPath(normalized)
    if (
        not normalized
        or normalized in {".", ".."}
        or "\x00" in normalized
        or "/" in normalized
        or "\\" in normalized
        or Path(normalized).is_absolute()
        or windows_path.is_absolute()
        or bool(windows_path.drive)
        or len(normalized) > 255
    ):
        raise DocumentPathError()
    if not normalized.lower().endswith(".pdf") or not normalized[:-4].strip():
        raise DocumentMimeError()
    return normalized


def resolve_stored_path(root: Path, relative_path: str) -> Path:
    """把数据库相对路径解析到受控根目录，并拒绝越界与符号链接逃逸。"""
    if (
        not isinstance(relative_path, str)
        or not relative_path
        or "\x00" in relative_path
        or "\\" in relative_path
    ):
        raise DocumentPathError()
    windows_path = PureWindowsPath(relative_path)
    if Path(relative_path).is_absolute() or windows_path.is_absolute() or windows_path.drive:
        raise DocumentPathError()

    try:
        root_resolved = root.resolve()
        candidate = (root_resolved / relative_path).resolve()
    except (OSError, RuntimeError, ValueError):
        raise DocumentPathError() from None
    try:
        candidate.relative_to(root_resolved)
    except ValueError:
        raise DocumentPathError() from None
    if candidate == root_resolved:
        raise DocumentPathError()
    return candidate


class UploadStorage:
    """使用服务端 UUID 路径原子保存一个 PDF 上传。"""

    def __init__(
        self,
        *,
        root: Path,
        max_bytes: int,
        max_pages: int,
        now: Callable[[], datetime] | None = None,
        uuid_factory: Callable[[], UUID] | None = None,
    ) -> None:
        if max_bytes <= 0:
            raise ValueError("max_bytes 必须大于 0")
        if max_pages <= 0:
            raise ValueError("max_pages 必须大于 0")
        self._root = root
        self._max_bytes = max_bytes
        self._max_pages = max_pages
        self._now = now or (lambda: datetime.now(timezone.utc))
        self._uuid_factory = uuid_factory or uuid4
        # 记录本实例成功发布文件的物理身份，清理时不误删同路径替换文件。
        self._owned_uploads: dict[str, tuple[int, int]] = {}

    async def save_upload(self, upload: UploadReadable) -> StoredUpload:
        """流式校验、摘要并保存上传；失败或取消不会遗留文件。"""
        original_name = validate_original_name(upload.filename)
        if upload.content_type != PDF_MIME_TYPE:
            raise DocumentMimeError()

        received_at = self._now()
        file_id = UUID(str(self._uuid_factory()))
        relative_path = (
            f"{received_at.year:04d}/{received_at.month:02d}/{file_id}.pdf"
        )
        final_path = resolve_stored_path(self._root, relative_path)
        part_path = final_path.with_suffix(".part")
        digest = hashlib.sha256()
        header = bytearray()
        size_bytes = 0
        part_created = False

        try:
            final_path.parent.mkdir(parents=True, exist_ok=True)
            if final_path.exists():
                raise DocumentStorageError()
            part_file = part_path.open("xb")
            part_created = True
            with part_file as output:
                while True:
                    # 多读一个字节即可尽早判定超限，避免继续消费上传流。
                    read_size = min(
                        UPLOAD_CHUNK_BYTES,
                        self._max_bytes - size_bytes + 1,
                    )
                    chunk = await upload.read(read_size)
                    if not chunk:
                        break
                    if size_bytes + len(chunk) > self._max_bytes:
                        raise DocumentTooLargeError()
                    if len(header) < len(PDF_MAGIC):
                        header.extend(chunk[: len(PDF_MAGIC) - len(header)])
                        if len(header) == len(PDF_MAGIC) and bytes(header) != PDF_MAGIC:
                            raise DocumentPdfInvalidError()
                    output.write(chunk)
                    digest.update(chunk)
                    size_bytes += len(chunk)

                if bytes(header) != PDF_MAGIC:
                    raise DocumentPdfInvalidError()

            await self._validate_pdf(part_path)
            stored = StoredUpload(
                relative_path=relative_path,
                size_bytes=size_bytes,
                sha256=digest.hexdigest(),
                original_name=original_name,
                mime_type=PDF_MIME_TYPE,
            )
            self._owned_uploads[relative_path] = self._publish_no_replace(
                part_path,
                final_path,
            )
            return stored
        except asyncio.CancelledError:
            if part_created:
                self._remove_if_exists(part_path)
            raise
        except AppError:
            if part_created:
                self._remove_if_exists(part_path)
            raise
        except Exception:
            if part_created:
                self._remove_if_exists(part_path)
            raise DocumentStorageError() from None

    async def delete_upload(self, relative_path: str) -> None:
        """删除本实例刚保存的文件，拒绝越界、非本次或已被替换的路径。"""
        final_path = resolve_stored_path(self._root, relative_path)
        expected_identity = self._owned_uploads.get(relative_path)
        if expected_identity is None:
            raise DocumentPathError()

        try:
            await asyncio.to_thread(
                self._delete_if_identity_matches,
                final_path,
                expected_identity,
            )
        finally:
            self._owned_uploads.pop(relative_path, None)

    def release_upload(self, relative_path: str) -> None:
        """数据库提交成功后仅释放清理所有权，不删除已持久化文件。"""
        self._owned_uploads.pop(relative_path, None)

    async def _validate_pdf(self, part_path: Path) -> None:
        """在线程中解析完整 PDF；取消时等待线程退出后再清理临时文件。"""
        validation = asyncio.create_task(
            asyncio.to_thread(
                extract_pdf_text,
                part_path,
                max_pages=self._max_pages,
            )
        )
        try:
            await asyncio.shield(validation)
        except asyncio.CancelledError:
            with suppress(Exception):
                await validation
            raise

    @staticmethod
    def _publish_no_replace(
        part_path: Path,
        final_path: Path,
    ) -> tuple[int, int]:
        """以原子硬链接发布文件，绝不覆盖并发创建的目标。"""
        final_created = False
        try:
            os.link(part_path, final_path)
            final_created = True
            stat = final_path.stat()
            part_path.unlink()
            return stat.st_dev, stat.st_ino
        except OSError:
            if final_created:
                # 仅回滚已确认由本调用创建的硬链接，不删除竞争方文件。
                with suppress(OSError):
                    final_path.unlink()
            raise DocumentStorageError() from None

    @staticmethod
    def _delete_if_identity_matches(
        path: Path,
        expected_identity: tuple[int, int],
    ) -> None:
        try:
            stat = path.stat()
        except FileNotFoundError:
            return
        except OSError:
            raise DocumentStorageError() from None
        if (stat.st_dev, stat.st_ino) != expected_identity:
            raise DocumentPathError()
        try:
            path.unlink()
        except FileNotFoundError:
            return
        except OSError:
            raise DocumentStorageError() from None

    @staticmethod
    def _remove_if_exists(path: Path) -> None:
        with suppress(OSError):
            path.unlink(missing_ok=True)
