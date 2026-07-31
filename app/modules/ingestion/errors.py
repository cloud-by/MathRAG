"""文档接收与 PDF 抽取的稳定公开异常。"""

from __future__ import annotations

from app.core.errors import AppError


class DocumentPathError(AppError):
    """文件名或受控存储相对路径不合法。"""

    def __init__(self) -> None:
        super().__init__(
            code="DOCUMENT_PATH_INVALID",
            message="文档文件名或存储路径不合法。",
            status_code=422,
        )


class DocumentMimeError(AppError):
    """上传声明的类型不受支持。"""

    def __init__(self) -> None:
        super().__init__(
            code="DOCUMENT_MIME_UNSUPPORTED",
            message="仅支持 application/pdf 类型的 PDF 文档。",
            status_code=415,
        )


class DocumentTooLargeError(AppError):
    """上传内容超过服务端大小限制。"""

    def __init__(self) -> None:
        super().__init__(
            code="DOCUMENT_TOO_LARGE",
            message="上传文档超过允许的大小限制。",
            status_code=413,
        )


class DocumentPdfInvalidError(AppError):
    """PDF 文件头、结构或页面内容无法安全解析。"""

    def __init__(self) -> None:
        super().__init__(
            code="DOCUMENT_PDF_INVALID",
            message="PDF 文档无效或无法解析。",
            status_code=422,
        )


class DocumentPdfEncryptedError(AppError):
    """PDF 已加密，不能进入自动导入流程。"""

    def __init__(self) -> None:
        super().__init__(
            code="DOCUMENT_PDF_ENCRYPTED",
            message="不支持加密的 PDF 文档。",
            status_code=422,
        )


class DocumentPdfPageCountError(AppError):
    """PDF 页数为零或超过配置上限。"""

    def __init__(self) -> None:
        super().__init__(
            code="DOCUMENT_PDF_PAGE_COUNT_INVALID",
            message="PDF 页数不在允许范围内。",
            status_code=422,
        )


class DocumentPdfEmptyError(AppError):
    """PDF 页面中没有可用于导入的文本。"""

    def __init__(self) -> None:
        super().__init__(
            code="DOCUMENT_PDF_EMPTY",
            message="PDF 文档没有可提取的文本。",
            status_code=422,
        )


class DocumentStorageError(AppError):
    """受控文件存储发生非预期 I/O 失败。"""

    def __init__(self) -> None:
        super().__init__(
            code="DOCUMENT_STORAGE_FAILED",
            message="文档存储暂时不可用。",
            status_code=503,
        )
