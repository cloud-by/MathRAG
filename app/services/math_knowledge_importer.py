from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence
from urllib.parse import quote, urljoin

import requests
from pydantic import ValidationError

from app.core.config import settings
from app.schemas.knowledge import KnowledgeRecord, SEED_FIELD_ORDER
from app.services.knowledge_extractor import (
    append_records,
    generate_next_ids,
    normalize_drafts,
)
from app.services.llm_service import chat_json


USER_AGENT = "MathRAG/0.1 educational knowledge importer"
REQUEST_TIMEOUT = 180


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class SourceDocument:
    source_name: str
    source_url: str
    title: str
    license: str
    text: str
    chapter: str = ""
    section: str = ""


@dataclass(frozen=True)
class TextChunk:
    document: SourceDocument
    chunk_index: int
    text: str


class CleanTextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: List[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrs_dict = {name.lower(): value or "" for name, value in attrs}
        classes = attrs_dict.get("class", "")
        element_id = attrs_dict.get("id", "")

        if tag in {"script", "style", "noscript", "footer", "header", "nav"}:
            self._skip_depth += 1
            return

        if "mw-editsection" in classes or "toc" in classes or element_id == "toc":
            self._skip_depth += 1
            return

        if tag in {"p", "div", "section", "article", "li", "h1", "h2", "h3", "h4", "tr"}:
            self.parts.append("\n")

        if tag == "img":
            alt = attrs_dict.get("alt", "").strip()
            if alt:
                self.parts.append(f" {alt} ")

    def handle_endtag(self, tag: str) -> None:
        if self._skip_depth:
            self._skip_depth -= 1
            return
        if tag in {"p", "div", "section", "article", "li", "h1", "h2", "h3", "h4", "tr"}:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        text = data.strip()
        if text:
            self.parts.append(text)

    def get_text(self) -> str:
        text = " ".join(self.parts)
        return clean_plain_text(text)


def clean_plain_text(text: str) -> str:
    text = text.replace("\u3000", " ")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\[\s*(edit|citation needed|note \d+|\d+)\s*\]", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(edit|source|citation needed)\b", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"[ \t\f\v]+", " ", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def html_to_clean_text(html: str) -> str:
    html = re.sub(r"<!--.*?-->", " ", html, flags=re.DOTALL)
    html = re.sub(r"<sup\b[^>]*class=\"[^\"]*\breference\b[^\"]*\"[^>]*>.*?</sup>", " ", html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r"<table\b[^>]*(?:navbox|metadata|infobox|ambox|vertical-navbox)[^>]*>.*?</table>", " ", html, flags=re.DOTALL | re.IGNORECASE)
    parser = CleanTextParser()
    parser.feed(html)
    return parser.get_text()


def chunk_text(text: str, max_chars: int = 6000) -> List[str]:
    text = clean_plain_text(text)
    if len(text) <= max_chars:
        return [text] if text else []

    paragraphs = [part.strip() for part in re.split(r"\n{2,}", text) if part.strip()]
    chunks: List[str] = []
    current = ""

    for paragraph in paragraphs:
        if len(paragraph) > max_chars:
            if current:
                chunks.append(current.strip())
                current = ""
            for start in range(0, len(paragraph), max_chars):
                chunks.append(paragraph[start : start + max_chars].strip())
            continue

        candidate = f"{current}\n\n{paragraph}".strip() if current else paragraph
        if len(candidate) > max_chars:
            chunks.append(current.strip())
            current = paragraph
        else:
            current = candidate

    if current:
        chunks.append(current.strip())
    return chunks


class MediaWikiSource:
    def __init__(self, name: str, api_url: str, page_base_url: str) -> None:
        self.name = name
        self.api_url = api_url
        self.page_base_url = page_base_url
        self._license: str | None = None

    def _get(self, params: Dict[str, Any]) -> Dict[str, Any]:
        response = requests.get(
            self.api_url,
            params=params,
            timeout=REQUEST_TIMEOUT,
            headers={"User-Agent": USER_AGENT},
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise ValueError(f"{self.name} returned a non-object API response")
        if "error" in payload:
            raise ValueError(f"{self.name} API error: {payload['error']}")
        return payload

    def license(self) -> str:
        if self._license:
            return self._license
        payload = self._get(
            {
                "action": "query",
                "meta": "siteinfo",
                "siprop": "rightsinfo",
                "format": "json",
                "formatversion": 2,
            }
        )
        rights = payload.get("query", {}).get("rightsinfo", {})
        self._license = str(rights.get("text") or rights.get("url") or "see source site").strip()
        return self._license

    def search(self, keyword: str, limit: int) -> List[str]:
        payload = self._get(
            {
                "action": "query",
                "list": "search",
                "srsearch": keyword,
                "srlimit": max(1, limit),
                "format": "json",
                "formatversion": 2,
            }
        )
        rows = payload.get("query", {}).get("search", [])
        if not isinstance(rows, list):
            return []
        titles = []
        for row in rows:
            if isinstance(row, dict) and row.get("title"):
                titles.append(str(row["title"]))
        return titles

    def fetch(self, title: str) -> SourceDocument:
        payload = self._get(
            {
                "action": "parse",
                "page": title,
                "prop": "text|sections|displaytitle",
                "disablelimitreport": 1,
                "format": "json",
                "formatversion": 2,
            }
        )
        parsed = payload.get("parse", {})
        html = parsed.get("text", "")
        if isinstance(html, dict):
            html = html.get("*", "")
        text = html_to_clean_text(str(html))
        clean_title = clean_plain_text(str(parsed.get("displaytitle") or title))
        url = urljoin(self.page_base_url, quote(title.replace(" ", "_"), safe="/:_"))
        chapter, section = split_wikibooks_title(title) if self.name == "wikibooks" else ("", "")
        return SourceDocument(
            source_name=self.name,
            source_url=url,
            title=clean_title,
            license=self.license(),
            text=text,
            chapter=chapter,
            section=section,
        )


class PlanetMathSource:
    """Limited HTML source.

    PlanetMath does not expose the same stable MediaWiki API used by the other
    supported sources here, so this implementation uses conservative HTML
    fetching and URL discovery. MathWorld, OpenStax, arXiv, and Math
    StackExchange are intentionally not used as primary bulk sources because
    their licensing, API, or quality-filtering constraints need a separate
    ingestion policy.
    """

    name = "planetmath"
    base_url = "https://planetmath.org/"
    license_text = "see PlanetMath source page"

    def search(self, keyword: str, limit: int) -> List[str]:
        candidates = [re.sub(r"[^a-z0-9]+", "", keyword.lower())]
        try:
            response = requests.get(
                urljoin(self.base_url, "search/node/" + quote(keyword)),
                timeout=REQUEST_TIMEOUT,
                headers={"User-Agent": USER_AGENT},
            )
            if response.ok:
                hrefs = re.findall(r'href=["\']([^"\']+)["\']', response.text)
                for href in hrefs:
                    if len(candidates) >= limit:
                        break
                    if "/node/" in href or href.startswith("/"):
                        url = urljoin(self.base_url, href)
                        if url not in candidates:
                            candidates.append(url)
        except requests.RequestException:
            pass
        return candidates[:limit]

    def fetch(self, title_or_url: str) -> SourceDocument:
        url = title_or_url if title_or_url.startswith("http") else urljoin(self.base_url, quote(title_or_url))
        response = requests.get(url, timeout=REQUEST_TIMEOUT, headers={"User-Agent": USER_AGENT})
        response.raise_for_status()
        title_match = re.search(r"<title[^>]*>(.*?)</title>", response.text, flags=re.DOTALL | re.IGNORECASE)
        title = clean_plain_text(title_match.group(1)) if title_match else title_or_url
        return SourceDocument(
            source_name=self.name,
            source_url=url,
            title=title,
            license=self.license_text,
            text=html_to_clean_text(response.text),
        )


def split_wikibooks_title(title: str) -> tuple[str, str]:
    parts = [part.strip() for part in title.split("/") if part.strip()]
    if not parts:
        return "", ""
    if len(parts) == 1:
        return parts[0], ""
    return parts[0], parts[-1]


SOURCE_REGISTRY: Dict[str, Any] = {
    "proofwiki": MediaWikiSource("proofwiki", "https://proofwiki.org/w/api.php", "https://proofwiki.org/wiki/"),
    "wikibooks": MediaWikiSource("wikibooks", "https://en.wikibooks.org/w/api.php", "https://en.wikibooks.org/wiki/"),
    "wikipedia": MediaWikiSource("wikipedia", "https://en.wikipedia.org/w/api.php", "https://en.wikipedia.org/wiki/"),
    "planetmath": PlanetMathSource(),
}


def write_source_error(path: Path, source_name: str, keyword: str, error: str) -> None:
    write_jsonl(
        path,
        [
            {
                "timestamp": utc_timestamp(),
                "error_type": "source_discovery",
                "source_name": source_name,
                "keyword": keyword,
                "error": error,
            }
        ],
    )


def discover_documents(
    sources: Sequence[str],
    keywords: Sequence[str],
    limit_per_source: int,
    delay_seconds: float = 1.0,
    error_path: Path | None = None,
) -> List[SourceDocument]:
    documents: List[SourceDocument] = []
    seen_urls = set()

    for source_name in sources:
        source = SOURCE_REGISTRY[source_name]
        for keyword in keywords:
            try:
                titles = source.search(keyword, limit_per_source)
            except Exception as exc:
                if error_path is not None:
                    write_source_error(error_path, source_name, keyword, str(exc))
                continue

            for title in titles:
                try:
                    document = source.fetch(title)
                except Exception as exc:
                    if error_path is not None:
                        write_source_error(error_path, source_name, keyword, f"fetch {title!r} failed: {exc}")
                    continue
                if not document.text or document.source_url in seen_urls:
                    continue
                documents.append(document)
                seen_urls.add(document.source_url)
                if delay_seconds > 0:
                    time.sleep(delay_seconds)

    return documents


def build_transform_messages(chunk: TextChunk, category: str | None) -> List[Dict[str, str]]:
    hints = {
        "category": category,
        "source_name": chunk.document.source_name,
        "source_url": chunk.document.source_url,
        "source_title": chunk.document.title,
        "license": chunk.document.license,
        "chapter": chunk.document.chapter,
        "section": chunk.document.section,
    }
    return [
        {
            "role": "system",
            "content": (
                "你是严格的中文数学知识库编辑。只返回合法 JSON，不要返回 markdown、注释或包裹文本。"
                "所有知识库字段内容必须用简体中文表达；数学符号、变量名和公式可以保留原文。"
                "涉及数学公式时，必须使用 KaTeX 可渲染的 LaTeX：行内公式用 \\( ... \\)，"
                "块级公式用 \\[ ... \\]，不要新增 $...$ 或 $$...$$。"
            ),
        },
        {
            "role": "user",
            "content": (
                "请把来源文本整理为 MathRAG 中文 seed 知识点。\n"
                "即使来源是英文，也必须翻译、改写、提炼为简体中文。\n"
                "JSON 对象必须严格为：{\"items\": [record, ...]}。\n"
                "每个 record 必须且只能包含以下字段：\n"
                f"{json.dumps(SEED_FIELD_ORDER, ensure_ascii=False)}\n\n"
                "字段规则：\n"
                "- id：省略或留空，导入器会自动分配 k0001 风格 id。\n"
                "- difficulty：只能是 easy、medium、hard。\n"
                "- category、title、keywords、content、example、steps 必须使用简体中文。\n"
                "- keywords 和 steps 必须是非空字符串数组。\n"
                "- content 要严谨、简洁，适合 RAG 问答，不要大段照搬来源文本。\n"
                "- 可保留必要公式、函数名、变量名、英文专名缩写，例如 f(x)、sin、L'Hospital。\n"
                "- content、example、steps 中如包含数学公式，必须统一为 KaTeX LaTeX 分隔符："
                "行内公式用 \\( ... \\)，块级公式用 \\[ ... \\]；不要新增 $...$ 或 $$...$$。\n"
                "- 如果来源文本中公式使用 $...$ 或 $$...$$，写入知识库前必须转换为 \\( ... \\) 或 \\[ ... \\]。\n"
                "- 字符串字段内部不要包含原始换行，不要把公式逐字符、逐行拆开。\n"
                "- 输出必须是可被 json.loads 解析的合法 JSON。\n"
                "- 从定义、定理、公式、证明、例题中抽取知识点；无关内容不要写入。\n\n"
                f"提示和来源元数据：\n{json.dumps(hints, ensure_ascii=False)}\n\n"
                f"清洗后的来源文本：\n{chunk.text}"
            ),
        },
    ]


def chinese_ratio(text: str) -> float:
    meaningful = re.findall(r"[\u4e00-\u9fffA-Za-z]", text)
    if not meaningful:
        return 0.0
    chinese = [char for char in meaningful if "\u4e00" <= char <= "\u9fff"]
    return len(chinese) / len(meaningful)


def validate_chinese_record(record: KnowledgeRecord, min_ratio: float = 0.35) -> None:
    text = " ".join(
        [
            record.category,
            record.title,
            " ".join(record.keywords),
            record.content,
            record.example,
            " ".join(record.steps),
        ]
    )
    ratio = chinese_ratio(text)
    if ratio < min_ratio:
        raise ValueError(
            f"record {record.id} is not Chinese enough: chinese_ratio={ratio:.2f}, title={record.title!r}"
        )


def normalize_ai_item(raw: Dict[str, Any], item_id: str, category: str | None) -> KnowledgeRecord:
    item = dict(raw)
    item["id"] = item_id
    if category and not str(item.get("category", "")).strip():
        item["category"] = category
    record = KnowledgeRecord(**{field: item.get(field) for field in SEED_FIELD_ORDER})
    validate_chinese_record(record)
    return record


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("a", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def write_error(path: Path, chunk: TextChunk, error: str, raw_response: Any | None = None) -> None:
    row = {
        "timestamp": utc_timestamp(),
        "error_type": "chunk_transform",
        "source_name": chunk.document.source_name,
        "source_url": chunk.document.source_url,
        "source_title": chunk.document.title,
        "license": chunk.document.license,
        "chunk_index": chunk.chunk_index,
        "error": error,
        "raw_response": raw_response,
        "text_preview": chunk.text[:1000],
    }
    write_jsonl(path, [row])


def transform_chunk(
    chunk: TextChunk,
    output_path: Path,
    error_path: Path,
    category: str | None = None,
) -> List[KnowledgeRecord]:
    raw_response: Any | None = None
    try:
        response = chat_json(messages=build_transform_messages(chunk, category), temperature=0.1)
        raw_response = response.data
        drafts = normalize_drafts(response.data, category)
        next_ids = generate_next_ids(len(drafts), output_path)
        records = [
            KnowledgeRecord(id=item_id, **draft.to_values())
            for item_id, draft in zip(next_ids, drafts, strict=True)
        ]
        for record in records:
            validate_chinese_record(record)

        append_records(records, output_path)
        return records
    except (ValidationError, ValueError, RuntimeError) as exc:
        write_error(error_path, chunk, str(exc), raw_response)
        return []


def import_math_knowledge(
    sources: Sequence[str],
    keywords: Sequence[str],
    limit_per_source: int = 3,
    output_path: Path = settings.RAW_KB_PATH,
    error_path: Path = settings.RAW_DATA_DIR / "math_knowledge_import_errors.jsonl",
    category: str | None = None,
    max_chunk_chars: int = 6000,
    delay_seconds: float = 1.0,
) -> Dict[str, int]:
    unknown_sources = sorted(set(sources) - set(SOURCE_REGISTRY))
    if unknown_sources:
        raise ValueError(f"unsupported sources: {', '.join(unknown_sources)}")
    if not keywords:
        raise ValueError("keywords cannot be empty")

    documents = discover_documents(
        sources=sources,
        keywords=keywords,
        limit_per_source=limit_per_source,
        delay_seconds=delay_seconds,
        error_path=error_path,
    )
    chunks = [
        TextChunk(document=document, chunk_index=index, text=chunk)
        for document in documents
        for index, chunk in enumerate(chunk_text(document.text, max_chars=max_chunk_chars))
    ]

    saved_records = 0
    for chunk in chunks:
        saved_records += len(
            transform_chunk(
                chunk=chunk,
                output_path=output_path,
                error_path=error_path,
                category=category,
            )
        )

    return {
        "documents": len(documents),
        "chunks": len(chunks),
        "saved_records": saved_records,
    }
