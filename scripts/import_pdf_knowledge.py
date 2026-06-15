from __future__ import annotations

import argparse
from pathlib import Path

from app.core.config import settings
from app.services.pdf_knowledge_importer import (
    DEFAULT_DATA_LAKE_DIR,
    DEFAULT_ERROR_OUTPUT,
    DEFAULT_TEXT_OUTPUT,
    import_pdf_knowledge,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract local PDFs from data/data_lake into cleaned text chunks, optionally importing them as MathRAG seed records."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_LAKE_DIR,
        help="Directory containing local PDF files. Defaults to data/data_lake.",
    )
    parser.add_argument(
        "--text-output",
        type=Path,
        default=DEFAULT_TEXT_OUTPUT,
        help="Cleaned PDF text chunk JSONL output path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=settings.RAW_KB_PATH,
        help="Knowledge seed JSONL output path when --import-to-knowledge is used.",
    )
    parser.add_argument(
        "--error-output",
        type=Path,
        default=DEFAULT_ERROR_OUTPUT,
        help="Invalid LLM outputs and PDF import errors are appended here.",
    )
    parser.add_argument("--no-recursive", action="store_true", help="Do not scan subdirectories.")
    parser.add_argument("--append-text-output", action="store_true", help="Append text chunks instead of overwriting the text output.")
    parser.add_argument("--max-chunk-chars", type=int, default=4000, help="Maximum cleaned text chars per chunk.")
    parser.add_argument("--max-chunks", type=int, default=None, help="Optional hard limit for text chunks.")
    parser.add_argument(
        "--import-to-knowledge",
        action="store_true",
        help="After extracting text chunks, call the LLM and append validated Chinese seed records.",
    )
    parser.add_argument("--category", default=None, help="Optional category hint, for example 函数.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = import_pdf_knowledge(
        data_dir=args.data_dir,
        text_output_path=args.text_output,
        output_path=args.output,
        error_path=args.error_output,
        recursive=not args.no_recursive,
        max_chunk_chars=args.max_chunk_chars,
        max_chunks=args.max_chunks,
        extract_only=not args.import_to_knowledge,
        append_text_output=args.append_text_output,
        category=args.category,
    )

    print("PDF import finished.")
    print(f"Documents: {result.documents}")
    print(f"Text chunks: {result.text_chunks}")
    print(f"Saved records: {result.saved_records}")
    print(f"Text output: {result.text_output}")
    print(f"Errors: {result.error_output}")
    if args.import_to_knowledge:
        print("Next: python -m scripts.validate_seed_jsonl")
        print("Next: python -m scripts.build_kb")
        print("Next: python -m scripts.build_index")


if __name__ == "__main__":
    main()
