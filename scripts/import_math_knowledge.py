from __future__ import annotations

import argparse
from pathlib import Path

from app.core.config import settings
from app.services.math_knowledge_importer import SOURCE_REGISTRY, import_math_knowledge


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Crawl public math knowledge sources and append validated MathRAG seed JSONL records."
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        default=["proofwiki", "wikibooks", "wikipedia"],
        choices=sorted(SOURCE_REGISTRY),
        help="Data sources to use. PlanetMath is supported through limited HTML fetching.",
    )
    parser.add_argument(
        "--keywords",
        nargs="+",
        required=True,
        help="Topic keywords to search, for example: derivative linear_algebra",
    )
    parser.add_argument(
        "--limit-per-source",
        type=int,
        default=3,
        help="Search result limit per source and keyword.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=settings.RAW_KB_PATH,
        help="Output JSONL path. Defaults to data/raw/math_knowledge_seed.jsonl.",
    )
    parser.add_argument(
        "--error-output",
        type=Path,
        default=settings.RAW_DATA_DIR / "math_knowledge_import_errors.jsonl",
        help="Invalid LLM outputs and transform errors are appended here.",
    )
    parser.add_argument("--category", default=None, help="Optional category hint.")
    parser.add_argument("--max-chunk-chars", type=int, default=6000, help="Maximum cleaned text chars per LLM chunk.")
    parser.add_argument("--delay-seconds", type=float, default=1.0, help="Delay between source requests.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = import_math_knowledge(
        sources=args.sources,
        keywords=args.keywords,
        limit_per_source=args.limit_per_source,
        output_path=args.output,
        error_path=args.error_output,
        category=args.category,
        max_chunk_chars=args.max_chunk_chars,
        delay_seconds=args.delay_seconds,
    )

    print("Import finished.")
    print(f"Documents: {summary['documents']}")
    print(f"Chunks: {summary['chunks']}")
    print(f"Saved records: {summary['saved_records']}")
    print(f"Output: {args.output}")
    print(f"Errors: {args.error_output}")
    print("Next: python -m scripts.build_kb")
    print("Next: python -m scripts.import_legacy_knowledge")
    print("Next: python -m scripts.reindex_knowledge")


if __name__ == "__main__":
    main()
