"""Command-line entrypoint for source-only streaming token timing."""

from __future__ import annotations

import argparse
import io
from pathlib import Path
import sys

from .core import build_timed_source, load_parsed_transcript, write_timed_source


DEFAULT_PARSED_ROOT = Path("data/streaming/parsed")
DEFAULT_TIMED_ROOT = Path("data/streaming/timed")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a source-only timed lexical token stream")
    parser.add_argument("--talk", required=True, help="Parsed talk ID")
    parser.add_argument("--parsed-root", type=Path, default=DEFAULT_PARSED_ROOT)
    parser.add_argument("--timed-root", type=Path, default=DEFAULT_TIMED_ROOT)
    parser.add_argument("--words-per-second", type=float, default=2.5)
    parser.add_argument("--allocation", choices=("uniform", "character_weighted"), default="character_weighted")
    parser.add_argument("--force", action="store_true", help="Replace an existing timed artifact")
    return parser


def main(argv: list[str] | None = None) -> int:
    if isinstance(sys.stdout, io.TextIOWrapper) and isinstance(sys.stderr, io.TextIOWrapper):
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
        sys.stderr.reconfigure(encoding="utf-8", errors="backslashreplace")
    args = build_parser().parse_args(argv)
    source_path = args.parsed_root / args.talk / "source.en.json"
    output_path = args.timed_root / args.talk / "source.en.json"
    if output_path.exists() and not args.force:
        print(f"Timed source already exists; pass --force to replace it: {output_path}", file=sys.stderr)
        return 2
    try:
        source = load_parsed_transcript(source_path)
        timed = build_timed_source(
            source,
            source_path=source_path,
            words_per_second=args.words_per_second,
            allocation=args.allocation,
        )
        write_timed_source(output_path, timed)
    except (OSError, ValueError) as error:
        print(error, file=sys.stderr)
        return 2
    statistics = timed.statistics
    print(
        f"{output_path.as_posix()}: {statistics['segment_count']} segments, "
        f"{statistics['token_count']} tokens, {statistics['duration_ms']} ms, "
        f"{timed.timing['mode']}, {timed.timing['parameters']['allocation']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
