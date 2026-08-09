"""Command-line entrypoint for canonical streaming-talk assembly."""

from __future__ import annotations

import argparse
import io
from pathlib import Path
import sys

from .builder import build_canonical_talk
from .core import write_canonical_talk


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Assemble a canonical streaming talk from completed upstream artifacts")
    parser.add_argument("--talk", required=True, help="Talk ID to canonicalize")
    parser.add_argument("--force", action="store_true", help="Replace an existing canonical artifact")
    return parser


def main(argv: list[str] | None = None) -> int:
    if isinstance(sys.stdout, io.TextIOWrapper) and isinstance(sys.stderr, io.TextIOWrapper):
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
        sys.stderr.reconfigure(encoding="utf-8", errors="backslashreplace")
    args = build_parser().parse_args(argv)
    output = Path("data/streaming/processed") / args.talk / "streaming-talk.json"
    if output.exists() and not args.force:
        print(f"Canonical talk already exists; pass --force to replace it: {output}", file=sys.stderr)
        return 2
    try:
        document = build_canonical_talk(args.talk)
        write_canonical_talk(output, document)
    except (OSError, ValueError) as error:
        print(error, file=sys.stderr)
        return 2
    print(f"{output.as_posix()}: {len(document['source']['segments'])} source segments, {len(document['target_reference']['segments'])} target segments, {len(document['alignments'])} alignment units, {len(document['stream']['tokens'])} stream tokens, {document['stream']['timing_mode']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
