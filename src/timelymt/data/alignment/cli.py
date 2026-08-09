"""Command-line entrypoint for monotonic English-Vietnamese alignment."""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
import sys

from .core import load_parsed_transcript, write_alignment, write_review
from .dp import AlignmentParameters, align_transcripts


DEFAULT_PARSED_ROOT = Path("data/streaming/parsed")
DEFAULT_ALIGNED_ROOT = Path("data/streaming/aligned")
DEFAULT_CONFIG = Path("configs/data/alignment.json")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Align parsed English and Vietnamese transcript segments")
    parser.add_argument("--talk", required=True, help="Parsed talk ID")
    parser.add_argument("--parsed-root", type=Path, default=DEFAULT_PARSED_ROOT)
    parser.add_argument("--aligned-root", type=Path, default=DEFAULT_ALIGNED_ROOT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--max-group-size", type=int, choices=(1, 2, 3))
    parser.add_argument("--skip-penalty", type=float)
    parser.add_argument("--group-penalty", type=float)
    parser.add_argument("--force", action="store_true", help="Replace an existing alignment artifact")
    return parser


def main(argv: list[str] | None = None) -> int:
    if isinstance(sys.stdout, io.TextIOWrapper) and isinstance(sys.stderr, io.TextIOWrapper):
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
        sys.stderr.reconfigure(encoding="utf-8", errors="backslashreplace")
    args = build_parser().parse_args(argv)
    try:
        parameters = _parameters(args)
    except (OSError, ValueError) as error:
        print(error, file=sys.stderr)
        return 2
    output_directory = args.aligned_root / args.talk
    alignment_path = output_directory / "alignment.json"
    review_path = output_directory / "review.tsv"
    if alignment_path.exists() and not args.force:
        print(f"Alignment already exists; pass --force to replace it: {alignment_path}", file=sys.stderr)
        return 2

    source_path = args.parsed_root / args.talk / "source.en.json"
    target_path = args.parsed_root / args.talk / "target.vi.json"
    try:
        source = load_parsed_transcript(source_path)
        target = load_parsed_transcript(target_path)
        aligned = align_transcripts(
            source,
            target,
            source_path=source_path,
            target_path=target_path,
            parameters=parameters,
        )
        write_alignment(alignment_path, aligned)
        write_review(review_path, aligned.alignments)
    except (OSError, ValueError) as error:
        print(error, file=sys.stderr)
        return 2

    statistics = aligned.statistics
    print(
        f"{alignment_path.as_posix()}: {statistics['alignment_unit_count']} units, "
        f"{statistics['unaligned_source_count']} source skips, "
        f"{statistics['unaligned_target_count']} target skips"
    )
    print(review_path.as_posix())
    return 0


def _parameters(args: argparse.Namespace) -> AlignmentParameters:
    values = {"max_group_size": 3, "skip_penalty": 1.6, "group_penalty": 0.65}
    if args.config.is_file():
        document = json.loads(args.config.read_text(encoding="utf-8"))
        if not isinstance(document, dict):
            raise ValueError("Alignment config must be a JSON object")
        for key in values:
            if key in document:
                values[key] = document[key]
    for key in values:
        override = getattr(args, key)
        if override is not None:
            values[key] = override
    parameters = AlignmentParameters(**values)
    parameters.validate()
    return parameters


if __name__ == "__main__":
    raise SystemExit(main())
