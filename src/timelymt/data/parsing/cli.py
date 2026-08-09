"""Command-line entrypoint for transcript and caption parsing."""

from __future__ import annotations

import argparse
import io
from pathlib import Path
import sys

from .core import write_parsed_transcript
from .ted import TedContinuousTranscriptParser
from .wit3 import Wit3CaptionParser


DEFAULT_RAW_ROOT = Path("data/streaming/raw")
DEFAULT_PARSED_ROOT = Path("data/streaming/parsed")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Parse raw transcripts into monolingual normalized segments")
    parser.add_argument("--provider", required=True, choices=("ted", "wit3"))
    parser.add_argument("--talk", help="Acquired TED candidate ID")
    parser.add_argument("--talk-id", help="Talk ID to select from WIT3 XML")
    parser.add_argument("--language", choices=("en", "vi"), help="Parse one language; TED defaults to both")
    parser.add_argument("--input", type=Path, help="Local WIT3 XML file")
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--parsed-root", type=Path, default=DEFAULT_PARSED_ROOT)
    return parser


def main(argv: list[str] | None = None) -> int:
    if isinstance(sys.stdout, io.TextIOWrapper) and isinstance(sys.stderr, io.TextIOWrapper):
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
        sys.stderr.reconfigure(encoding="utf-8", errors="backslashreplace")
    args = build_parser().parse_args(argv)
    try:
        outputs = _parse(args)
    except ValueError as error:
        print(error, file=sys.stderr)
        return 2
    for output, count in outputs:
        print(f"{output.as_posix()}: {count} segments")
    return 0


def _parse(args: argparse.Namespace) -> list[tuple[Path, int]]:
    if args.provider == "ted":
        if not args.talk or args.input or args.talk_id:
            raise ValueError("TED parsing requires --talk and does not accept --input or --talk-id")
        parser = TedContinuousTranscriptParser()
        raw_directory = args.raw_root / "ted" / args.talk
        languages = (args.language,) if args.language else ("en", "vi")
        outputs: list[tuple[Path, int]] = []
        for language in languages:
            filename = "source.en.txt" if language == "en" else "target.vi.txt"
            input_path = raw_directory / filename
            if not input_path.is_file():
                raise ValueError(f"Missing acquired TED transcript: {input_path}")
            transcript = parser.parse(input_path, talk_id=args.talk, language=language)
            output = args.parsed_root / args.talk / _output_filename(language)
            write_parsed_transcript(output, transcript)
            outputs.append((output, len(transcript.segments)))
        return outputs

    if args.talk or not args.input or not args.talk_id or not args.language:
        raise ValueError("WIT3 parsing requires --input, --talk-id, and --language, without --talk")
    parser = Wit3CaptionParser()
    transcript = parser.parse(args.input, talk_id=args.talk_id, language=args.language)
    output = args.parsed_root / transcript.talk_id / _output_filename(args.language)
    write_parsed_transcript(output, transcript)
    return [(output, len(transcript.segments))]


def _output_filename(language: str) -> str:
    return "source.en.json" if language == "en" else "target.vi.json"


if __name__ == "__main__":
    raise SystemExit(main())
