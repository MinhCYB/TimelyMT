"""Command-line entrypoint for curated talk acquisition."""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
import sys

from .core import acquire_candidates, load_manifest, validate_artifacts
from .ted import TedAdapter


DEFAULT_MANIFEST = Path("data/manifests/ted-ai-candidates.json")
DEFAULT_RAW_ROOT = Path("data/streaming/raw")
DEFAULT_RESULTS = Path("data/manifests/acquisition-results.jsonl")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Acquire curated public talk transcripts")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--talk", help="Candidate id or slug")
    selection.add_argument("--priority", help="Acquire one priority group, such as P0")
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--request-delay", type=float, default=1.0)
    parser.add_argument("--force", action="store_true", help="Re-acquire existing completed attempts")
    return parser


def main(argv: list[str] | None = None) -> int:
    if isinstance(sys.stdout, io.TextIOWrapper) and isinstance(sys.stderr, io.TextIOWrapper):
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
        sys.stderr.reconfigure(encoding="utf-8", errors="backslashreplace")
    args = build_parser().parse_args(argv)
    try:
        candidates = load_manifest(args.manifest)
    except ValueError as error:
        print(error, file=sys.stderr)
        return 2

    if args.talk:
        candidates = [item for item in candidates if item.id == args.talk or item.slug == args.talk]
    elif args.priority:
        candidates = [item for item in candidates if item.priority == args.priority]
    if not candidates:
        print("No candidates matched the selection", file=sys.stderr)
        return 2

    adapter = TedAdapter(
        timeout=args.timeout,
        retries=max(0, args.retries),
        request_delay=max(0.0, args.request_delay),
    )
    results = acquire_candidates(
        candidates,
        {adapter.provider: adapter},
        args.raw_root,
        args.results,
        skip_existing=not args.force,
    )
    validation_errors = validate_artifacts(results, Path.cwd())
    for result in results:
        print(json.dumps(result.to_dict(), ensure_ascii=False, sort_keys=True))
    for error in validation_errors:
        print(f"validation error: {error}", file=sys.stderr)

    failed = sum(result.status == "failed" for result in results)
    print(f"Attempted {len(results)} talk(s); failures: {failed}", file=sys.stderr)
    return 1 if failed or validation_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
