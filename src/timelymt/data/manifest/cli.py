"""Command-line interface for manifest construction and explicit split files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .builder import build_dataset_manifest
from .core import build_experimental_split, dataset_manifest_checksum, serialize_json, validate_dataset_manifest


DEFAULT_MANIFEST = Path("data/manifests/streaming-dataset.json")


def _load_manifest(path: Path) -> dict[str, object]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot load dataset manifest {path}: {error}") from error
    if not isinstance(document, dict):
        raise ValueError(f"Dataset manifest must be an object: {path}")
    validate_dataset_manifest(document)
    return document


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build and use TimelyMT dataset manifests and talk-level splits")
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build", help="Index validated canonical talks")
    build.add_argument("--output", type=Path, default=DEFAULT_MANIFEST)
    build.add_argument("--processed-root", type=Path, default=Path("data/streaming/processed"))
    summary = commands.add_parser("summary", help="Report dataset manifest statistics")
    summary.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    split = commands.add_parser("split", help="Persist a deterministic future experimental talk split")
    split.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    split.add_argument("--output", type=Path, default=Path("data/splits/experimental.json"))
    split.add_argument("--seed", type=int, required=True)
    split.add_argument("--train-ratio", type=float, required=True)
    split.add_argument("--dev-ratio", type=float, required=True)
    split.add_argument("--test-ratio", type=float, required=True)
    split.add_argument("--group-by", choices=("talk", "speaker"), default="speaker")
    split.add_argument("--minimum-talk-count", type=int, default=10)
    split.add_argument("--allow-tiny-dataset", action="store_true")
    split.add_argument("--exclude-from-test", action="append", default=[])
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "build":
            manifest = build_dataset_manifest(processed_root=args.processed_root)
            _write(args.output, manifest)
            print(f"{args.output.as_posix()}: {len(manifest['talks'])} talks, checksum {dataset_manifest_checksum(manifest)}")
        elif args.command == "summary":
            _print_summary(_load_manifest(args.manifest))
        else:
            manifest = _load_manifest(args.manifest)
            split = build_experimental_split(manifest, seed=args.seed, train_ratio=args.train_ratio, dev_ratio=args.dev_ratio, test_ratio=args.test_ratio, group_by=args.group_by, minimum_talk_count=args.minimum_talk_count, allow_tiny_dataset=args.allow_tiny_dataset, test_exclusions=args.exclude_from_test)
            _write(args.output, split)
            print(f"{args.output.as_posix()}: persisted deterministic {args.group_by}-grouped experimental split")
    except (OSError, ValueError) as error:
        print(error, file=sys.stderr)
        return 2
    return 0


def _write(path: Path, document: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(serialize_json(document), encoding="utf-8", newline="\n")


def _print_summary(manifest: dict[str, object]) -> None:
    talks = manifest["talks"]
    assert isinstance(talks, list)
    totals = {key: sum(talk["statistics"][key] for talk in talks) for key in ("source_segments", "target_segments", "alignment_units", "stream_tokens", "source_clock_duration_ms")}
    distributions = {field: {} for field in ("timing_mode", "domain")}
    for talk in talks:
        for field in distributions:
            value = talk.get(field, "unknown")
            distributions[field][value] = distributions[field].get(value, 0) + 1
    print(f"talks: {len(talks)}")
    for key, value in totals.items(): print(f"{key}: {value}")
    for field, values in distributions.items(): print(f"{field}: {json.dumps(values, sort_keys=True)}")
    print(f"checksum: {dataset_manifest_checksum(manifest)}")


if __name__ == "__main__":
    raise SystemExit(main())
