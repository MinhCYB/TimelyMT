"""High-level M0 dataset preparation and validation commands."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from timelymt.data.acquisition.core import load_manifest
from timelymt.data.acquisition.ted import TedAdapter
from timelymt.data.alignment.dp import AlignmentParameters

from .calibration import (
    build_review_set,
    evaluate_grid,
    import_alignment_review_tsv,
    load_alignment_config,
    write_review_set,
)
from .core import STAGES, prepare_dataset, write_pipeline_results
from .qa import build_quality_report, validate_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare and validate the TimelyMT streaming dataset")
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--manifest", type=Path, default=Path("data/manifests/ted-ai-candidates.json"))
    prepare.add_argument("--alignment-config", type=Path, default=Path("configs/data/alignment.json"))
    prepare.add_argument("--talk")
    prepare.add_argument("--priority")
    prepare.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    prepare.add_argument("--force-stage", choices=STAGES)
    prepare.add_argument("--timeout", type=float, default=20.0)
    prepare.add_argument("--request-delay", type=float, default=1.0)
    commands.add_parser("validate")
    summary = commands.add_parser("summary")
    summary.add_argument("--output", type=Path)
    review = commands.add_parser("build-calibration-set")
    review.add_argument("--aligned-root", type=Path, default=Path("data/streaming/aligned"))
    review.add_argument("--output", type=Path, default=Path("data/review/alignment-calibration.json"))
    review.add_argument("--per-talk", type=int, default=25)
    calibrate = commands.add_parser("calibrate")
    calibrate.add_argument("--review", type=Path, default=Path("data/review/alignment-calibration.json"))
    calibrate.add_argument("--parsed-root", type=Path, default=Path("data/streaming/parsed"))
    calibrate.add_argument("--output", type=Path, default=Path("data/review/alignment-calibration-results.json"))
    import_review = commands.add_parser("import-alignment-review")
    import_review.add_argument("--input", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "prepare":
            candidates = load_manifest(args.manifest)
            if args.talk:
                candidates = [item for item in candidates if item.id == args.talk or item.slug == args.talk]
            if args.priority:
                candidates = [item for item in candidates if item.priority == args.priority]
            if not candidates:
                raise ValueError("No candidates matched the selection")
            adapter = TedAdapter(timeout=args.timeout, request_delay=args.request_delay)
            records = prepare_dataset(
                candidates,
                adapters={adapter.provider: adapter},
                alignment_parameters=load_alignment_config(args.alignment_config),
                resume=args.resume,
                force_stage=args.force_stage,
            )
            write_pipeline_results(Path("outputs/dataset/pipeline-results.json"), records)
            print(f"attempted: {len(records)}; accepted: {sum(item['status'] == 'accepted' for item in records)}")
            return 1 if any(item["status"] == "failed" for item in records) else 0
        if args.command == "build-calibration-set":
            paths = sorted(args.aligned_root.glob("*/alignment.json"))
            if not paths:
                raise ValueError("No alignment artifacts found")
            document = build_review_set(paths, per_talk=args.per_talk)
            write_review_set(args.output, document)
            print(f"{args.output.as_posix()}: {len(document['examples'])} examples requiring review")
            return 0
        if args.command == "calibrate":
            review = json.loads(args.review.read_text(encoding="utf-8"))
            candidates = [
                AlignmentParameters(3, skip, group)
                for skip in (1.2, 1.4, 1.6)
                for group in (0.35, 0.5, 0.65)
            ]
            result = evaluate_grid(review, parsed_root=args.parsed_root, candidates=candidates)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
            print(f"{args.output.as_posix()}: {result['status']}")
            return 0 if result["status"] == "complete" else 1
        if args.command == "import-alignment-review":
            reviewed = import_alignment_review_tsv(
                args.input,
                Path("data/review/alignment-calibration.json"),
                parsed_root=Path("data/streaming/parsed"),
            )
            print(f"data/review/alignment-calibration.json: {reviewed} reviewed examples")
            return 0
        manifest_path = Path("data/manifests/streaming-dataset.json")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if args.command == "validate":
            print(json.dumps(validate_dataset(manifest), sort_keys=True))
        else:
            paths = [Path(item["canonical_path"]) for item in manifest["talks"]]
            report = build_quality_report(paths)
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
            print(json.dumps(report["summary"], sort_keys=True))
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(error, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
