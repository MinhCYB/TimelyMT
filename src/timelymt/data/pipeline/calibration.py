"""Persistent review sampling and small deterministic alignment calibration grids."""

from __future__ import annotations

from dataclasses import asdict
import csv
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from timelymt.data.alignment.core import load_parsed_transcript
from timelymt.data.alignment.dp import AlignmentParameters, align_transcripts


CONFIG_VERSION = "1.0.0"
REVIEW_TSV_COLUMNS = (
    "talk_id",
    "alignment_id",
    "alignment_type",
    "cost",
    "selection_reasons",
    "source_segment_ids",
    "source_text",
    "target_segment_ids",
    "target_text",
    "verdict",
    "preferred_source_ids",
    "preferred_target_ids",
    "note",
)


def load_alignment_config(path: Path) -> AlignmentParameters:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
        parameters = AlignmentParameters(
            max_group_size=document["max_group_size"],
            skip_penalty=document["skip_penalty"],
            group_penalty=document["group_penalty"],
        )
    except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError) as error:
        raise ValueError(f"Cannot load alignment config {path}: {error}") from error
    parameters.validate()
    return parameters


def write_alignment_config(
    path: Path,
    parameters: AlignmentParameters,
    *,
    selection_basis: str,
) -> dict[str, Any]:
    parameters.validate()
    document = {
        "version": CONFIG_VERSION,
        **asdict(parameters),
        "selection_basis": selection_basis,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    return document


def build_review_set(
    alignment_paths: Iterable[Path],
    *,
    per_talk: int = 25,
) -> dict[str, Any]:
    examples: list[dict[str, Any]] = []
    for path in sorted(alignment_paths):
        document = json.loads(path.read_text(encoding="utf-8"))
        units = document["alignments"]
        selected = _sample_indices(units, per_talk)
        reasons = _selection_reasons(units, selected)
        for index in selected:
            unit = units[index]
            examples.append(
                {
                    "talk_id": document["talk_id"],
                    "alignment_id": unit["alignment_id"],
                    "source_segment_ids": unit["source_segment_ids"],
                    "target_segment_ids": unit["target_segment_ids"],
                    "source_text": unit["source_text"],
                    "target_text": unit["target_text"],
                    "current_alignment_type": f"{len(unit['source_segment_ids'])}:{len(unit['target_segment_ids'])}",
                    "current_cost": unit["score"],
                    "selection_reasons": sorted(reasons[index]),
                    "review": {
                        "verdict": None,
                        "preferred_source_ids": [],
                        "preferred_target_ids": [],
                        "note": "Requires human inspection.",
                    },
                }
            )
    return {
        "version": "1.0.0",
        "purpose": "Assistant-assisted alignment review approved by the researcher; examples must not enter the final test set.",
        "allowed_verdicts": ["correct", "questionable", "incorrect"],
        "examples": examples,
    }


def write_review_set(path: Path, document: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def write_alignment_review_tsv(path: Path, document: Mapping[str, Any]) -> None:
    """Write calibration units in document order for researcher-approved review."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=REVIEW_TSV_COLUMNS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for example in document["examples"]:
            review = example.get("review", {})
            writer.writerow(
                {
                    "talk_id": example["talk_id"],
                    "alignment_id": example["alignment_id"],
                    "alignment_type": example["current_alignment_type"],
                    "cost": example["current_cost"],
                    "selection_reasons": _json_cell(example["selection_reasons"]),
                    "source_segment_ids": _json_cell(example["source_segment_ids"]),
                    "source_text": example["source_text"],
                    "target_segment_ids": _json_cell(example["target_segment_ids"]),
                    "target_text": example["target_text"],
                    "verdict": review.get("verdict") or "",
                    "preferred_source_ids": _json_cell(review.get("preferred_source_ids", [])),
                    "preferred_target_ids": _json_cell(review.get("preferred_target_ids", [])),
                    "note": "",
                }
            )


def import_alignment_review_tsv(
    input_path: Path,
    review_path: Path,
    *,
    parsed_root: Path,
) -> int:
    """Apply validated researcher-approved review fields without altering selections."""
    document = json.loads(review_path.read_text(encoding="utf-8"))
    rows = _read_review_rows(input_path)
    examples = document.get("examples")
    if not isinstance(examples, list) or len(rows) != len(examples):
        raise ValueError("Review TSV must contain exactly one row for every calibration unit")

    allowed_verdicts = set(document.get("allowed_verdicts", []))
    updated_reviews: list[dict[str, Any]] = []
    segment_ids_by_talk: dict[str, tuple[set[str], set[str]]] = {}
    for example, row in zip(examples, rows, strict=True):
        _validate_immutable_review_row(example, row)
        verdict = row["verdict"].strip()
        if verdict and verdict not in allowed_verdicts:
            raise ValueError(f"Invalid verdict for {example['alignment_id']}: {verdict}")
        source_ids = _parse_ids(row["preferred_source_ids"], "preferred_source_ids", example["alignment_id"])
        target_ids = _parse_ids(row["preferred_target_ids"], "preferred_target_ids", example["alignment_id"])
        if source_ids or target_ids:
            talk_id = example["talk_id"]
            if talk_id not in segment_ids_by_talk:
                segment_ids_by_talk[talk_id] = _talk_segment_ids(parsed_root, talk_id)
            known_source, known_target = segment_ids_by_talk[talk_id]
            if not set(source_ids).issubset(known_source):
                raise ValueError(f"Unknown preferred source IDs for {example['alignment_id']}")
            if not set(target_ids).issubset(known_target):
                raise ValueError(f"Unknown preferred target IDs for {example['alignment_id']}")
        updated_reviews.append(
            {
                "verdict": verdict or None,
                "preferred_source_ids": source_ids,
                "preferred_target_ids": target_ids,
                "note": row["note"],
            }
        )

    for example, review in zip(examples, updated_reviews, strict=True):
        example["review"] = review
    write_review_set(review_path, document)
    return sum(review["verdict"] is not None for review in updated_reviews)


def _read_review_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        if tuple(reader.fieldnames or ()) != REVIEW_TSV_COLUMNS:
            raise ValueError("Review TSV columns do not match the expected schema")
        return list(reader)


def _validate_immutable_review_row(example: Mapping[str, Any], row: Mapping[str, str]) -> None:
    expected = {
        "talk_id": example["talk_id"],
        "alignment_id": example["alignment_id"],
        "alignment_type": example["current_alignment_type"],
        "cost": str(example["current_cost"]),
        "selection_reasons": _json_cell(example["selection_reasons"]),
        "source_segment_ids": _json_cell(example["source_segment_ids"]),
        "source_text": example["source_text"],
        "target_segment_ids": _json_cell(example["target_segment_ids"]),
        "target_text": example["target_text"],
    }
    for column, value in expected.items():
        if row[column] != value:
            raise ValueError(f"Calibration metadata changed for {example['alignment_id']}: {column}")


def _parse_ids(value: str, column: str, alignment_id: str) -> list[str]:
    if not value:
        return []
    try:
        identifiers = json.loads(value)
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid {column} for {alignment_id}; expected a JSON array") from error
    if not isinstance(identifiers, list) or not all(isinstance(identifier, str) for identifier in identifiers):
        raise ValueError(f"Invalid {column} for {alignment_id}; expected a JSON array of strings")
    return identifiers


def _talk_segment_ids(parsed_root: Path, talk_id: str) -> tuple[set[str], set[str]]:
    source = load_parsed_transcript(parsed_root / talk_id / "source.en.json")
    target = load_parsed_transcript(parsed_root / talk_id / "target.vi.json")
    return ({segment.segment_id for segment in source.segments}, {segment.segment_id for segment in target.segments})


def _json_cell(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def evaluate_grid(
    review_document: Mapping[str, Any],
    *,
    parsed_root: Path,
    candidates: Sequence[AlignmentParameters],
) -> dict[str, Any]:
    reviewed = [item for item in review_document.get("examples", []) if item.get("review", {}).get("verdict")]
    if not reviewed:
        return {
            "status": "blocked_pending_human_review",
            "reviewed_examples": 0,
            "candidates": [asdict(item) for item in candidates],
            "message": "No researcher-approved review verdicts are present; alignment cost is not used as correctness.",
        }
    by_talk: dict[str, list[Mapping[str, Any]]] = {}
    for item in reviewed:
        by_talk.setdefault(item["talk_id"], []).append(item)
    results: list[dict[str, Any]] = []
    for parameters in candidates:
        exact = source_covered = target_covered = incorrect_corrected = correct_preserved = correct_broken = 0
        skipped_source = skipped_target = grouped = long_range_drift = 0
        for talk_id, examples in by_talk.items():
            source_path = parsed_root / talk_id / "source.en.json"
            target_path = parsed_root / talk_id / "target.vi.json"
            aligned = align_transcripts(
                load_parsed_transcript(source_path),
                load_parsed_transcript(target_path),
                source_path=source_path,
                target_path=target_path,
                parameters=parameters,
            )
            boundaries = {(unit.source_segment_ids, unit.target_segment_ids) for unit in aligned.alignments}
            source_ids = {identifier for unit in aligned.alignments for identifier in unit.source_segment_ids}
            target_ids = {identifier for unit in aligned.alignments for identifier in unit.target_segment_ids}
            skipped_source += len(aligned.unaligned_source_segment_ids)
            skipped_target += len(aligned.unaligned_target_segment_ids)
            grouped += sum(
                len(unit.source_segment_ids) != 1 or len(unit.target_segment_ids) != 1
                for unit in aligned.alignments
            )
            source_order = {segment.segment_id: segment.index for segment in load_parsed_transcript(source_path).segments}
            target_order = {segment.segment_id: segment.index for segment in load_parsed_transcript(target_path).segments}
            long_range_drift += sum(
                abs(
                    (sum(source_order[item] for item in unit.source_segment_ids) / len(unit.source_segment_ids))
                    / max(1, len(source_order) - 1)
                    - (sum(target_order[item] for item in unit.target_segment_ids) / len(unit.target_segment_ids))
                    / max(1, len(target_order) - 1)
                ) > 0.1
                for unit in aligned.alignments
            )
            for example in examples:
                review = example["review"]
                if review["verdict"] == "questionable" and not (
                    review.get("preferred_source_ids") or review.get("preferred_target_ids")
                ):
                    continue
                preferred_source = tuple(review.get("preferred_source_ids") or example["source_segment_ids"])
                preferred_target = tuple(review.get("preferred_target_ids") or example["target_segment_ids"])
                matched = (preferred_source, preferred_target) in boundaries
                exact += int(matched)
                source_covered += sum(identifier in source_ids for identifier in preferred_source)
                target_covered += sum(identifier in target_ids for identifier in preferred_target)
                incorrect_corrected += int(review["verdict"] == "incorrect" and matched)
                correct_preserved += int(review["verdict"] == "correct" and matched)
                correct_broken += int(review["verdict"] == "correct" and not matched)
        results.append(
            {
                "parameters": asdict(parameters),
                "exact_reviewed_boundary_matches": exact,
                "reviewed_source_segment_coverage": source_covered,
                "reviewed_target_segment_coverage": target_covered,
                "confirmed_incorrect_cases_corrected": incorrect_corrected,
                "confirmed_correct_cases_preserved": correct_preserved,
                "confirmed_correct_cases_broken": correct_broken,
                "skip_behavior": {"unaligned_source_segments": skipped_source, "unaligned_target_segments": skipped_target},
                "grouped_alignment_behavior": {"grouped_units": grouped},
                "long_range_drift_units": long_range_drift,
            }
        )
    return {"status": "complete", "reviewed_examples": len(reviewed), "candidates": results}


def _sample_indices(units: Sequence[Mapping[str, Any]], limit: int) -> list[int]:
    if not units:
        return []
    anchors = set(range(min(4, len(units))))
    anchors.update(range(max(0, len(units) - 4), len(units)))
    middle = len(units) // 2
    anchors.update(range(max(0, middle - 2), min(len(units), middle + 3)))
    unusual = [index for index, unit in enumerate(units) if len(unit["source_segment_ids"]) != 1 or len(unit["target_segment_ids"]) != 1]
    annotations = [index for index, unit in enumerate(units) if unit["source_text"].lstrip().startswith(("(", "[")) or unit["target_text"].lstrip().startswith(("(", "["))]
    highest = sorted(range(len(units)), key=lambda index: (-units[index]["score"], index))
    for index in unusual + annotations + highest:
        if len(anchors) >= limit:
            break
        anchors.add(index)
    if len(anchors) < limit:
        step = max(1, len(units) // max(1, limit - len(anchors)))
        anchors.update(range(0, len(units), step))
    return sorted(anchors)[:limit]


def _selection_reasons(units: Sequence[Mapping[str, Any]], selected: Sequence[int]) -> dict[int, set[str]]:
    result = {index: set() for index in selected}
    highest = set(sorted(range(len(units)), key=lambda index: (-units[index]["score"], index))[:6])
    for index in selected:
        if index < 4:
            result[index].add("beginning")
        if index >= len(units) - 4:
            result[index].add("end")
        if abs(index - len(units) // 2) <= 2:
            result[index].add("middle")
        if index in highest:
            result[index].add("highest_cost")
        unit = units[index]
        alignment_type = f"{len(unit['source_segment_ids'])}:{len(unit['target_segment_ids'])}"
        result[index].add(alignment_type)
        if unit["source_text"].lstrip().startswith(("(", "[")) or unit["target_text"].lstrip().startswith(("(", "[")):
            result[index].add("annotation")
        if len(unit["source_text"].split()) <= 5:
            result[index].add("short_source")
    return result
