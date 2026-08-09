"""Validation, checksums, and deterministic talk-level split assignment."""

from __future__ import annotations

import copy
import hashlib
import json
import random
from collections import defaultdict
from typing import Any, Mapping


SCHEMA_VERSION = "1.0.0"
EXPERIMENTAL_SPLITS = ("train", "dev", "test")
GROUP_BY_VALUES = ("talk", "speaker")


def serialize_json(document: Mapping[str, Any]) -> str:
    return json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def dataset_manifest_checksum(document: Mapping[str, Any]) -> str:
    """Hash manifest content while excluding its intentionally volatile timestamp."""
    stable = copy.deepcopy(document)
    provenance = stable.get("provenance")
    if isinstance(provenance, dict):
        provenance.pop("generated_at", None)
    payload = json.dumps(stable, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def validate_dataset_manifest(document: Mapping[str, Any]) -> None:
    if set(document) != {"schema_version", "dataset", "talks", "provenance"}:
        raise ValueError("Dataset manifest must contain exactly schema_version, dataset, talks, and provenance")
    if document.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported dataset manifest schema version")
    dataset = _mapping(document.get("dataset"), "dataset")
    if set(dataset) != {"name", "source_language", "target_language", "status"} or not isinstance(dataset.get("name"), str) or not dataset["name"]:
        raise ValueError("Dataset manifest requires a dataset name")
    if dataset.get("source_language") != "en" or dataset.get("target_language") != "vi":
        raise ValueError("Dataset manifest languages must be en to vi")
    if dataset.get("status") not in {"pilot", "unassigned"}:
        raise ValueError("Dataset manifest status must be pilot or unassigned")
    talks = document.get("talks")
    if not isinstance(talks, list):
        raise ValueError("Dataset manifest talks must be an array")
    talk_ids: set[str] = set()
    previous_id = ""
    for talk in talks:
        talk = _mapping(talk, "manifest talk")
        required = {"talk_id", "canonical_path", "content_checksum", "timing_mode", "statistics", "cohort"}
        if not required.issubset(talk) or set(talk) - {"talk_id", "canonical_path", "content_checksum", "speaker", "domain", "provider", "topics", "timing_mode", "statistics", "cohort"}:
            raise ValueError("Dataset manifest talk has unsupported or missing fields")
        talk_id = talk.get("talk_id")
        if not isinstance(talk_id, str) or not talk_id or talk_id in talk_ids or talk_id <= previous_id:
            raise ValueError("Dataset manifest talk IDs must be unique and ordered")
        if not isinstance(talk.get("canonical_path"), str) or not talk["canonical_path"].endswith("/streaming-talk.json"):
            raise ValueError(f"Dataset manifest canonical path is invalid for {talk_id}")
        if not _sha256(talk.get("content_checksum")):
            raise ValueError(f"Dataset manifest content checksum is invalid for {talk_id}")
        if talk.get("timing_mode") not in {"simulated", "recovered_from_caption_starts"}:
            raise ValueError(f"Dataset manifest timing mode is invalid for {talk_id}")
        if talk.get("cohort") not in {"pilot", "unassigned"}:
            raise ValueError(f"Dataset manifest cohort is invalid for {talk_id}")
        statistics = _mapping(talk.get("statistics"), "talk statistics")
        if set(statistics) != {"source_segments", "target_segments", "alignment_units", "stream_tokens", "source_clock_duration_ms"} or any(not isinstance(value, int) or value < 0 for value in statistics.values()):
            raise ValueError(f"Dataset manifest statistics are invalid for {talk_id}")
        talk_ids.add(talk_id)
        previous_id = talk_id
    provenance = _mapping(document.get("provenance"), "provenance")
    if not isinstance(provenance.get("builder"), str) or not provenance["builder"]:
        raise ValueError("Dataset manifest provenance requires a builder")


def validate_split_manifest(document: Mapping[str, Any], dataset_manifest: Mapping[str, Any]) -> None:
    validate_dataset_manifest(dataset_manifest)
    if document.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported dataset split schema version")
    if document.get("dataset_manifest_checksum") != dataset_manifest_checksum(dataset_manifest):
        raise ValueError("Dataset manifest checksum mismatch")
    manifest_ids = {talk["talk_id"] for talk in dataset_manifest["talks"]}
    split_type = document.get("split_type")
    if split_type == "pilot":
        if set(document) != {"schema_version", "split_type", "dataset_manifest_checksum", "talk_ids"}:
            raise ValueError("Pilot split manifest has unsupported or missing fields")
        _validate_ids(document.get("talk_ids"), manifest_ids, "pilot")
        return
    if split_type != "experimental":
        raise ValueError("Split manifest type must be pilot or experimental")
    if set(document) != {"schema_version", "split_type", "dataset_manifest_checksum", "strategy", "splits"}:
        raise ValueError("Experimental split manifest has unsupported or missing fields")
    strategy = _mapping(document.get("strategy"), "split strategy")
    if set(strategy) != {"name", "seed", "group_by", "ratios", "test_exclusions"} or strategy.get("name") != "seeded_sorted_groups" or not isinstance(strategy.get("seed"), int) or strategy.get("group_by") not in GROUP_BY_VALUES:
        raise ValueError("Experimental split strategy is invalid")
    _validate_ratios(strategy.get("ratios"))
    test_exclusions = _validate_ids(strategy.get("test_exclusions"), manifest_ids, "test_exclusions")
    splits = _mapping(document.get("splits"), "splits")
    if set(splits) != set(EXPERIMENTAL_SPLITS):
        raise ValueError("Experimental split names must be train, dev, and test")
    assigned: set[str] = set()
    talk_to_split: dict[str, str] = {}
    for name in EXPERIMENTAL_SPLITS:
        ids = _validate_ids(splits.get(name), manifest_ids, name)
        overlap = assigned.intersection(ids)
        if overlap:
            raise ValueError(f"Talk IDs occur in multiple experimental splits: {sorted(overlap)}")
        assigned.update(ids)
        talk_to_split.update({talk_id: name for talk_id in ids})
    if assigned != manifest_ids:
        raise ValueError("Experimental split union must include every dataset talk exactly once")
    contamination = set(test_exclusions).intersection(splits["test"])
    if contamination:
        raise ValueError(f"Calibration talks occur in test: {sorted(contamination)}")
    if strategy["group_by"] == "speaker":
        speakers: dict[str, str] = {}
        for talk in dataset_manifest["talks"]:
            speaker = talk.get("speaker")
            if not isinstance(speaker, str) or not speaker:
                raise ValueError("Speaker grouping requires speaker metadata for every talk")
            previous = speakers.setdefault(speaker, talk_to_split[talk["talk_id"]])
            if previous != talk_to_split[talk["talk_id"]]:
                raise ValueError(f"Speaker group crosses experimental splits: {speaker}")


def build_experimental_split(dataset_manifest: Mapping[str, Any], *, seed: int, train_ratio: float, dev_ratio: float, test_ratio: float, group_by: str = "speaker", minimum_talk_count: int = 10, allow_tiny_dataset: bool = False, test_exclusions: list[str] | None = None) -> dict[str, Any]:
    validate_dataset_manifest(dataset_manifest)
    talks = dataset_manifest["talks"]
    if len(talks) < minimum_talk_count and not allow_tiny_dataset:
        raise ValueError(f"Dataset contains only {len(talks)} talks; refusing to generate an experimental split. Pass --allow-tiny-dataset to override.")
    ratios = {"train": train_ratio, "dev": dev_ratio, "test": test_ratio}
    _validate_ratios(ratios)
    if group_by not in GROUP_BY_VALUES:
        raise ValueError("group_by must be talk or speaker")
    groups: dict[str, list[str]] = defaultdict(list)
    for talk in talks:
        if group_by == "speaker":
            speaker = talk.get("speaker")
            if not isinstance(speaker, str) or not speaker:
                raise ValueError("Speaker grouping requires speaker metadata for every talk")
            key = speaker
        else:
            key = talk["talk_id"]
        groups[key].append(talk["talk_id"])
    group_keys = sorted(groups)
    random.Random(seed).shuffle(group_keys)
    counts = _allocation_counts(len(group_keys), ratios)
    exclusions = set(test_exclusions or [])
    unknown_exclusions = exclusions - {talk["talk_id"] for talk in talks}
    if unknown_exclusions:
        raise ValueError(f"Test exclusions contain unknown talk IDs: {sorted(unknown_exclusions)}")
    excluded_groups = {key for key, talk_ids in groups.items() if exclusions.intersection(talk_ids)}
    eligible_test = [key for key in group_keys if key not in excluded_groups]
    if len(eligible_test) < counts["test"]:
        raise ValueError("Not enough non-calibration speaker groups for the requested test split")
    test_groups = eligible_test[:counts["test"]]
    remaining = [key for key in group_keys if key not in test_groups]
    dev_groups = remaining[:counts["dev"]]
    train_groups = remaining[counts["dev"]:]
    splits = {name: [] for name in EXPERIMENTAL_SPLITS}
    assignments = {"train": train_groups, "dev": dev_groups, "test": test_groups}
    for name in EXPERIMENTAL_SPLITS:
        for key in assignments[name]:
            splits[name].extend(sorted(groups[key]))
        splits[name].sort()
    result = {"schema_version": SCHEMA_VERSION, "split_type": "experimental", "dataset_manifest_checksum": dataset_manifest_checksum(dataset_manifest), "strategy": {"name": "seeded_sorted_groups", "seed": seed, "group_by": group_by, "ratios": ratios, "test_exclusions": sorted(exclusions)}, "splits": splits}
    validate_split_manifest(result, dataset_manifest)
    return result


def lookup_split_for_talk(split_manifest: Mapping[str, Any], talk_id: str) -> str:
    """Return the externally assigned split inherited by every derived record."""
    if split_manifest.get("split_type") == "pilot":
        if talk_id in split_manifest.get("talk_ids", []):
            return "pilot"
    elif split_manifest.get("split_type") == "experimental":
        for split, talk_ids in split_manifest.get("splits", {}).items():
            if talk_id in talk_ids:
                return split
    raise ValueError(f"Talk ID is not assigned by split manifest: {talk_id}")


def _allocation_counts(total: int, ratios: Mapping[str, float]) -> dict[str, int]:
    raw = {name: total * ratios[name] for name in EXPERIMENTAL_SPLITS}
    counts = {name: int(raw[name]) for name in EXPERIMENTAL_SPLITS}
    for name in sorted(EXPERIMENTAL_SPLITS, key=lambda item: (raw[item] - counts[item], -EXPERIMENTAL_SPLITS.index(item)), reverse=True)[:total - sum(counts.values())]:
        counts[name] += 1
    return counts


def _validate_ratios(value: Any) -> None:
    if not isinstance(value, Mapping) or set(value) != set(EXPERIMENTAL_SPLITS) or any(not isinstance(value[name], (int, float)) or isinstance(value[name], bool) or value[name] <= 0 for name in EXPERIMENTAL_SPLITS) or abs(sum(value.values()) - 1.0) > 1e-9:
        raise ValueError("Split ratios must be positive train/dev/test values summing to 1.0")


def _validate_ids(value: Any, manifest_ids: set[str], split_name: str) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(talk_id, str) for talk_id in value) or len(value) != len(set(value)):
        raise ValueError(f"Split {split_name} must contain unique talk IDs")
    unknown = set(value) - manifest_ids
    if unknown:
        raise ValueError(f"Split {split_name} contains unknown talk IDs: {sorted(unknown)}")
    return value


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value)
