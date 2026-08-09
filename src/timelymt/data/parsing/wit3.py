"""Parser for independently distributed WIT3/IWSLT seekvideo captions."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET

from .core import (
    ParsedTranscript,
    make_provenance,
    make_segments,
    normalize_text,
    validate_parsed_transcript,
)


class Wit3CaptionParser:
    provider = "wit3"
    name = "Wit3CaptionParser"
    segmentation_method = "wit3_seekvideo_captions"

    def parse(self, input_path: Path, *, talk_id: str | None, language: str) -> ParsedTranscript:
        try:
            raw_bytes = input_path.read_bytes()
            root = ET.fromstring(raw_bytes)
        except (OSError, ET.ParseError) as error:
            raise ValueError(f"Cannot parse WIT3 XML {input_path}: {error}") from error

        entries = _talk_entries(root)
        if not entries:
            raise ValueError("WIT3 XML contains no transcription entries")
        identified = [(_talk_id(entry), entry, transcription) for entry, transcription in entries]
        if any(identifier is None for identifier, _, _ in identified):
            raise ValueError("WIT3 transcription entry is missing a talk ID")
        if talk_id is None:
            if len(identified) != 1:
                raise ValueError("WIT3 XML contains multiple talks; specify talk_id")
            selected_id, entry, transcription = identified[0]
        else:
            matches = [item for item in identified if item[0] == talk_id]
            if not matches:
                raise ValueError(f"Talk ID {talk_id!r} not found in WIT3 XML")
            if len(matches) > 1:
                raise ValueError(f"Talk ID {talk_id!r} is ambiguous in WIT3 XML")
            selected_id, entry, transcription = matches[0]

        texts: list[str] = []
        starts: list[int | None] = []
        for caption in transcription:
            if _local_name(caption.tag) != "seekvideo":
                continue
            raw_id = caption.get("id")
            try:
                start_ms = int(raw_id) if raw_id is not None else None
            except ValueError as error:
                raise ValueError(f"Malformed seekvideo id: {raw_id!r}") from error
            if start_ms is None or start_ms < 0:
                raise ValueError(f"Malformed seekvideo id: {raw_id!r}")
            text = normalize_text("".join(caption.itertext()))
            if not text:
                raise ValueError(f"Empty seekvideo caption at {start_ms} ms")
            texts.append(text)
            starts.append(start_ms)
        if not texts:
            raise ValueError(f"Talk {selected_id!r} contains no seekvideo captions")

        metadata = _entry_metadata(entry)
        transcript = ParsedTranscript(
            talk_id=selected_id or "",
            language=language,
            provider=self.provider,
            segmentation_method=self.segmentation_method,
            segments=make_segments(texts, language, starts_ms=starts, timing_source="wit3_seekvideo"),
            provenance=make_provenance(input_path, raw_bytes, self.name, source_metadata=metadata),
        )
        validate_parsed_transcript(transcript)
        return transcript


def _talk_entries(root: ET.Element) -> list[tuple[ET.Element, ET.Element]]:
    entries: list[tuple[ET.Element, ET.Element]] = []
    containers = {"file", "talk", "document"}

    def visit(element: ET.Element, owner: ET.Element | None) -> None:
        if _local_name(element.tag) in containers:
            owner = element
        if _local_name(element.tag) == "transcription" and owner is not None:
            entries.append((owner, element))
            return
        for child in element:
            visit(child, owner)

    visit(root, None)
    if entries:
        return entries

    # Some small exports omit a file wrapper and expose transcription directly.
    for element in root.iter():
        for child in element:
            if _local_name(child.tag) == "transcription":
                entries.append((element, child))
    if _local_name(root.tag) == "transcription":
        entries.append((root, root))
    return entries


def _talk_id(entry: ET.Element) -> str | None:
    for element in entry.iter():
        if _local_name(element.tag) in {"talkid", "talk_id"}:
            value = normalize_text("".join(element.itertext()))
            if value:
                return value
    for key in ("talkid", "talk_id", "id"):
        value = entry.get(key)
        if value and normalize_text(value):
            return normalize_text(value)
    return None


def _entry_metadata(entry: ET.Element) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    for element in entry.iter():
        name = _local_name(element.tag)
        if name in {"title", "speaker", "url"}:
            value = normalize_text("".join(element.itertext()))
            if value:
                metadata[name] = value
    return metadata


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1].lower()
