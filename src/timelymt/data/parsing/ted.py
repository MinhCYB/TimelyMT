"""Parser for untimed continuous transcript text acquired from TED pages."""

from __future__ import annotations

from pathlib import Path
import re

from .core import (
    ParsedTranscript,
    make_provenance,
    make_segments,
    normalize_text,
    validate_parsed_transcript,
)


_BOUNDARY = re.compile(r"(?P<end>[.!?]+[\"'”’)]*)\s+(?=[A-ZÀ-Ỹ0-9\"'“‘(])")
_NON_TERMINAL_ABBREVIATIONS = {
    "dr.",
    "e.g.",
    "etc.",
    "i.e.",
    "mr.",
    "mrs.",
    "ms.",
    "prof.",
    "st.",
    "vs.",
}


class TedContinuousTranscriptParser:
    provider = "ted"
    name = "TedContinuousTranscriptParser"
    segmentation_method = "ted_continuous_sentence_heuristic"

    def parse(self, input_path: Path, *, talk_id: str | None, language: str) -> ParsedTranscript:
        if not talk_id:
            raise ValueError("TED parsing requires a talk_id")
        try:
            raw_bytes = input_path.read_bytes()
            raw_text = raw_bytes.decode("utf-8")
        except (OSError, UnicodeError) as error:
            raise ValueError(f"Cannot read UTF-8 transcript {input_path}: {error}") from error

        paragraphs = re.split(r"\n\s*\n", raw_text.replace("\r\n", "\n").replace("\r", "\n"))
        texts: list[str] = []
        for paragraph in paragraphs:
            normalized = normalize_text(paragraph)
            if normalized:
                texts.extend(_split_sentences(normalized))
        if not texts:
            raise ValueError(f"Empty TED transcript rejected: {input_path}")

        transcript = ParsedTranscript(
            talk_id=talk_id,
            language=language,
            provider=self.provider,
            segmentation_method=self.segmentation_method,
            segments=make_segments(texts, language, timing_source="none"),
            provenance=make_provenance(input_path, raw_bytes, self.name),
        )
        validate_parsed_transcript(transcript)
        return transcript


def _split_sentences(paragraph: str) -> list[str]:
    sentences: list[str] = []
    start = 0
    for match in _BOUNDARY.finditer(paragraph):
        candidate = paragraph[start : match.end("end")]
        final_word = candidate.lower().rsplit(maxsplit=1)[-1].strip("\"'“”‘’()")
        if final_word in _NON_TERMINAL_ABBREVIATIONS or re.fullmatch(r"[a-z]\.", final_word):
            continue
        sentences.append(candidate.strip())
        start = match.end()
    remainder = paragraph[start:].strip()
    if remainder:
        sentences.append(remainder)
    return sentences
