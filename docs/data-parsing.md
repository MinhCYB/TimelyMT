# Transcript And Caption Parsing

## Purpose

M0.3 converts one raw language artifact for one talk into a provider-neutral sequence of normalized monolingual segments. The machine-readable intermediate contract is [`schemas/parsed-transcript.schema.json`](../schemas/parsed-transcript.schema.json), using JSON Schema Draft 2020-12 and schema version `1.0.0`.

M0.3 parsing is not bilingual alignment. English and Vietnamese are parsed and stored independently even when both are available for the same talk. M0.4 may later establish 1:1, 1:N, N:1, or N:M relationships without changing either parsed sequence.

## Data Lifecycle

```text
data/streaming/raw/       provider-specific acquired artifacts
data/streaming/parsed/    normalized monolingual transcript artifacts (M0.3)
data/streaming/aligned/   bilingual alignment artifacts (M0.4)
data/streaming/timed/     source-only token timing artifacts (M0.5)
data/streaming/processed/ fully canonical streaming talks
```

Parsed files must not be placed under `processed/` or claimed to conform to the canonical streaming-talk schema. Generated raw and parsed artifacts remain ignored by Git; only their stage markers are version-controlled.

The conventional output for a bilingual talk is:

```text
data/streaming/parsed/<talk_id>/
├── source.en.json
└── target.vi.json
```

The two files are separate parser outputs, not a paired or zipped representation.

## Parsed Structure

Each file contains `schema_version`, `talk_id`, `language`, `provider`, `segmentation`, ordered `segments`, and `provenance`. Each segment contains a deterministic language-prefixed `segment_id`, contiguous zero-based `index`, preserved `text`, nullable `start_ms` and `end_ms`, and a narrow `timing_source` value.

Semantic validation guarantees non-empty supported identifiers and text, unique deterministic IDs, contiguous indices, non-negative integer timestamps when present, and `end_ms >= start_ms` when both exist. JSON output is UTF-8 with LF newlines. The schema enforces the provider-neutral shape; Python validation enforces cross-item invariants that JSON Schema cannot reliably express.

Provenance records the raw input path, SHA-256 checksum of the input bytes, parser name and version, UTC processing timestamp, and small source metadata when available. Re-running a parser preserves segment content, ordering, indices, and IDs; only the processing timestamp may change.

## TED Continuous Text

The TED parser consumes the M0.2 `source.en.txt` or `target.vi.txt` artifact. It preserves blank-line paragraph boundaries, normalizes line endings and repeated formatting whitespace, then splits within each paragraph only after terminal `.`, `!`, or `?` punctuation when the following text looks like a new sentence. Punctuation remains attached to reference text. Commas and semicolons never cause a split, and a small list of common abbreviations avoids obvious false boundaries.

The recorded method is `ted_continuous_sentence_heuristic`. Because the acquired TED page transcript is continuous text, this is intentionally a lightweight reference-unit heuristic rather than recovery of original captions. It can mis-handle unusual abbreviations, initials, quoted dialogue, punctuation-free sentence transitions, and language-specific sentence conventions. It does not use an NLP model.

TED segments have `start_ms: null`, `end_ms: null`, and `timing_source: "none"`. M0.3 does not invent timing.

Parse both languages of one acquired TED talk:

```console
make parse-data ARGS="--provider ted --talk ted-jeff-dean-ai-smart"
```

Pass `--language en` or `--language vi` to parse only one available artifact.

## WIT3 Captions

The WIT3 parser consumes a local XML file using the Python standard library. It finds transcription-bearing talk entries, identifies the requested `talk_id`, reads `seekvideo` children in XML order, parses each `id` as non-negative integer milliseconds, and emits one segment per caption. Small identifying fields such as title, speaker, and URL are retained in provenance when present.

Caption punctuation and annotations such as `(Music)` and `(Applause)` are preserved. A `seekvideo id="7555"` produces `start_ms: 7555`, `end_ms: null`, and `timing_source: "wit3_seekvideo"`. End times are neither inferred from the next caption nor estimated from text.

Parse one language from a local WIT3 XML file:

```console
make parse-data ARGS="--provider wit3 --input path/to/file.xml --talk-id 1903 --language en"
```

Run the command separately for English and Vietnamese XML. The parser never zips captions across languages and never assumes equal caption counts or boundaries.

## Normalization

Parsing decodes standard XML/HTML entities, normalizes CRLF/CR to LF, trims outer whitespace, and collapses repeated formatting whitespace inside a segment. It preserves Unicode, Vietnamese accents, case, punctuation, terminology, parenthetical markers, music/applause annotations, and linguistic wording.

Parsing does not lowercase, translate, remove punctuation or accents, normalize terminology, tokenize for models, or filter transcript annotations.

## Explicit Non-Goals

M0.3 does not implement bilingual alignment, similarity scoring, embedding or dynamic-programming alignment, canonical streaming-talk construction, stream-token generation, simulated or recovered timing, WIT3 `end_ms` inference, translator inference, commit baselines, pseudo-labels, policy models, evaluation, or demos. Those concerns belong to later milestones.
