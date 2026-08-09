# Curated Talk Acquisition

## Scope

M0.2 provides a reproducible acquisition layer for curated English-Vietnamese technical and AI talks. It loads manually selected candidates, delegates provider access to a source adapter, stores lightly normalized raw artifacts, and appends an acquisition result for every attempt. These files are intermediate data, not canonical streaming talks.

M0.2 does not align English and Vietnamese, generate stream tokens, remove punctuation, simulate timing, invoke a translator, create LISTEN/COMMIT labels, train policies, or download video.

## Source Strategy

The initial `ted` adapter reads publicly accessible TED talk pages. It records page metadata and advertised languages, and stores transcripts only when they are embedded in public page JSON-LD. It uses an identifiable user agent, a 20-second default timeout, two bounded retries for transient failures, and a one-second minimum delay between requests. It does not authenticate, inspect protected media URLs, automate a browser, evade HTTP errors, or fetch video.

The adapter contract returns discovered availability, provider metadata, raw named artifacts, and warnings. Downstream writing and manifesting are provider-independent. A future IWSLT/WIT adapter can implement this same narrow contract and retain corpus talk IDs, caption structure, and metadata without changing artifact writing or later parsing.

TED page JSON-LD currently exposes transcript text but not subtitle blocks with timing. Accordingly, this adapter writes `source.en.txt` and `target.vi.txt` when available and records `subtitle_timing_available: false`. A corpus adapter may later preserve structured caption files with deterministic extensions.

## Candidate Manifest

`data/manifests/ted-ai-candidates.json` is version-controlled and contains nine candidates: four P0 and five P1 talks. Curated fields (`id`, `slug`, `title`, `speaker`, `domain`, `priority`, `provider`, and `source_url`) are separate from discovery output. The candidate list does not assert English or Vietnamese availability.

Candidate IDs must be unique and URL fields must be syntactically valid HTTP(S) URLs. Manifest loading fails before network access if these invariants do not hold.

## Commands

Acquire the Jeff Dean pilot by candidate ID:

```console
make acquire-data ARGS="--talk ted-jeff-dean-ai-smart"
```

The slug is also accepted by `--talk`. Acquire a priority group:

```console
make acquire-data ARGS="--priority P0"
```

Acquire every entry in a specific manifest:

```console
make acquire-data ARGS="--manifest data/manifests/ted-ai-candidates.json"
```

Existing `available`, `partial`, or `unavailable` attempts are reused by default. Pass `--force` to repeat discovery. A failed attempt may be retried normally. Network controls are exposed as `--timeout`, `--retries`, and `--request-delay`.

## Raw Layout

Artifacts use deterministic provider and candidate ID paths:

```text
data/streaming/raw/
└── ted/
    └── <candidate_id>/
        ├── metadata.json
        ├── source.en.txt       # when publicly available
        ├── target.vi.txt       # when publicly available
        └── acquisition.json

data/manifests/
├── ted-ai-candidates.json
└── acquisition-results.jsonl
```

`metadata.json` keeps curated metadata separate from provider-discovered metadata. `acquisition.json` is the latest per-talk result. `acquisition-results.jsonl` is an append-only run log and records candidate ID, provider, URL, UTC acquisition timestamp, status, EN/VI discovery, artifact paths, metadata, warnings, and a failure reason when applicable. Generated raw data and result logs are ignored by Git; the candidate manifest remains version-controlled.

Text is decoded and written as UTF-8 with normalized LF newlines. Transcript wording and punctuation are otherwise retained. Empty transcript artifacts are rejected.

## Status And Failure Behavior

`available` means both EN and VI artifacts were acquired. `partial` means one language was acquired. `unavailable` means the page was accessed but neither transcript was publicly exposed. `failed` means acquisition or writing raised an error. A failed candidate is logged and the batch continues; the CLI reports all results and exits nonzero after the batch if any attempts failed or artifact validation found errors.

Validation checks the candidate manifest, result status, referenced-file existence, UTF-8 readability, and non-empty transcript artifacts. This is raw-artifact validation only and is not the future canonical semantic validator.

## Source Use

Acquisition does not grant redistribution rights. Researchers must follow provider terms, dataset licenses, attribution requirements, and applicable law. Keep generated transcripts out of version control unless their license explicitly permits redistribution. If a provider rejects access, preserve the recorded failure and use an authorized source such as a licensed IWSLT/WIT release rather than attempting to bypass the restriction.
