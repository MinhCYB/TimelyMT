# TimelyMT Dataset Report Notes

## Dataset Identity

**TimelyMT Streaming Dataset v1** is a technical, AI-focused English-to-Vietnamese corpus of 17 accepted TED talks. It supports research on streaming interpretation timing while preserving source-stream causality and keeping Vietnamese references separate from runtime-visible input.

| Field | Value |
| --- | --- |
| Languages | English -> Vietnamese |
| Accepted talks | 17 |
| Provider | TED |
| Scope | AI, ML, NLP, robotics, computer science, and related technology talks |

## Construction Pipeline

`acquisition -> parsing -> EN-VI alignment -> source timing simulation/recovery -> canonical streaming-talk -> dataset manifest -> talk-level split`

Each accepted talk passed acquisition, parsing, alignment with the frozen configuration, timing construction, canonical build, and semantic validation.

## Alignment Methodology

The dataset uses deterministic monotonic EN-VI segment alignment with bounded N:M grouping.

| Frozen parameter | Value |
| --- | ---: |
| `max_group_size` | 4 |
| `skip_penalty` | 1.6 |
| `group_penalty` | 0.65 |
| Calibration set | 75 reviewed units |
| Review result | 72 correct / 3 incorrect |
| Final calibration result | 74/74 reviewed boundaries matched |

Calibration used an **assistant-assisted alignment review approved by the researcher**. Alignment cost is a structural optimization diagnostic, not confidence.

## Dataset Statistics

| Measure | Value |
| --- | ---: |
| Source segments | 2,471 |
| Target segments | 2,372 |
| Alignment units | 2,323 |
| Lexical streaming tokens | 41,739 |
| Source-clock duration | 16,695,600 ms |
| Provider distribution | TED: 17 |
| Timing mode distribution | simulated: 17 |

## Split

| Partition | Talks | Share |
| --- | ---: | ---: |
| Train | 12 | 70.6% |
| Dev | 3 | 17.6% |
| Test | 2 | 11.8% |

The persisted split is speaker-aware and talk-level. It has no speaker leakage. The calibration talks (Alona Fyshe, Jeff Dean, and Yejin Choi) are explicitly excluded from test.

## Reproducibility

| Artifact | Path / checksum |
| --- | --- |
| Frozen alignment configuration | `configs/data/alignment.json` |
| Dataset manifest | `data/manifests/streaming-dataset.json` |
| Dataset manifest checksum | `6730be08eff2ea874aad693e195ff05488a9b2222902f23e6e83c88e3afb2cce` |
| Experimental split | `data/splits/experimental.json` |
| Split checksum | `aabc06af1836e5d66a69d3b0305f6044892cbe0d3e45883ee7aeed53edd3ddc4` |
| Dataset v1 snapshot | `data/manifests/timelymt-streaming-dataset-v1.json` |
| Dataset v1 snapshot checksum | `a846600eea3275900f3698091983049981499969def750f036a213d1fc23f2de` |

## Methodological Safeguards

- The target reference is never runtime-visible.
- Future source tokens are not runtime-visible.
- Split membership is inherited from `talk_id`.
- There is no sentence-level or prefix-level random resplitting.
- Dataset v1 must not be silently changed based on later model results.

## Known Limitations

- TED timing is simulated source-clock timing, not acoustic word alignment.
- Deterministic alignment has no bilingual semantic model.
- All accepted talks currently come from one provider.
- Some grouped or short boundaries have high diagnostic alignment costs.

## Reporting Terminology

Use these phrases consistently in the research report:

- "simulated source-clock latency"
- "assistant-assisted alignment review approved by the researcher"
- "deterministic monotonic EN-VI segment alignment"
- "speaker-aware talk-level split"
