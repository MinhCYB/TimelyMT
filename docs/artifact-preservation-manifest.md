# TimelyMT Artifact Preservation Manifest

Created: 2026-08-19  
Authoritative audit: `docs/repository-audit.md`  
Machine-readable manifest: `docs/artifact-preservation-manifest.json`

## Purpose

This manifest records the frozen scientific artifacts and source-of-truth inputs that were validated before TimelyMT Cleanup Phase A. It is a preservation index, not authorization to move, deduplicate, regenerate, evaluate, or delete any listed artifact.

TEST remains **PROTECTED - DO NOT TOUCH**. TEST entries below preserve only path/existence metadata already permitted by the repository audit. TEST translations/references were not opened, parsed, summarized, evaluated, or hashed.

## Hash Method

- File hashes are SHA-256 over raw file bytes.
- Directory tree hashes are SHA-256 over UTF-8, newline-joined records sorted by relative path. Each record is `relative-path<TAB>lowercase-file-sha256`.
- The canonical-talk tree hash excludes both protected TEST talk directories. Their paths are recorded without content-derived metadata.

## Frozen Identities

| Identity | Value | Validation |
|---|---|---|
| Dataset manifest checksum | `6730be08eff2ea874aad693e195ff05488a9b2222902f23e6e83c88e3afb2cce` | Documented by Dataset v1 metadata |
| Split manifest checksum | `aabc06af1836e5d66a69d3b0305f6044892cbe0d3e45883ee7aeed53edd3ddc4` | Documented by Dataset v1/V1/V2 metadata |
| Translator fingerprint | `a54ba8356642a7a696234453b3fc0a29d2dcf85db5299677c492ae967281bd1c` | Documented by V1/V2 metadata |
| Prepared-context fingerprint | `d9b910afd1941873826065bcf6e343be28cd850d339b356457daadbde60ad2eb` | Matches `data/prepared_context/manifest.json` file SHA-256 |
| P3 checkpoint SHA-256 | `ccf829fdb7ab521cc12c299583efa7222c965440b1257ddfb35e03ddd7bcadb9` | Recalculated and matched exactly |
| V1 frozen commit | `6c75da5d60cc626ab79e7e82cae471e18be27531` | V1 checkpoint metadata |
| P3 publication commit | `e836f7ac5658f22de3d907289671fcd0caf2b42d` | Kaggle publication metadata |

## Dataset and Supervision

| Path | Category / stage | Size | SHA-256 or tree SHA-256 | Git state | Producer / consumers | Preservation role |
|---|---|---:|---|---|---|---|
| `data/manifests/ted-ai-candidates.json` | SOURCE-OF-TRUTH / acquisition | 8,605 B | `804063336361c446e8e2c297512bfd89f5cb35839938cf64abfb674effe1bb35` | tracked | Manual curation -> acquisition/pipeline | External source scope |
| `data/manifests/streaming-dataset.json` | SOURCE-OF-TRUTH / Dataset v1 | 10,723 B | `c423c856c5c5c3514b18f7cc3777ef1bdb15aba7778128e02e22fa0cbb88715a` | tracked | Manifest builder -> all research | Canonical 17-talk index |
| `data/manifests/timelymt-streaming-dataset-v1.json` | REQUIRED-FROZEN-ARTIFACT | 1,044 B | `f958eafb16656af891cf4210276023ce68b70cf98d36cd4c00d04a253d9b4426` | tracked | Dataset freeze -> V1/V2/P3/TEST gate | Immutable dataset identity |
| `data/splits/experimental.json` | REQUIRED-FROZEN-ARTIFACT / split | 1,236 B | `3ee6ce479806e2f5db909d8f7ed871dc027eea53368752b52e88f2dc9b39455a` | tracked | Split process -> all research | Frozen TRAIN/DEV/TEST membership |
| `data/streaming/processed/` | REQUIRED-FROZEN-ARTIFACT / canonical talks | 9,610,522 B in non-TEST hash scope | tree `1c522cac5471804f97d8a0639f623728685ab89042d41ddaa10834bc26f677aa` | tracked | Canonical builder -> manifests/research | 15 TRAIN/DEV talks hashed; TEST excluded and protected |
| `data/review/` | SOURCE-OF-TRUTH / human review | 100,064 B | tree `6e45672ebe3fbb5fe54ab0093ffa4cbfe73c8a9be23c0d0b6bf03bdfbbc7968d` | tracked | Calibration + human review -> quality report | Non-regenerable review evidence |
| `outputs/dataset/` | REQUIRED-FROZEN-ARTIFACT / QA | 119,563 B | tree `ab5a11077b75f83a264f1d4f315697c26d2708a8e9a2cf455cd481dd2dd7cb95` | ignored | Pipeline QA -> dataset report | Machine-readable QA evidence |
| `data/policy/pseudo_labels/train/` | REQUIRED-FROZEN-ARTIFACT / V1 TRAIN | 26,871,299 B | tree `1cdd1794ff715dea9f1030ddd1fe47f5fba6b0abc02bcbaa0c76b2a31dcbe82a` | ignored | V1 pseudo labeling -> V1/V2/P3 training | Immutable upstream supervision |
| `data/policy/pseudo_labels/dev/` | REQUIRED-FROZEN-ARTIFACT / V1 DEV | 6,980,289 B | tree `c2f051379aa31a8457a7db7b8a07395f5eab11e2fe791fa994c9785a7b4a30cf` | ignored | V1 pseudo labeling -> V1/V2 DEV | Frozen DEV supervision |
| `docs/archive/timelymt-checkpoint/` | REQUIRED-FROZEN-ARTIFACT / V1 dev-frozen-complete | 90,421,625 B | tree `7db12d115496818601a96b9c4881f71775c5e21de457352dadf66a02573323d9` | tracked | V1 Kaggle export -> V2 import/TEST gate | Complete authoritative V1 package |
| `docs/archive/checkpoint-summary.json` | PROVENANCE / V1 archive | 1,127 B | `57e0eeb8f6174db20d1cb32b3bfbbe8cc6ff43a102df34bf67ed1f783087ad5b` | tracked | V1 export | Records original archive identity and SHA |

The active V1 TRAIN/DEV supervision has exact file-level duplicates in the expanded V1 archive. Both path contracts remain protected; no deduplication is authorized.

## V2 Artifacts

| Path | Stage | Size | SHA-256 or tree SHA-256 | Git state | Preservation reason |
|---|---|---:|---|---|---|
| `checkpoints/policy_v2/V2P0.pt` | V2 P0 full | 474,949 B | `f7f0c58d7ab4d3ec662aebc697a385c9747ab42ad0df77676aab7c962e03d299` | tracked | Frozen exploratory checkpoint |
| `checkpoints/policy_v2/V2P1.pt` | V2 P1 full | 868,165 B | `e1102f6c5245949a46335d61f9a054c1c9dbbdfc235f6946de1a5f3228413fd3` | tracked | Frozen exploratory checkpoint |
| `checkpoints/policy_v2/V2P2.pt` | V2 P2 full/selected family | 1,261,381 B | `4d531caf165175a4c8b5ef00b54ad09ef7effb3b5f453f0d3f28e1480263fbe7` | tracked | Direct P3-vs-P2 report baseline |
| `checkpoints/policy_v2/*.metadata.json` | V2 metadata | 18,417 B total | Per-file hashes in JSON manifest | tracked | Training/runtime/checkpoint identities |
| `outputs/experiments/policy-v2/v2-frozen-config.json` | V2 DEV frozen | 2,409 B | `baaa099242f3ebccf7b546a308af41899479df569b4f5fe4c5a36bf656430192` | ignored | Exact frozen run config |
| `outputs/experiments/policy-v2/predictions/` | V2 DEV | 21,049,852 B / 45 files | tree `43dd5f7d1c780f50412227dffe6dcf78471d51efb77b9d4693b43bb5ab11f4ae` | ignored | Input to metrics and P3 comparison |
| `outputs/experiments/policy-v2/metrics/` | V2 DEV | 59,180 B / 16 files | tree `8880d097bf3775f57ebb5a8ed8c5680f73dff38d8146dd9099681e1604ba5a5a` | ignored | Published quality/latency evidence |
| `outputs/experiments/policy-v2/dev-selection.json` | V2 selection | 2,634 B | `564f4b3960f4c46e6692a9ee49b4b4ab6792f56a3e3444aff2e87387da72dbcb` | ignored | Historical `v2_P2_0.50` selection |
| `outputs/experiments/policy-v2/run-provenance.json` | V2 provenance | 495 B | `9ec18121fe3e3558fb069543a0d3380f364a4c98a821d277dbe9d9fa3c6b4ff8` | ignored | Execution identity |
| `outputs/experiments/policy-v2/comparison-v1-v2.json` | V1/V2 comparison | 62,621 B | `dd16b810a55a9abb29373b3b823468c84549f5470b43706a6ff6f14d49594834` | ignored | Machine-readable final interpretation evidence |

V2 producer/origin: frozen multilingual MiniLM revision `e62509716f15c5fd03a6fd3156a4bc5e43f83f26` plus MLP trained from immutable V1 supervision. Known consumers include V2 selection, P3 full DEV analysis, final reports, and the P2 comparison path.

## Prepared Context and P3

| Path | Category / stage | Size | SHA-256 or tree SHA-256 | Git state | Preservation reason |
|---|---|---:|---|---|---|
| `data/prepared_context/` | SOURCE-OF-TRUTH / prepared-context-v0 | 19,829 B / 16 files | tree `b1a3dacae9befeb6ccda5d2f99b33ded53b51c44fd87ef3ef8a77828c39f60c3` | tracked | Manual eligibility/provenance and exact source checksums |
| `data/prepared_context/manifest.json` | SOURCE-OF-TRUTH | 3,288 B | `d9b910afd1941873826065bcf6e343be28cd850d339b356457daadbde60ad2eb` | tracked | Exact fingerprint bound into P3 metadata |
| `checkpoints/policy_p3_global/P3_GLOBAL.pt` | MODEL_CHECKPOINT / final P3 | 1,654,579 B | `ccf829fdb7ab521cc12c299583efa7222c965440b1257ddfb35e03ddd7bcadb9` | tracked | Final model cited by reports/traces |
| `checkpoints/policy_p3_global/P3_GLOBAL.metadata.json` | CHECKPOINT_METADATA | 2,325 B | `e0a429acad54598897d39c9ee6cc1e601a15a02afa5c1bdfd482c5b91dfe2444` | tracked | Validates model, labels, architecture, runtime, and context fingerprint |
| `_kaggle_restore/timelymt-p3-global-checkpoints/checkpoint-metadata.json` | PUBLICATION_PROVENANCE | 578 B | `65626a8fe0b0ca235326d15af490be8741680dd28e48f8a388db0be925bdd606` | tracked | Unique stage/date/repository/Kaggle provenance |

Known backup/provenance location: Kaggle Dataset `iteams24/timelymt-p3-global-checkpoints`. The publication sidecar records stage `TRAINED`, repository commit `e836f7ac5658f22de3d907289671fcd0caf2b42d`, and the same validated checkpoint SHA-256.

## P3 Predictions and Metrics

| Path/group | Condition | Size / files | SHA-256 evidence | Git state | Consumers |
|---|---|---:|---|---|---|
| `outputs/experiments/policy-p3-global/predictions/dev/p3_global_*` | REAL_CONTEXT | 8,216,499 B / 15 | Five per-threshold tree hashes in JSON manifest | ignored | Metrics, full DEV report, ablation, trace generation |
| `outputs/experiments/policy-p3-global/predictions/dev/p3_global_zeroctx_*` | ZERO_CONTEXT | 8,136,129 B / 15 | Five per-threshold tree hashes in JSON manifest | ignored | Zero metrics, controlled ablation, trace generation |
| `outputs/experiments/policy-p3-global/metrics/dev/all.json` | REAL_CONTEXT | 6,867 B | `699fd41a727b2013ce9218443152e5a71ea9419217e497e05cd964f5ad71aa30` | ignored | P3/full/final reports |
| `outputs/experiments/policy-p3-global/metrics/dev/all-zeroctx.json` | ZERO_CONTEXT | 6,905 B | `313802451c1e1425a1f8e0512c4f8379f0d9a767632ec51e3a668604bb6af769` | ignored | Controlled ablation/final reports |

These outputs are regenerable only by model execution and evaluation, which is forbidden in this cleanup. Their exact frozen bytes are therefore protected.

## Demo Traces

| Path | Size | SHA-256 | Git state | Consumers |
|---|---:|---|---|---|
| `outputs/demo_traces/sims-real-0.60.json` | 1,609,112 B | `ae39a327b0e50061bf2e6eb85126156b63dcc814c47cf54c393a69f9f86779fd` | tracked | Static demo, trace report, final reports |
| `outputs/demo_traces/sims-zero-0.60.json` | 1,670,520 B | `e10dd76c9ffbc6b3fef07d87b258c5711d3db4fc5ccc2d275558de6216d29a1e` | tracked | Static demo, trace report, final reports |
| `outputs/demo_traces/sims-0.60-bookmarks.json` | 1,653 B | `3cf6b2a9911da7a6e2b4739038eab31988485e275fbdd2ec2805be1d80f4bb94` | tracked | Static demo and presentation workflow |

All three required paths existed before cleanup. They are frozen DEV artifacts and were not changed.

## Reports, Figures, Overleaf, and Demo

| Path | Size / files | Tree SHA-256 | Git state | Role |
|---|---:|---|---|---|
| `reports/` | 1,800,189 B / 21 | `062dbfdc6e317ea62335a8f1e2458232808934b1e1e73bad08b09634d5abee1b` | tracked | Complete final reporting package |
| `reports/figures/` | 1,456,295 B / 9 | `b80749ca7819111ea14efe4edafe446603fde7befb8aa9be06a75352a91a0472` | tracked | Accepted PNG figures and archival SVG sources |
| `reports/overleaf/` | 62,793 B / 3 | `3400fbff483345ce71934150a259b90155dbcad04ea1604c17903b561f3c78d5` | tracked | Canonical Vietnamese report package |
| `demo/` | 44,236 B / 7 | `60070e4d9fecb90e076c6c8283c23c4d3938255fe0299f0ee452a2535f03473d` | tracked | Static model-free trace viewer |

Key report hashes:

- `reports/final-report.md`: `c21aff6b9c4ac59e8a2e5908c269d07a783d8b24c28dff426885e543858261d3`
- `reports/p3_prepared_context_ablation.json`: `01ef82afe93fdee6a57bbce0b41926b6cd75e5313de06fb5b1eaf4da0e8841d8`
- `reports/p3_prepared_context_ablation.md`: `9cc9f5d01ae6916c4f6eb8175ad9599677da6fd8a92531de92d82791b0787c35`
- Individual hashes for every report, figure, Overleaf file, and prepared-context pool are represented by the tree hashes and were captured during manifest generation; key individual hashes are also retained in the machine-readable manifest or Phase A verification record.

## Source Contracts

| Path | Size / files before cleanup | Tree SHA-256 before cleanup | Notes |
|---|---:|---|---|
| `src/` | 1,361,815 B / 184 | `86fb1344ee5ed71ac6f3707c66511e095c56f607cb9325af6c332470586ea2f0` | Includes generated bytecode/egg-info that Phase A is allowed to remove; tracked source must remain |
| `configs/` | 6,246 B / 10 | `04fac899123b8b4d08d4f46f94da88d2d30084433a880110344161989b7d9afc` | Frozen data/translator/V1/V2/P3 contracts |
| `schemas/` | 37,833 B / 10 | `934faec1d8b7addd03250dd91286cb836a0fc4bcb1e47ec7b8312bd10f88f0ec` | Data and artifact schemas |

The `src/` pre-cleanup tree hash is informational and is expected to change only because allowed generated `__pycache__`, `.pyc`, and egg-info files are removed. Git-tracked source content is not authorized to change.

## Protected TEST Paths

The following path-only records were confirmed without opening, enumerating, sizing, or hashing TEST content:

- `data/streaming/raw/ted/ted-joy-buolamwini-algorithmic-bias/`
- `data/streaming/raw/ted/ted-margaret-mitchell-ai-values/`
- `data/streaming/parsed/ted-joy-buolamwini-algorithmic-bias/`
- `data/streaming/parsed/ted-margaret-mitchell-ai-values/`
- `data/streaming/aligned/ted-joy-buolamwini-algorithmic-bias/`
- `data/streaming/aligned/ted-margaret-mitchell-ai-values/`
- `data/streaming/timed/ted-joy-buolamwini-algorithmic-bias/`
- `data/streaming/timed/ted-margaret-mitchell-ai-values/`
- `data/streaming/processed/ted-joy-buolamwini-algorithmic-bias/`
- `data/streaming/processed/ted-margaret-mitchell-ai-values/`
- `data/policy/pseudo_labels/test/` if present now or in any future/external workspace
- Any `outputs/**/test/` path if present now or in any future/external workspace

Protection for every item: **DO NOT TOUCH**. `content_accessed=false`; `sha256=null`.

## Validation Result

- All required protected roots existed before cleanup.
- P3_GLOBAL checkpoint SHA-256 matched exactly.
- V2 P0/P1/P2 checkpoint SHA-256 values matched their metadata exactly.
- Prepared-context manifest SHA-256 matched its documented P3 fingerprint exactly.
- Demo trace paths existed and were hashed.
- Reports, figures, Overleaf source, and static demo existed and were hashed.
- TEST paths were marked protected without content access.

Preservation validation passed. Low-risk cleanup may proceed only for the explicit Phase A allowlist.
