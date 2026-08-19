# TimelyMT Repository Audit

Audit date: 2026-08-19  
Scope: read-only repository, metadata, dependency, disk-usage, and artifact audit.  
Permitted mutation: this file only.  
TEST safety: held-out TEST translation/reference content was not opened, parsed, summarized, or evaluated. TEST membership below comes only from `data/splits/experimental.json`, path names, sizes, and code/document references.

## 1. Executive Summary

TimelyMT is a compact Python research codebase surrounded by a much larger local runtime and generated-artifact tree. The working tree is approximately **4.94 GiB in 175,676 files** including `.git/`; approximately **4.43 GiB (89%)** is the ignored local `venv/`. The next largest areas are `outputs/` (337.61 MiB, 141,596 files), `docs/` (94.60 MiB), `data/` (61.97 MiB), and `.git/` (19.75 MiB).

The current scientific chain is coherent and evidence-backed:

```text
curated TED metadata
  -> raw/parsed/aligned/timed artifacts
  -> 17 canonical streaming talks + frozen manifest/split
  -> V1 TRAIN/DEV pseudo-label supervision
  -> V1 policies and frozen DEV archive
  -> V2 MiniLM+MLP checkpoints and DEV artifacts
  -> prepared-context pools + V1 TRAIN supervision
  -> P3_GLOBAL checkpoint
  -> P3 DEV REAL/ZERO rollout and evaluation artifacts
  -> reports, figures, static demo traces, and final report
```

The key conclusions are:

- `src/timelymt/`, configs, schemas, dataset manifests/splits, canonical data, prepared context, V1 frozen archive, V2/P3 checkpoints, P3 DEV outputs, reports, figures, and demo traces are connected by direct code or report references. They are not cleanup candidates.
- The final P3 checkpoint is `checkpoints/policy_p3_global/P3_GLOBAL.pt`, 1.58 MiB, SHA-256 `ccf829fdb7ab521cc12c299583efa7222c965440b1257ddfb35e03ddd7bcadb9`. Its metadata binds it to frozen V1 TRAIN supervision and `data/prepared_context/manifest.json`.
- `data/policy/pseudo_labels/{train,dev}` exactly duplicates the corresponding files in `docs/archive/timelymt-checkpoint/` by SHA-256. Both locations currently have a purpose: active V2/P3 loaders use `data/`, while V1 preservation and the locked TEST gate reference `docs/archive/`.
- `outputs/experiments/policy-v2/embedding-cache/` (207.64 MiB, 66,603 files), `outputs/experiments/policy-p3-global/embedding-cache/` (59.22 MiB, 18,997 files), and `outputs/translator/` (31.11 MiB, 55,861 files) are regenerable caches, but should be cleaned only after frozen outputs/checkpoints have been independently preserved.
- Strong local cleanup candidates are `venv/`, `build/`, `.pytest_cache/`, `.playwright-cli/`, Python `__pycache__/`, `*.pyc`, `src/timelymt.egg-info/`, and generated caches. No cleanup was performed.
- `_kaggle_restore/` is only 578 bytes and contains unique P3 package provenance (`repo_commit`, stage, creation time) not present verbatim in the local checkpoint metadata. Keep or merge its metadata before considering removal.
- The repository is already dirty with pre-existing modified and untracked source/scientific files. This audit does not attribute, revert, or alter those changes.
- TEST is **PROTECTED - DO NOT TOUCH**. The split metadata names two held-out talks, but their translation/reference content was not inspected.

## 2. Current Repository Overview

### Research generations

| Generation | Role | Evidence | Status |
|---|---|---|---|
| Dataset v1 | Frozen 17-talk English-Vietnamese streaming dataset; 12 TRAIN, 3 DEV, 2 TEST | `data/manifests/timelymt-streaming-dataset-v1.json`; checksum `6730...cce`; `data/splits/experimental.json`; checksum recorded as `aabc...dc4` | SOURCE-OF-TRUTH / frozen |
| V1 | Sparse causal P0/P1/P2 and baselines; DEV-frozen upstream supervision | `configs/experiments/research-mvp.json`, `src/timelymt/research/cli.py`, `docs/archive/checkpoint-summary.json` | REQUIRED-FROZEN-ARTIFACT |
| V2 | Post-hoc exploratory frozen-MiniLM+MLP P0/P1/P2 extension | `configs/experiments/policy-v2.json`; `checkpoints/policy_v2/`; `outputs/experiments/policy-v2/` | Frozen exploratory DEV artifact |
| P3_GLOBAL | P2-like policy plus fixed prepared-global embedding | `configs/experiments/policy-p3-global.json`; `checkpoints/policy_p3_global/`; `src/timelymt/research/policy_p3_global*.py` | Final/frozen reporting artifact |
| Controlled ablation | Same P3 checkpoint, REAL vs ZERO prepared-context slice | P3 prediction/metrics trees; `reports/p3_prepared_context_ablation.*`; demo traces | Strongest prepared-context evidence; frozen DEV only |

### Git state at audit time

- Modified tracked files already present: five research/translator source files and three tests.
- Untracked scientific/reporting areas already present include `_kaggle_restore/`, P3 checkpoint, demo, demo traces, reports, five reporting/trace scripts, and related docs/tests.
- Ignored local/generated areas include `venv/`, `build/`, caches, raw/parsed/aligned/timed data intermediates, pseudo-label working copies, experiment outputs, and translator cache.
- No Git LFS entries were reported by `git lfs ls-files`. LFS or an external immutable artifact store is relevant for future growth, but the current largest individual research files are only single-digit MiB.

### Status terminology

- `tracked`: present in Git index.
- `untracked`: not in Git index and not ignored.
- `ignored`: matched by `.gitignore`.
- A directory can contain a mixture; status below describes the dominant/current material.
- Sizes are approximate and include ignored files unless noted.

## 3. Top-Level Directory Inventory

| Path | Type / size | Git status | Apparent purpose and usage evidence | Classification | Eventual action |
|---|---:|---|---|---|---|
| `_kaggle_restore/` | dir, 0.001 MiB, 1 file | untracked | P3 Kaggle package boundary metadata; hash matches local P3 checkpoint and adds stage/date/repo identity | FROZEN_RESEARCH_ARTIFACT | KEEP or ARCHIVE_CANDIDATE after metadata merge; MANUAL_REVIEW |
| `.git/` | VCS, 19.75 MiB | internal | Repository history; pack 17.11 MiB | DEVELOPMENT_TOOLING | KEEP |
| `.playwright-cli/` | dir, 0.04 MiB, 7 files | ignored | Browser snapshots/logs; machine/session-specific; no runtime reference | CACHE | DELETE_CANDIDATE |
| `.pytest_cache/` | dir, 0.02 MiB, 5 files | ignored | Pytest cache; automatically generated | CACHE | DELETE_CANDIDATE |
| `.serena/` | dir, 0.01 MiB, 9 files | mixed tracked/ignored local config | Agent project metadata and memories, not research runtime | DEVELOPMENT_TOOLING / LOCAL_ENVIRONMENT | KEEP_BUT_REORGANIZE or MANUAL_REVIEW |
| `build/` | dir, 0.27 MiB, 49 files | ignored | Stale package build copy of `src/timelymt`; generated by packaging | GENERATED_OUTPUT | DELETE_CANDIDATE / REGENERABLE |
| `checkpoints/` | dir, 5.63 MiB, 18 files | mixed tracked/untracked | V1 smoke, V2 full, final P3, and an export ZIP | MODEL_CHECKPOINT / FROZEN_RESEARCH_ARTIFACT | KEEP; archive only after identity checks |
| `configs/` | dir, 0.01 MiB, 10 files | tracked | Alignment, translator, V1, V2, P3 contracts; direct code references | CONFIG | KEEP |
| `data/` | dir, 61.97 MiB, 242 files | mixed tracked/ignored | Raw/intermediate/canonical dataset, supervision, prepared context, review, manifests/splits | RAW_DATA / INTERMEDIATE_DATA / DERIVED_DATA / FROZEN_RESEARCH_ARTIFACT | KEEP; normalize cautiously |
| `demo/` | dir, 0.04 MiB, 7 files | untracked plus placeholders | Static artifact-replay viewer; directly loads three frozen DEV traces | DEMO | KEEP |
| `docs/` | dir, 94.60 MiB, 179 files before this audit | tracked/untracked | Dataset/research documentation plus V1 archive and duplicate ZIP | REPORTING / FROZEN_RESEARCH_ARTIFACT | KEEP; archive redundancy requires review |
| `experiments/` | dir, 0 MiB, 11 placeholders | tracked | Empty planned result taxonomy; no executable content | DEVELOPMENT_TOOLING / LEGACY_CANDIDATE | KEEP_BUT_REORGANIZE or MANUAL_REVIEW |
| `notebooks/` | dir, 0.27 MiB, 7 files | tracked | Kaggle V1/V2/P3 launch/restore/publication notebooks; STOP-before-TEST controls | DEVELOPMENT_TOOLING / FROZEN_RESEARCH_ARTIFACT | KEEP; label canonical vs historical |
| `outputs/` | dir, 337.61 MiB, 141,596 files | mostly ignored, demo traces untracked, test-plan tracked | V1/V2/P3 predictions, metrics, caches, logs, demo traces | GENERATED_OUTPUT / FROZEN_RESEARCH_ARTIFACT | KEEP protected subsets; caches REGENERABLE |
| `reports/` | dir, 1.72 MiB, 21 files | untracked | Canonical final Markdown/Overleaf reports, analyses, accepted figures | REPORTING / FROZEN_RESEARCH_ARTIFACT | KEEP |
| `schemas/` | dir, 0.04 MiB, 10 files | tracked | JSON contracts for data and derived artifacts | CORE_SOURCE / CONFIG | KEEP |
| `scripts/` | dir, 0.22 MiB, 12 files | mixed tracked/untracked/ignored bytecode | Local V2 bootstrap, static report builders, trace validation/comparison | DEVELOPMENT_TOOLING / REPORTING | KEEP source; remove bytecode |
| `src/` | dir, 1.30 MiB, 184 files | tracked modified plus ignored caches/egg-info | Canonical package implementation | CORE_SOURCE | KEEP |
| `tests/` | dir, 0.92 MiB, 92 files | tracked modified/untracked plus ignored caches | Offline data, translator, research, checkpoint, trace tests | TEST_CODE | KEEP; remove caches only |
| `training/` | dir, 0 MiB, 2 placeholders | tracked | Empty planned separation; actual training lives in `src/timelymt/research/` | LEGACY_CANDIDATE / DEVELOPMENT_TOOLING | MANUAL_REVIEW |
| `venv/` | dir, 4,534.59 MiB, 33,027 files | ignored | Machine-local Python 3.13 environment; README specifies reproducible `.venv` and Python 3.10 | LOCAL_ENVIRONMENT | DELETE_CANDIDATE; recreate from package dependencies |
| `.editorconfig` | file, 198 B | tracked | Editor formatting defaults | DEVELOPMENT_TOOLING | KEEP |
| `.env.example` | file, 58 B | tracked | Environment template | CONFIG | KEEP |
| `.gitignore` | file, 1.94 KiB | tracked | Current generated/local exclusion policy | CONFIG / DEVELOPMENT_TOOLING | KEEP; later improve |
| `Makefile` | file, 2.57 KiB | tracked | Data pipeline/test/V2 TEST targets; many advertised targets are explicit stubs | DEVELOPMENT_TOOLING | KEEP_BUT_REORGANIZE |
| `opencode.json` | file, 3.32 KiB | ignored | Machine/tool agent configuration, not application config | LOCAL_ENVIRONMENT | KEEP local; remain ignored |
| `pyproject.toml` | file, 314 B | tracked | Python package metadata/dependencies; no console scripts declared | CONFIG | KEEP |
| `README.md` | file, 4.20 KiB | tracked | Project overview and V1-era quick-start; demo status is now stale | REPORTING | KEEP; update later, not during frozen audit |
| `requirements-policy-v2-local.txt` | file, 187 B | tracked | V2 local environment dependency pin set used by bootstrap | CONFIG / DEVELOPMENT_TOOLING | KEEP |

Hidden directories are therefore divided as follows: `.git/` is VCS state; `.serena/` and `opencode.json` are editor/agent metadata; `.playwright-cli/` and `.pytest_cache/` are temporary caches; `venv/` is local environment state. None is a scientific artifact.

## 4. Code and Entrypoint Map

### Practical entrypoints

| Entry point | Purpose | Inputs | Outputs | Config/modules | Stage / relevance |
|---|---|---|---|---|---|
| `python -m timelymt.data.acquisition.cli` / `make acquire-data` | Acquire curated public TED transcript artifacts | candidate manifest, talk selection | raw provider files, acquisition results | `data/manifests/ted-ai-candidates.json`; acquisition core/TED adapter | Data acquisition; active/regenerable |
| `python -m timelymt.data.parsing.cli` / `make parse-data` | Normalize raw source/target transcript segments | `data/streaming/raw/` | `data/streaming/parsed/` | parsing core, TED/WIT3 parsers | Preprocess; active |
| `python -m timelymt.data.alignment.cli` / `make align-data` | Dynamic-programming source/target segment alignment | parsed source/target | `data/streaming/aligned/` | `configs/data/alignment.json`, alignment core/DP/scoring | Preprocess; active/frozen config |
| `python -m timelymt.data.timing.cli` / `make time-data` | Produce causal source-only lexical timing | parsed source | `data/streaming/timed/` | timing core/recovery/simulation/tokenization | Preprocess; active |
| `python -m timelymt.data.canonical.cli` / `make build-data` | Assemble one canonical streaming talk | raw, parsed, aligned, timed | `data/streaming/processed/<talk>/streaming-talk.json` | canonical builder/core | Canonicalization; active |
| `python -m timelymt.data.manifest.cli {build,summary,split}` | Index canonical talks and persist split | canonical processed tree | dataset manifest/split | manifest builder/core | Dataset freeze; active, do not rerun against frozen v1 casually |
| `python -m timelymt.data.pipeline.cli {prepare,validate,summary,build-calibration-set,calibrate,import-alignment-review}` | End-to-end data preparation, QA, calibration | TED candidates/config/review | all intermediate/canonical data, `outputs/dataset/` | all data modules | Active orchestration; mutation-heavy if run, not run here |
| `python -m timelymt.translator.cli` | Frozen EnViT5 translation/smoke | English text or prefixes | stdout/cache | `configs/translator/envit5.json`; translator core/cache/prefix/envit5 | Translator utility; active; not run here |
| `python -m timelymt.research.cli <stage>` | Main V1/V2/P3 research CLI | frozen dataset, supervision, checkpoints | labels/checkpoints/predictions/metrics/reports/traces | research modules and all experiment configs | Canonical research entrypoint |
| `scripts/policy_v2_bootstrap.py` / `make policy-v2-local` | Probe/create local Conda V2 environment and resume pipeline | requirements, repository, V1 archive/current supervision | local env/provenance/log/V2 outputs | `policy_v2_local`, `policy_v2_runner` | Active but machine-specific |
| `python -m timelymt.research.policy_v2_local` | Local V2 stage coordinator | V1 supervision/archive and runtime | V2 checkpoints/outputs/log | V2 modules | Standalone CLI, active |
| `python -m timelymt.research.policy_v2_test` / `make policy-v2-test` | Locked held-out TEST protocol | locked plan/checkpoints plus protected TEST paths | TEST artifacts if deliberately run | V2 gate | **PROTECTED; never run during cleanup/audit** |
| `scripts/report_p3_global_full_dev.py` | Static P3-vs-P2 full DEV analysis | existing P2/P3 outputs and context metadata | `reports/p3_global_full_dev_analysis.{md,json}` | no model execution required by design | Reporting; active/frozen output |
| `scripts/report_p3_prepared_context_ablation.py` | Static REAL-vs-ZERO ablation report | existing P3 metrics/predictions | `reports/p3_prepared_context_ablation.{md,json}` | P3 outputs | Reporting; active/frozen output |
| `scripts/report_rollout_artifact.py` | Read-only single-talk P3/P2 sanity report | existing rollout and context artifacts | configurable Markdown/JSON, currently sanity report | P2/P3 outputs | Reporting; standalone CLI |
| `scripts/compare_demo_traces.py` | Compare synchronized REAL/ZERO traces | two trace JSONs | analysis JSON/MD and bookmarks | trace contract | Demo/report builder; active |
| `scripts/validate_demo_trace.py` | Validate one trace schema/invariants | trace JSON | status only | trace contract | Static validation tool; active |
| `python -m http.server 8000` | Serve static demo | `demo/` and three trace artifacts | HTTP view only | browser JS/CSS/HTML | Demo; active and model-free |
| `python -m unittest discover ...` / `make test` | Offline tests | source and fixtures | test results/caches | all package areas | Development verification; not run in this audit |

`timelymt.research.cli` exposes these stages: `pseudo`, `validate-pseudo`, `mu-supervision`, `validate-mu`, `train`, `train-mu`, `rollout`, `rollout-selected`, `evaluate`, `select`, `freeze`, `report`, `import-v1`, `train-v2`, `rollout-v2`, `evaluate-v2`, `compare-v2`, `select-v2`, `freeze-v2`, `validate-p3`, `inspect-p3`, `train-p3`, `inspect-p3-checkpoint`, `rollout-p3`, and `evaluate-p3`. It supports `--split train|dev|test`; TEST remains an intentionally gated path and was not invoked.

### Makefile reliability

Implemented targets are the six data stage CLIs, manifest/pipeline operations, tests, local V2, and the locked V2 TEST gate. `inspect-data`, `fetch-talks`, `prepare-data`, `validate-data`, baseline/training/evaluation targets, demo targets, and `clean` currently print `Not implemented yet.` They are documentation placeholders, not practical entrypoints. The presence of `clean` is not evidence that any cleanup policy exists.

### Notebook roles

| Notebook | Role | Assessment |
|---|---|---|
| `kaggle-research-mvp.ipynb` | Original V1 end-to-end Kaggle workflow | Historical/canonical V1 runner; README points here |
| `kaggle-research-mvp-resume-safe.ipynb` | Resume-safe V1 workflow | Later operational snapshot; keep until canonical choice is documented |
| `TimelyMT-Kaggle-Resume-Final.ipynb` | Final V1 resume packaging variant | Historical snapshot / possible canonical V1 completion runner |
| `kaggle-policy-v2.ipynb` | V1 restore, V2 train/evaluate/persist through DEV | Canonical V2 Kaggle runner; explicitly stops before TEST |
| `kaggle-p3-global.ipynb` | P3 checkpoint infrastructure/restore/publication | Canonical P3 persistence runner; no evaluation stage |
| `kaggle-p3-global-lab.ipynb` | Controlled TRAIN/DEV P3 lab with disabled-by-default stages | Most complete P3 workflow; explicit STOP BEFORE TEST |

No notebook was executed. Similar V1 notebook names represent historical/resume snapshots, not proven exact duplicates.

## 5. Source Dependency Overview

### Package relationships

```text
timelymt.data.acquisition
  -> raw provider artifacts
timelymt.data.parsing
  -> parsed transcripts
timelymt.data.alignment + parsing
  -> aligned transcript
timelymt.data.timing + parsing
  -> timed source stream
timelymt.data.canonical + timing
  -> canonical streaming talk
timelymt.data.manifest + canonical
  -> dataset manifest/split
timelymt.data.translation_artifacts
  -> canonical + split + translator
timelymt.translator
  -> frozen EnViT5 + translation cache
timelymt.research.streaming/evaluation/policy/pseudo_labels/meaningful_units
  -> V1 supervision, baselines, checkpoints, rollouts, metrics
timelymt.research.policy_v2 + policy_v2_runner/local/test
  -> frozen MiniLM embeddings + MLP V2 artifacts
timelymt.data.prepared_context
  -> validated immutable pre-talk pools
timelymt.research.policy_p3_global + runner + p3_checkpointing
  -> P3 checkpoint, REAL/ZERO rollouts, package persistence
scripts/report_* and compare_demo_traces
  -> frozen reports/demo artifacts
```

### Module-by-module reference assessment

| Module/group | Discovered callers/references | Assessment |
|---|---|---|
| `data/acquisition/{cli,core,ted}.py` | Makefile, pipeline CLI/core, tests, data-acquisition docs | Active |
| `data/parsing/{cli,core,ted,wit3}.py` | Makefile, pipeline, alignment/timing, tests/docs | Active; WIT3 parser remains tested though current corpus is TED |
| `data/alignment/{cli,core,dp,scoring}.py` | Makefile, pipeline, tests/docs/config | Active |
| `data/timing/{cli,core,recovery,simulation,tokenization}.py` | Makefile, canonical, pipeline, tests/docs | Active |
| `data/canonical/{cli,builder,core}.py` | Makefile, manifest, research loaders, tests/docs | Active/source-of-truth loader |
| `data/manifest/{cli,builder,core}.py` | Makefile, translation artifacts, tests/docs | Active |
| `data/pipeline/{cli,core,calibration,qa}.py` | Makefile, tests, docs | Active orchestration/QA |
| `data/translation_artifacts.py` | research CLI/streaming/tests/docs | Active shared runtime-artifact contract |
| `data/prepared_context.py` | P3 policy/runner, tests, config/docs | Active P3 contract |
| `translator/{core,cache,envit5,prefix,cli}.py` | README/docs, research CLI, tests | Active frozen translator boundary |
| `research/{streaming,evaluation,policy,pseudo_labels,meaningful_units}.py` | V1 CLI, V2/P3 runners, tests | Active shared V1 foundation |
| `research/cli.py` | docs/notebooks/tests | Canonical monolithic research CLI; high responsibility overlap but frozen-path risk argues against immediate refactor |
| `research/policy_v2.py` | V2 local/runner/test, P3 modules, notebooks/tests | Active shared V2/P3 primitives |
| `research/policy_v2_runner.py` | research CLI/local/tests | Active V2 execution logic |
| `research/policy_v2_local.py` | bootstrap/Makefile/tests | Active machine-local coordinator |
| `research/policy_v2_test.py` | Makefile/tests | Deliberate standalone protected TEST gate; keep |
| `research/policy_p3_global.py` | runner/checkpointing/CLI/tests | Active final P3 model/representation |
| `research/policy_p3_global_runner.py` | research CLI/tests/report outputs | Active P3 train/rollout/evaluate logic |
| `research/p3_checkpointing.py` | P3 notebooks/tests | Active Kaggle/local persistence and validation |
| package `__init__.py` files | imports/package discovery | Keep even where no explicit caller is reported |

### Responsibility overlap

- `src/timelymt/research/` contains model definitions, training, rollout, evaluation, checkpointing, and CLI orchestration. It overlaps conceptually with the empty `training/` and `experiments/` roots.
- `scripts/` contains machine bootstrap and report/demo artifact transformations. These are operational/report tools rather than reusable package modules.
- `training/` and `experiments/` are currently taxonomy placeholders only. Moving active code into them now would break imports/notebooks/frozen commands for little immediate benefit.
- V2 and P3 deliberately reuse V1 streaming/evaluation and V2 encoder/MLP primitives. Similarity is architectural reuse, not accidental duplication.
- Reporting scripts overlap in JSON loading, metric extraction, and Markdown emission, but each has a distinct frozen artifact contract. Consolidation should wait until report regeneration checks exist.

## 6. Data Inventory

### Dataset lineage table

| Path/group | Format / approx. size | Origin and class | Split | Producer | Consumers / generation | Regenerable / frozen | Eventual action |
|---|---:|---|---|---|---|---|---|
| `data/manifests/ted-ai-candidates.json` | JSON; small | Curated external-source acquisition list; SOURCE-OF-TRUTH metadata | metadata | Manual curation | acquisition/pipeline | Not fully regenerable without curation; frozen evidence | KEEP |
| `data/manifests/acquisition-results.jsonl` | JSONL; ignored | Acquisition run ledger | metadata | acquisition/pipeline | dataset QA/docs | Regenerable from network but records actual run | KEEP or archive with dataset snapshot |
| `data/streaming/raw/ted/<talk>/` | provider files; part of 23.31 MiB streaming tree | Downloaded original TED metadata/transcripts; RAW_EXTERNAL_DATA | TRAIN/DEV/TEST by split mapping | acquisition TED adapter | parsing/canonical provenance | Network-regenerable in principle, upstream can drift | KEEP archived; TEST whole paths protected |
| `data/streaming/parsed/<talk>/` | JSON; ignored | Normalized monolingual transcript segments; INTERMEDIATE_DATA | TRAIN/DEV/TEST | parsing CLI/pipeline | alignment/timing | Regenerable from raw+code; frozen snapshot value | KEEP until verified archive; TEST protected |
| `data/streaming/aligned/<talk>/` | JSON; ignored | DP alignment and provenance; INTERMEDIATE_DATA | TRAIN/DEV/TEST | alignment CLI/pipeline | canonical builder/QA | Regenerable from parsed+frozen config | KEEP until verified; TEST protected |
| `data/streaming/timed/<talk>/source.en.json` | JSON; ignored | Source-only simulated lexical timing; INTERMEDIATE_DATA | TRAIN/DEV/TEST | timing CLI/pipeline | canonical builder | Regenerable from parsed+timing code | KEEP until verified; TEST protected |
| `data/streaming/processed/<talk>/streaming-talk.json` | 17 JSON files; tracked; largest 1.32 MiB | Canonical streaming talks; DERIVED_DATA and frozen dataset | 12 TRAIN, 3 DEV, 2 TEST | canonical builder/pipeline | manifests, all research loaders/evaluation | Technically regenerable but frozen checksums make current bytes authoritative | **KEEP; TEST protected** |
| `data/manifests/streaming-dataset.json` | JSON | Canonical index; checksum documented | all | manifest builder | research/data QA/docs | Regenerable but frozen identity | KEEP |
| `data/manifests/timelymt-streaming-dataset-v1.json` | JSON metadata snapshot | Frozen v1 identity; 17 talks, split counts/checksum | all | dataset freeze process | V1/V2/P3 configs/gates/reports | REQUIRED-FROZEN-ARTIFACT | KEEP |
| `data/splits/experimental.json` | JSON | Frozen speaker-aware split identity | 12 TRAIN, 3 DEV, 2 TEST | manifest split CLI | all research stages | REQUIRED-FROZEN-ARTIFACT | KEEP; TEST membership protected |
| `data/splits/pilot.json` | JSON | Earlier three-talk pipeline pilot | pilot | manifest tooling | docs/historical tests | Historical | ARCHIVE_CANDIDATE, manual review |
| `data/review/*` | JSON/TSV; 0.095 MiB | Human alignment calibration and results | calibration metadata | pipeline review/calibration + manual decisions | frozen alignment/quality docs | Human review not safely regenerable | KEEP |
| `outputs/dataset/*` | JSON; 0.114 MiB | Pipeline records, QA report/samples | all metadata, may identify TEST but not opened here | pipeline/QA | dataset-quality docs | Regenerable but records frozen QA | KEEP or archive with dataset v1 |
| `data/policy/prefixes/*` | JSONL/manifests; ignored smoke files | Translation request prefixes | smoke/derived | translation artifact tooling | translator/hypothesis generation | Regenerable | DELETE_CANDIDATE only for clearly labeled smoke after confirmation |
| `data/policy/hypotheses/*` | JSONL/manifests; ignored smoke files | Frozen-translator hypotheses | smoke/derived | translator tooling | pseudo-label generation | Regenerable but model/network costly | REGENERABLE / cache-like |
| `data/policy/pseudo_labels/train/` | 13 files, about 27 MiB | V1 oracle-derived supervision | TRAIN | `research.cli pseudo` | V1/V2/P3 training | Exact duplicate of V1 archive; active required input | KEEP |
| `data/policy/pseudo_labels/dev/` | 4 files, about 6.7 MiB | V1 DEV supervision | DEV | `research.cli pseudo` | V1/V2 validation/evaluation | Exact duplicate of V1 archive; active/historical | KEEP |
| `data/policy/pseudo_labels/smoke/` | ignored | Smoke supervision | smoke | V1 research CLI/tests | smoke training | Regenerable | DELETE_CANDIDATE after confirming no debugging need |
| `data/policy/pseudo_labels/_local_backup/` | ignored partial TRAIN backup; several MiB | Local V2 restore safety snapshot | TRAIN partial | `policy_v2_local.py` | rollback only | Local-only, but may be evidence of interrupted restore | MANUAL_REVIEW before delete |
| `data/policy/mu_zhang2020/` | JSONL/manifest; active smoke locally, full copy in archive | Literature baseline supervision | TRAIN/DEV | `research.cli mu-supervision` | meaningful-unit policy | Full frozen source in archive; local smoke regenerable | KEEP archive; review local smoke |
| `data/prepared_context/{train,dev}/` + manifest | 15 JSON pools, 0.019 MiB | Manually governed pre-talk context; 8 eligible sources, six context-bearing talks | 12 TRAIN, 3 DEV; explicitly no TEST pools | Manual source selection/curation and loader validation | P3 training/rollout/report | Human provenance and exact checksums are not safely regenerable | **KEEP** |
| `docs/archive/timelymt-checkpoint/data/policy/` | 37 files, 61.75 MiB per metadata | Frozen V1 supervision package | TRAIN/DEV | V1 Kaggle export | V1 reproduction, V2 import, TEST gate archive dependency | REQUIRED-FROZEN-ARTIFACT | **KEEP** |
| Dataset-like notebook/Kaggle packages | notebooks + ZIP/export metadata | Persistence wrappers, not primary dataset | TRAIN/DEV | Kaggle workflows | restoration/publication | Historical/frozen operational evidence | KEEP/ARCHIVE after cataloging |

### Split safety

The frozen split metadata states 12 TRAIN, 3 DEV, and 2 TEST talks. The two TEST IDs were learned from the split manifest only. This audit did not open their raw, parsed, aligned, timed, canonical, pseudo-label, prediction, or reference content. All corresponding trees are classified **PROTECTED - DO NOT TOUCH**. No `data/policy/pseudo_labels/test/` contents were observed or read; code merely references that protected path.

### Prepared-context situation

The manifest contains 15 pools: 12 TRAIN and 3 DEV. Eight eligible source documents cover five TRAIN talks and one DEV talk; nine pools are valid empty pools. It records source IDs and SHA-256 checksums. P3 checkpoint metadata records the manifest fingerprint `d9b910afd1941873826065bcf6e343be28cd850d339b356457daadbde60ad2eb`. No prepared-context embedding files are stored under `data/`; embeddings are generated into experiment caches. Thus pools/manifests are source-of-truth, while embedding caches are regenerable.

## 7. Data Lineage

```text
data/manifests/ted-ai-candidates.json
    |
    | timelymt.data.acquisition.cli / pipeline prepare
    v
data/streaming/raw/ted/<talk>/
    |
    | timelymt.data.parsing.cli (TED/WIT3 parsers)
    v
data/streaming/parsed/<talk>/{source.en,target.vi}.json
    |                                      |
    | alignment CLI + configs/data/alignment.json
    |                                      | timing CLI (source only)
    v                                      v
data/streaming/aligned/<talk>/       data/streaming/timed/<talk>/source.en.json
    \                                      /
     \ timelymt.data.canonical.builder    /
      v                                  v
data/streaming/processed/<talk>/streaming-talk.json (17 frozen canonical talks)
    |
    +--> data/manifests/streaming-dataset.json
    |       +--> data/manifests/timelymt-streaming-dataset-v1.json
    |       +--> data/splits/experimental.json (12 TRAIN / 3 DEV / 2 protected TEST)
    |
    +--> frozen EnViT5 + translation requests/hypotheses
            |
            +--> data/policy/pseudo_labels/{train,dev}/
            |       +--> V1 P0/P1/P2 + baselines
            |       |       +--> checkpoints/policy + outputs/experiments/research-mvp
            |       |       +--> docs/archive/timelymt-checkpoint (DEV-frozen V1 package)
            |       |
            |       +--> V2 MiniLM+MLP training
            |       |       +--> checkpoints/policy_v2/V2P{0,1,2}.pt
            |       |       +--> outputs/experiments/policy-v2/{predictions,metrics}
            |       |
            |       +--> P3_GLOBAL training
            |               ^
            |               |
            |       data/prepared_context/{train,dev}/ + manifest
            |               |
            |               v
            |       checkpoints/policy_p3_global/P3_GLOBAL.pt
            |               |
            |               +--> P3 DEV REAL and ZERO rollouts
            |                       +--> outputs/experiments/policy-p3-global/{predictions,metrics}
            |                       +--> reports/p3_*.{md,json}
            |                       +--> outputs/demo_traces/*.json
            |                               +--> demo/ static viewer
            |                               +--> reports/demo_trace_*.{md,json}
            |                               +--> reports/figures + final-report + overleaf
            |
            +--> data/policy/mu_zhang2020/{train,dev}/ -> literature baseline
```

TEST branches intentionally terminate at protected split membership in this diagram. No TEST evaluation lineage was inspected or executed.

## 8. Checkpoint Inventory

| Path | Size | Identity/references | Frozen / superseded / reproducibility | Action |
|---|---:|---|---|---|
| `checkpoints/policy/smoke/{P0,P1,P2,mu_zhang2020}.{joblib,metadata.json}` | 0.076 MiB total | Smoke V1 models; paths are under generated V1 checkpoint root | Not final; useful test/debug artifacts only | REGENERABLE; MANUAL_REVIEW before delete |
| `docs/archive/timelymt-checkpoint/checkpoints/policy/` | 3.81 MiB per archive metadata | Full V1 P0/P1/P2/MU checkpoints; stage `dev-frozen-complete`, commit `6c75...` | Frozen historical foundation; V2/TEST gate dependency | **KEEP** |
| `checkpoints/policy_v2/V2P0.pt` + metadata | 463.8 KiB | SHA `f7f0...d299`; V2 P0; full, exploratory | Historical V2 comparison/reproducibility | KEEP |
| `checkpoints/policy_v2/V2P1.pt` + metadata | 847.8 KiB | SHA `e110...13fd`; V2 P1; full, exploratory | Historical V2 comparison/reproducibility | KEEP |
| `checkpoints/policy_v2/V2P2.pt` + metadata | 1,231.8 KiB | SHA `4d53...3fbe`; selected V2 family includes frozen `v2_P2_0.50` | Direct P3/P2 comparison input | **KEEP** |
| `checkpoints/policy_p3_global/P3_GLOBAL.pt` + metadata | 1,615.8 + 2.3 KiB | SHA `ccf829...adb9`; prepared manifest fingerprint; directly cited by both final reports | Known final/frozen P3 checkpoint; required for reproducibility and trace identity | **DO NOT TOUCH** |
| `checkpoints/archive.zip` | 1.47 MiB | Untracked export; exact contained identity was not assumed from name | Possible duplicate checkpoint package | MANUAL_REVIEW / ARCHIVE_CANDIDATE after listing/manifest comparison |
| `_kaggle_restore/.../checkpoint-metadata.json` | 578 B | Same P3 SHA, stage `TRAINED`, created `2026-08-18`, repo commit `e836...`, dirty flag | Unique provenance sidecar, not model bytes | KEEP until merged into canonical provenance |

The P3 model is not merely the newest checkpoint: it is referenced by `reports/final-report.md`, `reports/overleaf/main.tex`, P3 reports, demo trace identities, configuration, loaders, and packaging notebooks. V2 is not safely superseded because final analysis uses P2 as comparison. V1 is not safely superseded because V2/P3 supervision and experiment identity derive from it.

## 9. Outputs and Generated Artifacts

| Output group | Size/count | Producer / consumer | Reproducibility / scientific status | Action |
|---|---:|---|---|---|
| `outputs/experiments/research-mvp/` | 0.53 MiB current local; frozen archive copy is 24.86 MiB/104 files | V1 research CLI -> V1 selection/freeze and later V2 import | Current tree appears smoke/interrupted subset; archive is authoritative frozen V1 | Keep archive; local subset MANUAL_REVIEW |
| `outputs/experiments/policy-v2/embedding-cache/` | 207.64 MiB, 66,603 files | Frozen MiniLM encoder cache -> V2 train/rollout | Regenerable from inputs + pinned encoder; expensive, not report source-of-truth | REGENERABLE / cleanup candidate after preservation check |
| `outputs/experiments/policy-v2/predictions/` | 20.08 MiB, 45 files | V2 DEV rollout -> metrics/P3 comparison reports | Frozen exploratory DEV evidence | **KEEP** |
| `outputs/experiments/policy-v2/metrics/` | 0.056 MiB, 16 files | V2 evaluation -> selection/comparison/report | Frozen exploratory DEV evidence | **KEEP** |
| V2 provenance/config/selection/comparison files | ~0.11 MiB | V2 runner/local -> reports and reproducibility | Small, high-value source-of-truth outputs | **KEEP** |
| `outputs/experiments/policy-p3-global/embedding-cache/` | 59.22 MiB, 18,997 files | P3 encoder cache | Regenerable from frozen inputs/pinned model | REGENERABLE |
| `outputs/experiments/policy-p3-global/predictions/` | 15.60 MiB, 30 files | P3 DEV REAL/ZERO rollouts -> metrics/reports/trace generation | Frozen controlled-ablation evidence | **DO NOT TOUCH** |
| `outputs/experiments/policy-p3-global/metrics/` | 0.013 MiB, 2 files | P3 evaluation -> reports | Frozen report inputs | **DO NOT TOUCH** |
| `outputs/translator/` | 31.11 MiB, 55,861 files | EnViT5 translation cache | Regenerable from requests + pinned translator, but costly/network-dependent | REGENERABLE; clean only after artifact validation |
| `outputs/demo_traces/` | 3.13 MiB, 3 files | P3 traced DEV rollout + compare script | Directly loaded by demo and cited in final reports | **DO NOT TOUCH** |
| `outputs/dataset/` | 0.114 MiB, 5 files | pipeline QA | Frozen dataset QA evidence referenced by docs | KEEP |
| Empty `outputs/{figures,logs,metrics,predictions,runs}/.gitkeep` | negligible | Planned generic namespaces | Placeholders only | KEEP or remove in later structure decision |

The 141,596-file count is driven by JSON-oriented embedding and translator caches, not by predictions. Cleaning those caches would improve filesystem performance more than deleting any checkpoint. Canonical/frozen output groups must be excluded from any broad `outputs/` cleanup command.

## 10. Reports, Documentation, and Demo

### Canonical reporting set

- `reports/final-report.md`: canonical English report source and explicit artifact index.
- `reports/overleaf/main.tex`, `references.bib`, `README.md`: canonical Vietnamese Overleaf package. It uses the five PNG figures; SVGs are archival/editable sources.
- `reports/figures/figure-1..5`: accepted report figures. PNGs are compilation inputs; SVGs for figures 1-4 are useful source/archival forms. Figure 5 currently exists as PNG only.
- `reports/p3_global_full_dev_analysis.{md,json}`: full P3 DEV vs P2 analysis.
- `reports/p3_prepared_context_ablation.{md,json}`: controlled REAL/ZERO analysis.
- `reports/p3_global_sims_0.50_sanity.{md,json}`: single-talk sanity evidence.
- `reports/demo_trace_sims_0.60_analysis.{md,json}`: synchronized trace analysis.

### Documentation set

- Dataset pipeline/source-of-truth docs: `data-acquisition.md`, `data-parsing.md`, `data-alignment.md`, `data-timing.md`, `canonical-dataset.md`, `dataset-manifests.md`, `dataset-quality-report.md`, `alignment-review.md`.
- Runtime/research contracts: `translator.md`, `translation-artifacts.md`, `research-mvp-runbook.md`, `policy-v2-test-plan.md`, `p3-global.md`, `prepared-context-workplan.md`, `demo-trace-spec.md`, `demo-ui-spec.md`.
- Research synthesis: `research-final-summary.md` and short version. These directly describe the final frozen state.
- `docs/report.tex` and `docs/24022397_nlp_report.pdf`: earlier report artifacts; likely historical now that `reports/overleaf/` is canonical, but no deletion recommendation without researcher confirmation.
- `docs/results.md` and `docs/experiment-plan.md` are empty; tracked placeholders are low-value but not harmful.

### V1 archive

`docs/archive/timelymt-checkpoint/` is an expanded 86.23 MiB DEV-frozen V1 package containing checkpoint, supervision, predictions, metrics, selection, and frozen configuration. `docs/archive.zip` is 7.77 MiB and `docs/archive/checkpoint-summary.json` records the original archive identity and SHA. The expanded tree is actively referenced by V2 import and the locked V2 TEST gate; it is not just documentation. The ZIP appears to be a packaging duplicate but must be compared to the recorded archive checksum/contents before any decision.

### Demo

`demo/index.html`, `app.js`, and `styles.css` form a static viewer. `app.js` directly loads the three `outputs/demo_traces/` artifacts. `demo/README.md` and `PRESENTATION.md` provide operation and presentation guidance. Empty `demo/server/` and `demo/web/` are obsolete scaffold candidates because the implemented demo is root-level and backend-free, but manual confirmation is required before removing placeholders.

## 11. Local Tooling, Cache, and Environment Files

| Path | Generated/machine-specific? | Runtime/repro need | Tracked/ignored | Safe-candidate assessment |
|---|---|---|---|---|
| `venv/` | Yes; Python 3.13 local env | Not required; package requires >=3.10 and project says 3.10 verified | ignored | Strong DELETE_CANDIDATE; recreate instead of archive |
| `build/` | Yes; package build output | No; duplicates `src/` | ignored | Strong DELETE_CANDIDATE |
| `.pytest_cache/` | Yes | No | ignored | Strong DELETE_CANDIDATE |
| `.playwright-cli/` | Yes; session snapshots/logs | No runtime need | ignored | Strong DELETE_CANDIDATE unless screenshots/logs are needed for UI debugging |
| `**/__pycache__/`, `*.pyc` | Yes | No | ignored | Strong DELETE_CANDIDATE |
| `src/timelymt.egg-info/` | Editable-install metadata | No; regenerated by install | ignored | Strong DELETE_CANDIDATE |
| `.serena/project.local.yml` | Yes | No research need | ignored | Local-only; keep if tooling in use |
| `.serena/project.yml`, memories | Tool project knowledge | No research runtime need | tracked | Manual team decision; do not conflate with cache |
| `opencode.json` | Local agent config | No app/runtime need | ignored | Keep local or remove manually; contains tooling config, not research evidence |
| embedding caches | Generated, hardware-independent values tied to model revision | Speeds reruns, not required if source/model available | ignored | Regenerable; preserve frozen outputs first |
| translator cache | Generated, tied to translator fingerprint | Speeds expensive reruns | ignored | Regenerable but high recomputation/network risk |
| `_kaggle_restore/` | Kaggle restore residue, but contains unique provenance | Small and scientifically informative | untracked | Not a safe deletion candidate yet |

## 12. Git Hygiene

### What is already handled well

`.gitignore` covers Python caches/builds, virtual environments, IDE files, Jupyter checkpoints, logs/temp files, Playwright state, model/runtime caches, downloaded model weights, generated dataset intermediates, generated policy inputs/supervision, V1 generated checkpoints, most outputs, and exported `*.tar.gz` archives. `opencode.json` is intentionally ignored.

### Gaps and risks

| Finding | Evidence | Recommendation, not executed |
|---|---|---|
| P3 checkpoint is untracked with no explicit policy | `checkpoints/policy_p3_global/` appears untracked; final reports directly reference it | Explicitly whitelist and track the final pair, or document an immutable external artifact/LFS retrieval contract |
| V2 checkpoints are tracked while P3 policy differs | V2 `.pt` files tracked; P3 untracked | Normalize per-artifact policy rather than blanket `checkpoints/` ignore |
| Reports/demo/final traces are untracked | Git status shows whole canonical reporting stack untracked | Add intentionally; do not ignore canonical report/demo artifacts |
| `.serena/` policy is partial | local YAML ignored by nested ignore; memories/project config tracked | Decide whether agent memories are team tooling or local-only |
| `docs/archive.zip`, `checkpoints/archive.zip` are untracked | ZIPs are not covered by archive rules | Add a targeted archive policy only after deciding which ZIP is canonical |
| Generic `outputs/experiments/*` ignore hides P3 frozen evidence | check-ignore reports P3 root ignored | Use negation rules for specific frozen P3 metrics/predictions/provenance if they belong in Git; keep caches ignored |
| Pseudo-label working copies are ignored despite active dependencies | V2/P3 load `data/policy/pseudo_labels/train` | Preserve through archive/external artifact manifests; avoid assuming ignored means disposable |
| Raw/intermediate data are ignored | Intended, but frozen dataset reproduction depends on them if upstream changes | Keep an immutable external archive with checksums and provenance |
| No LFS | `git lfs ls-files` empty | LFS is optional at present; useful if canonical checkpoints/data grow. Do not put regenerable caches in LFS |

Important artifacts that must **not** be broadly ignored without a retrieval contract include frozen manifests/splits, prepared context, final checkpoints, final reports/figures, controlled-ablation metrics/predictions, and demo traces. Conversely, `venv`, build/caches, local logs, and embedding/translator caches should stay ignored.

## 13. Disk Usage

### A. Top-level sizes

| Path | Files | Approx. size |
|---|---:|---:|
| `venv/` | 33,027 | 4,534.59 MiB |
| `outputs/` | 141,596 | 337.61 MiB |
| `docs/` | 179 before audit | 94.60 MiB |
| `data/` | 242 | 61.97 MiB |
| `.git/` | 180 | 19.75 MiB |
| `checkpoints/` | 18 | 5.63 MiB |
| `reports/` | 21 | 1.72 MiB |
| `src/` | 184 | 1.30 MiB |
| `tests/` | 92 | 0.92 MiB |
| `build/`, `notebooks/`, `scripts/` | 49 / 7 / 12 | 0.27 / 0.27 / 0.22 MiB |
| all remaining top-level items | small | <0.1 MiB each |

### B. Largest individual files outside `venv/` and `.git/`

| File | Approx. size | Assessment |
|---|---:|---|
| `docs/archive.zip` | 7.77 MiB | V1 package archive candidate; verify before any action |
| `data/policy/pseudo_labels/train/ted-greg-brockman-chatgpt-potential.jsonl` | 4.78 MiB | Active frozen TRAIN supervision; exact archive duplicate |
| archive copy of same | 4.78 MiB | Frozen V1 package copy |
| archive V1 MU copy for same talk | 3.90 MiB | Frozen literature-baseline supervision |
| local partial backup Chris Urmson label file | 2.75 MiB | Interrupted/local rollback data; manual review |
| pseudo-label Chris Urmson active/archive copies | 2.72 MiB each | Active + frozen duplicate |
| DEV Jeff/Luis pseudo labels and archive copies | 2.70 / 2.55 MiB each | Frozen DEV supervision |
| `outputs/demo_traces/sims-zero-0.60.json` | 1.59 MiB | Frozen demo/report input |
| `checkpoints/policy_p3_global/P3_GLOBAL.pt` | 1.58 MiB | Final checkpoint |
| `outputs/demo_traces/sims-real-0.60.json` | 1.53 MiB | Frozen demo/report input |
| `checkpoints/archive.zip` | 1.47 MiB | Ambiguous export package |
| largest canonical talk | 1.32 MiB | Frozen dataset artifact |
| `checkpoints/policy_v2/V2P2.pt` | 1.20 MiB | Frozen V2 comparison checkpoint |

### C. Largest directory trees

- `venv/`: 4.43 GiB local environment.
- `outputs/experiments/policy-v2/`: approximately 227.9 MiB, mostly embedding cache.
- `docs/archive/`: 86.23 MiB expanded frozen V1 package.
- `outputs/experiments/policy-p3-global/`: approximately 74.8 MiB, mostly embedding cache plus frozen predictions.
- `outputs/translator/`: 31.11 MiB cache.
- `data/policy/`: 38.49 MiB, mainly V1 supervision and local backup.
- `data/streaming/`: 23.31 MiB, raw/intermediate/canonical data.

### D. Obvious disk consumers safe to classify as local/generated

`venv/`, embedding caches, translator cache, package build, bytecode, pytest cache, and Playwright state. Together they account for roughly 4.73 GiB. The first cleanup should target these, not scientific outputs.

### E. Large intentionally required research artifacts

The expanded V1 archive, active V1 supervision, V2/P3 predictions and metrics, P3 checkpoint, demo traces, canonical dataset, raw acquisition snapshot, and report figures are all evidence-bearing. Their combined size is modest relative to the local environment and caches.

## 14. Duplicate and Redundant Files

### Exact duplicates

SHA-256 analysis found exact byte duplicates for every active V1 pseudo-label TRAIN/DEV artifact and manifest compared with `docs/archive/timelymt-checkpoint/data/policy/pseudo_labels/{train,dev}/`. Examples include all 12 TRAIN talk JSONLs, all three DEV talk JSONLs, and both manifests. This is intentional operational duplication: active paths satisfy current loaders; archive paths preserve the V1 package and satisfy explicit V2/TEST archive references.

Empty `.gitkeep` files naturally share the empty-file hash and are not meaningful data duplicates.

### Probable duplicates requiring confirmation

| Items | Evidence | Classification |
|---|---|---|
| `docs/archive.zip` vs expanded `docs/archive/` | Names, size, and checkpoint summary describe an exported archive; expanded tree mirrors package layout | Probable packaged duplicate; verify archive SHA/content before action |
| `checkpoints/archive.zip` vs checkpoint trees | Name and 1.47 MiB size suggest export, but no assumption about contents was made | UNKNOWN / NEEDS MANUAL CONFIRMATION |
| Active pseudo labels vs V1 archive | Exact hashes | Exact duplicate with two active path roles |
| V1 notebook trio | Similar naming and workflow stages | Historical snapshots, not proven duplicates |
| P3 notebook pair | One persistence-focused and one full lab | Distinct artifacts with similar subject |
| `build/lib/timelymt` vs `src/timelymt` | Build output path and package files | Generated snapshot; may be stale, safe to regenerate |

### Distinct similar artifacts

- V1, V2, and P3 checkpoints are different model families/input dimensions.
- P3 REAL and ZERO predictions/traces intentionally differ and are the controlled ablation.
- Markdown and JSON report pairs serve human-readable and machine-readable roles.
- PNG and SVG figures serve report compilation and editable archival roles.

## 15. Reproducibility Classification

| Conceptual class | Artifacts | Regeneration requirements |
|---|---|---|
| SOURCE-OF-TRUTH | `src/`, configs, schemas, curated candidate metadata, manifests/splits, prepared-context pools/provenance | Version-controlled source and manual curation history |
| REQUIRED-FROZEN-ARTIFACT | Dataset v1 canonical bytes/checksums, V1 archive, V2 checkpoints/DEV outputs, P3 checkpoint/DEV outputs, controlled ablation, traces, reports/figures | Preserve exact bytes and hashes; regeneration may not reproduce environment/model/network state exactly |
| REGENERABLE-FROM-SOURCE | parsed/aligned/timed intermediates, request/hypothesis artifacts, embedding caches, translator cache, report JSON/MD when producer inputs are frozen, build/bytecode | Raw source + pinned code/config/models; report scripts require frozen output inputs |
| RAW-EXTERNAL-DATA | downloaded TED raw tree | Curated manifest, network access, unchanged upstream pages, adapter version |
| LOCAL-ONLY | `venv/`, `opencode.json`, `.serena/project.local.yml`, logs, local environment/provenance files | Recreate environment/tool setup |
| CACHE | pytest/Playwright/Python caches; translator and embedding caches | Rerun the corresponding deterministic operation with pinned dependencies/models |
| HISTORICAL | pilot split, old report source/PDF, V1 notebook snapshots, V1/V2 historical stages, smoke checkpoints | Preserve until a canonical archive index and researcher approval exist |
| UNKNOWN | `checkpoints/archive.zip`, empty scaffold roots, partial local pseudo-label backup ownership | Manual confirmation and safe archive inspection needed |

Reproducibility is currently split across Git-tracked source, ignored local scientific artifacts, untracked final artifacts, and a tracked expanded V1 archive. Before cleanup, create a checksum inventory and immutable external backup for all REQUIRED-FROZEN-ARTIFACT items; this audit does not create that inventory because only this report was permitted.

## 16. Files With No Discovered References

No Python implementation module was proven dead. Most are imported by another module, a CLI, tests, docs, or notebooks. A lack of an import is not sufficient for CLI tools.

### Likely dead or obsolete scaffold, not deletion-approved

- `training/policy/.gitkeep`, `training/pseudo_labeling/.gitkeep`: no implementation; training is in `src/timelymt/research/`.
- most `experiments/**/.gitkeep`: planned taxonomy only; outputs actually live under `outputs/experiments/`.
- `demo/server/.gitkeep`, `demo/web/.gitkeep`: static demo is implemented directly under `demo/` with no server application.
- empty `docs/results.md`, `docs/experiment-plan.md`: placeholders with no content.
- empty package placeholders such as `src/timelymt/{baselines,evaluation,memory,retrieval,simulator,utils}/.gitkeep` and policy subdirectories: planned architecture, no current code.

### Standalone CLI tools

- All six `scripts/*.py` files are intentionally standalone and have direct docs/test/report artifact evidence. They should not be classified as dead because package code does not import them.
- Data and translator `cli.py` modules are invoked by Makefile/docs.
- `policy_v2_local.py` and `policy_v2_test.py` are module entrypoints.

### Research historical artifacts

- V1 notebook variants, pilot split, smoke checkpoints, `docs/report.tex`, and `docs/24022397_nlp_report.pdf` may lack current runtime callers but preserve research history.

### Unknown / manual confirmation

- `checkpoints/archive.zip`.
- Whether `docs/archive.zip` is the exact original archive identified by `docs/archive/checkpoint-summary.json`.
- Whether `_kaggle_restore/` metadata has been copied to any external canonical P3 package index.
- Whether local partial pseudo-label backup is still needed for rollback.

## 17. Proposed Canonical Repository Structure

### Safe immediate principle

Do not move frozen artifacts merely to make the tree prettier. Current loaders and reports hard-code existing paths. The safe immediate structure is the current structure with explicit protected/cache boundaries and a small artifact index, not a mass relocation.

### Ideal future layout

```text
TimelyMT/
|-- configs/                         # unchanged identities
|-- data/
|   |-- manifests/                   # frozen dataset and split identities
|   |-- raw/                         # future normalized external source archive
|   |-- interim/{parsed,aligned,timed}/
|   |-- processed/                   # canonical streaming talks
|   |-- supervision/{v1,mu}/         # active derived supervision with manifests
|   `-- prepared_context/            # keep current contract/name
|-- src/timelymt/
|   |-- data/
|   |-- translator/
|   `-- research/{v1,v2,p3}/         # only after compatibility plan
|-- scripts/
|   |-- environment/
|   |-- reporting/
|   `-- demo/
|-- checkpoints/
|   |-- v1/
|   |-- v2/
|   `-- p3_global/
|-- outputs/
|   |-- frozen/{v1,v2,p3,ablation,demo_traces}/
|   `-- cache/{translator,embeddings}/
|-- archives/                        # immutable packages + manifests, not mixed into docs
|-- demo/
|-- reports/{analysis,figures,overleaf}/
|-- docs/
|-- notebooks/{canonical,historical}/
|-- schemas/
`-- tests/
```

This is an ideal future layout only. Migration requires compatibility shims or a coordinated update of code constants, configs, notebook paths, report citations, archive manifests, tests, and checksums. The highest-value structural change is separating frozen outputs from caches, not splitting every source module.

## 18. Cleanup Plan

Every action below is proposed only. Nothing was executed.

### Phase A - Safe Local Cleanup

| Current path | Proposed action | Reason / evidence | Risk | Breakage risk | Verify before action | Rollback |
|---|---|---|---|---|---|---|
| `venv/` | Remove local environment | 4.43 GiB, ignored, machine-local; dependencies declared | LOW | Shells using this interpreter stop until recreated | Record interpreter/package freeze only if needed; ensure no unique files were manually placed there | Recreate virtualenv and install dependencies |
| `build/` | Remove | Ignored generated copy of package | LOW | None for editable/source execution | Confirm no command imports `build/lib` via custom `PYTHONPATH` | Rebuild package |
| `.pytest_cache/`, `.playwright-cli/` | Remove | Ignored test/browser session caches | LOW | Loss of last-run/UI debug history | Confirm Playwright logs are not needed for an unresolved demo issue | Regenerate by rerunning tools later |
| all `__pycache__/`, `*.pyc`, `src/timelymt.egg-info/` | Remove | Generated Python/install metadata | LOW | First import/install is slower | Confirm no unusual bytecode-only module exists; source inventory shows corresponding source | Import/install regenerates |
| embedding and translator caches | Remove in a later local-cache pass | 298 MiB and >141k files; ignored/regenerable | MEDIUM | Reruns need model access and substantial compute; no change to existing predictions | Back up frozen checkpoints/predictions; validate cache paths do not contain provenance-only files | Restore backup or regenerate with pinned models |

### Phase B - Git Hygiene

| Current path | Proposed action | Reason / evidence | Risk | Breakage risk | Verify before action | Rollback |
|---|---|---|---|---|---|---|
| `.gitignore` output/checkpoint rules | Add explicit cache ignores and explicit frozen-artifact exceptions | P3/report/demo artifacts are currently untracked/ignored despite report references | MEDIUM | Incorrect glob could hide or expose scientific artifacts | Dry-run `git check-ignore -v` on every protected path | Revert the isolated `.gitignore` commit |
| canonical untracked reports/demo/P3 artifacts | Intentionally track or register in immutable artifact store | Current handoff is not reproducible from Git alone | MEDIUM | Repository growth or accidental omission | Hash, review licensing/data policy, and compare report artifact index | Restore from artifact backup/history |
| large future binary artifacts | Evaluate Git LFS | No current LFS; checkpoints may grow | LOW/MEDIUM | LFS availability/clone changes | Pilot on a non-frozen branch and document retrieval | Migrate back using standard Git procedures |

### Phase C - Data Normalization

| Current path | Proposed action | Reason / evidence | Risk | Breakage risk | Verify before action | Rollback |
|---|---|---|---|---|---|---|
| `data/streaming/{raw,parsed,aligned,timed,processed}` | Keep paths now; document lifecycle and archive raw/interim by dataset version | Existing pipeline is already coherent; names differ from ideal only cosmetically | LOW now / HIGH if moved | Hard-coded defaults, provenance paths, canonical checksums | Build a path-reference matrix and checksum manifest without touching TEST content | Restore versioned archive and original paths |
| active and archived V1 pseudo labels | Retain both until loaders can use a manifest-addressed artifact store | Exact duplicates but two real path contracts | MEDIUM | V2/P3 training or TEST gate could fail | Search all references; validate archive and active hashes | Restore copied tree from frozen package |
| local smoke/partial backups | Decide after researcher review | Clearly noncanonical, but interrupted backup may be only rollback evidence | MEDIUM | Loss of unfinished local work | Researcher confirms run is complete and active TRAIN hashes match archive | Restore from local backup/archive |
| TEST-related data trees | No cleanup or movement | Protected held-out evidence | HIGH | Contamination, loss, or protocol violation | Independent researcher-approved TEST handling plan | Restore immutable backup; stop work |

### Phase D - Code Organization

| Current path | Proposed action | Reason / evidence | Risk | Breakage risk | Verify before action | Rollback |
|---|---|---|---|---|---|---|
| empty `training/`, `experiments/` | Remove placeholders or repurpose only after deciding canonical taxonomy | They overlap conceptually with active `research/` and `outputs/experiments/` | LOW/MEDIUM | Docs/scripts may expect planned paths later | Search all path references and ask researcher | Restore placeholders from Git |
| `scripts/report_*.py` | Later group under `scripts/reporting/` only with compatibility wrappers | Clear purpose group | MEDIUM | Frozen commands/docs/tests break | Update and test every documented command; compare generated files byte-for-byte | Keep wrappers or revert move |
| monolithic `research/cli.py` | Do not refactor during cleanup; eventually split stage registration from implementations | 23 stage choices and V1/V2/P3 orchestration in one file | HIGH | Research behavior/path contracts can change | Full offline test suite plus frozen artifact nonmutation checks | Revert isolated refactor |
| V1/V2/P3 modules | Add documentation namespaces before physical reorganization | Versions are distinguishable but share intentional primitives | MEDIUM/HIGH | Import/notebook/checkpoint compatibility | Dependency graph and compatibility tests | Preserve existing imports |

### Phase E - Archive Legacy Artifacts

| Current path | Proposed action | Reason / evidence | Risk | Breakage risk | Verify before action | Rollback |
|---|---|---|---|---|---|---|
| V1 notebook variants | Mark one canonical, move others only in a versioned archive later | Historical/resume snapshots clutter discovery | MEDIUM | Loss of exact Kaggle operational history | Researcher identifies final successful runner; compare cell roles | Restore from Git/history |
| `docs/report.tex`, old PDF | Move to historical reports archive if superseded | Final canonical report now under `reports/` | MEDIUM | May be required submission evidence | Researcher confirms submission lineage | Restore archive |
| pilot split and smoke artifacts | Archive by milestone | Useful development history, not final science | LOW/MEDIUM | Tests/debug docs may refer to them | Search references and run offline tests in later authorized task | Restore milestone archive |
| ZIP packages | Keep one canonical immutable package plus expanded active form only if needed | Probable package redundancy | MEDIUM | Loss of original export metadata/compression identity | List safely, compare file hashes and recorded archive SHA | Restore from immutable copy |

### Phase F - Optional Deep Cleanup

| Current path | Proposed action | Reason / evidence | Risk | Breakage risk | Verify before action | Rollback |
|---|---|---|---|---|---|---|
| expanded V1 archive vs active copies | Deduplicate using artifact store/content-addressed links only after path abstraction | Largest true duplicate group | HIGH | Explicit code paths and TEST gate break | End-to-end static loader/checkpoint validation without TEST execution | Restore full expanded package |
| raw/intermediate data | Move out of primary Git workspace to immutable dataset storage | External/regenerable in principle | HIGH | Upstream drift prevents exact dataset reconstruction | Archive all bytes/checksums and perform authorized reconstruction on TRAIN/DEV only | Restore immutable snapshot |
| frozen predictions | Archive older threshold grids only after final report dependency matrix | Some are not directly cited individually | HIGH | Analyses/figures may silently depend on them | Trace every report JSON field to source paths | Restore artifact bundle |

## 19. DO NOT TOUCH

The following is the proposed explicit protected set before cleanup:

- **All TEST-related paths and content**, including the two held-out talk trees identified by the frozen split, any future/current `data/policy/pseudo_labels/test/`, and any TEST predictions/metrics. Do not open, move, hash semantically, evaluate, or regenerate without a separate approved protocol.
- `data/manifests/timelymt-streaming-dataset-v1.json`, `data/manifests/streaming-dataset.json`, `data/splits/experimental.json`.
- `data/streaming/processed/` as the frozen canonical dataset, especially protected TEST members.
- `data/review/` and `configs/data/alignment.json` as human/frozen dataset-quality evidence.
- `configs/translator/envit5.json` and all experiment configs.
- `data/policy/pseudo_labels/{train,dev}/` and `docs/archive/timelymt-checkpoint/`.
- `data/prepared_context/` in full, including empty pools and manifest.
- `checkpoints/policy_v2/` because P2 is a final comparison baseline.
- `checkpoints/policy_p3_global/P3_GLOBAL.pt` and metadata.
- `_kaggle_restore/.../checkpoint-metadata.json` until its unique provenance is merged elsewhere.
- `outputs/experiments/policy-v2/{predictions,metrics,dev-selection.json,v2-frozen-config.json,comparison-v1-v2.json,run-provenance.json,v1-source/}`.
- `outputs/experiments/policy-p3-global/{predictions,metrics}`.
- `outputs/demo_traces/`.
- all `reports/`, including figures and Overleaf source.
- `demo/` and its required trace paths.
- `docs/research-final-summary*.md`, demo specs, P3/V2/V1 runbooks/contracts, dataset-quality documentation.
- `src/`, `tests/`, `schemas/`, `configs/`, and canonical notebooks while scientific state is frozen.
- `docs/archive.zip` and `checkpoints/archive.zip` until their package identities are manually resolved.

## 20. LIKELY SAFE TO CLEAN

Only strong evidence-backed local/generated candidates are listed:

- `venv/`: ignored 4.43 GiB machine-local environment.
- `build/`: ignored packaging output duplicating source.
- `.pytest_cache/`: ignored test cache.
- `.playwright-cli/`: ignored session/browser captures, subject only to unresolved UI-debug need.
- all `__pycache__/` and `*.pyc` under source, scripts, and tests.
- `src/timelymt.egg-info/`: ignored install metadata.
- embedding caches under V2 and P3 experiment roots, **only after** frozen checkpoints/predictions/metrics are backed up and exact cache-only path selection is reviewed.
- translator cache under `outputs/translator/`, **only after** confirming all final prediction and trace artifacts are independent and preserved.

DO NOT DELETE THEM YET. Even low-risk deletion should be a separate authorized change with a before/after disk and Git check.

## 21. NEEDS MANUAL DECISION

| Path/item | Exact researcher question |
|---|---|
| `_kaggle_restore/` | “Has the P3 Kaggle `completed_stage`, creation time, repository commit, and dirty-state provenance been preserved in another canonical package manifest?” |
| `docs/archive.zip` | “Is this the original V1 export whose SHA-256 is recorded as `f1f58c...82ca`, and must original compressed bytes be retained for submission/recovery?” |
| `checkpoints/archive.zip` | “What generated this ZIP, which checkpoints does it contain, and is it the only copy of a published package?” |
| `data/policy/pseudo_labels/_local_backup/` | “Is the interrupted TRAIN partial backup still needed to recover unfinished local work?” |
| smoke policy artifacts | “Are smoke checkpoints/labels retained for active debugging or can tests regenerate them?” |
| V1 notebook variants | “Which notebook produced the frozen `dev-frozen-complete` archive, and which variants are historical snapshots?” |
| P3 notebook pair | “Should the full lab be canonical and the persistence-only notebook historical, or are both operator workflows supported?” |
| `docs/report.tex`, `docs/24022397_nlp_report.pdf` | “Are these prior submitted/reporting artifacts that must remain in the primary repository?” |
| `training/`, `experiments/`, empty package scaffolds | “Are these promised future architecture paths, or can empty placeholders be removed?” |
| `.serena/` tracked memories | “Are agent memories shared project documentation or developer-local state?” |
| active/archive pseudo-label duplication | “Is preserving both path contracts required, or may loaders be changed later to consume a single immutable V1 artifact store?” |
| ignored raw/intermediate data | “Is there a separately backed-up immutable copy of every downloaded and intermediate Dataset v1 artifact, including protected TEST data?” |
| final untracked artifacts | “Should final P3/report/demo artifacts be committed, placed in LFS, or registered in an external DOI/object store?” |

## 22. Canonical Workflow Map

The recommended canonical TRAIN/DEV-only workflow, using actual repository commands and paths, is:

```text
DATA ACQUISITION
  data/manifests/ted-ai-candidates.json
  python -m timelymt.data.acquisition.cli
        |
        v
PREPROCESS / QA
  python -m timelymt.data.pipeline.cli prepare
  configs/data/alignment.json
  raw -> parsed -> aligned + timed -> processed
        |
        v
FREEZE DATASET IDENTITY
  python -m timelymt.data.manifest.cli build
  data/manifests/streaming-dataset.json
  data/manifests/timelymt-streaming-dataset-v1.json
  data/splits/experimental.json
        |
        v
V1 SUPERVISION AND BASELINES (TRAIN/DEV ONLY)
  python -m timelymt.research.cli pseudo --split train|dev
  python -m timelymt.research.cli mu-supervision --split train|dev
  python -m timelymt.research.cli train / train-mu
  python -m timelymt.research.cli rollout/evaluate/select/freeze --split dev
  docs/archive/timelymt-checkpoint (canonical V1 frozen package)
        |
        v
V2 EXPLORATORY DEV
  scripts/policy_v2_bootstrap.py
  or notebooks/kaggle-policy-v2.ipynb
  checkpoints/policy_v2 + outputs/experiments/policy-v2
        |
        v
PREPARED CONTEXT
  data/prepared_context/{train,dev} + manifest
  validate through timelymt.data.prepared_context / `validate-p3`
        |
        v
P3 TRAIN / DEV ROLLOUT
  notebooks/kaggle-p3-global-lab.ipynb
  python -m timelymt.research.cli train-p3
  python -m timelymt.research.cli rollout-p3 --split dev
  python -m timelymt.research.cli evaluate-p3 --split dev
  checkpoints/policy_p3_global/P3_GLOBAL.pt
  outputs/experiments/policy-p3-global/{predictions,metrics}
        |
        v
CONTROLLED ABLATION / STATIC REPORTING
  REAL_CONTEXT vs ZERO_CONTEXT, same P3 checkpoint
  scripts/report_p3_global_full_dev.py
  scripts/report_p3_prepared_context_ablation.py
  scripts/report_rollout_artifact.py
        |
        v
TRACE / DEMO
  rollout-p3 --split dev --trace-output outputs/demo_traces/<run>.json
  scripts/validate_demo_trace.py
  scripts/compare_demo_traces.py
  python -m http.server 8000 -> /demo/
        |
        v
FINAL REPORT
  reports/final-report.md
  reports/overleaf/main.tex + references.bib + figures/*.png

STOP BEFORE TEST
```

No stage above should be rerun while the scientific state is frozen merely to validate cleanup. Static checks, hashes, and immutable backups are sufficient for cleanup preparation.

## 23. Recommended Next Step

Create a separate, researcher-approved **artifact preservation manifest** before cleanup. It should record path, size, SHA-256, Git status, artifact class, producer command, upstream fingerprints, and backup location for every item in Section 19. For TEST, record only approved path-level metadata under the TEST protocol and do not inspect content. Then perform Phase A as a dedicated local-only cleanup, with before/after `git status`, disk usage, and protected-path existence/hash checks. Do not begin data normalization or archive deduplication until the manual questions in Section 21 are answered.

This audit performed no training, inference, rollout, evaluation, cleanup, file movement, configuration change, checkpoint change, report change, figure change, Git-ignore change, or TEST-content access.
