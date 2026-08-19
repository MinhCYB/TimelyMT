# TimelyMT Cleanup Phase A

Cleanup date: 2026-08-19  
Authoritative audit: `docs/repository-audit.md`  
Preservation manifests: `docs/artifact-preservation-manifest.md` and `docs/artifact-preservation-manifest.json`

## 1. Safety Checks

Phase A followed the audit's LOW-risk allowlist only. Before deletion:

- All required protected roots were checked for existence.
- `checkpoints/policy_p3_global/P3_GLOBAL.pt` was recalculated as SHA-256 `ccf829fdb7ab521cc12c299583efa7222c965440b1257ddfb35e03ddd7bcadb9`, exactly matching the required frozen identity.
- V2 P0/P1/P2 checkpoint hashes were recalculated and matched their metadata exactly.
- `data/prepared_context/manifest.json` was recalculated as SHA-256 `d9b910afd1941873826065bcf6e343be28cd850d339b356457daadbde60ad2eb`, exactly matching the documented prepared-context fingerprint.
- V1 TRAIN/DEV supervision, the expanded V1 archive, V2/P3 predictions and metrics, demo traces, reports, figures, Overleaf, demo, configs, and schemas were hashed before deletion.
- Preservation JSON parsed successfully with Node's `JSON.parse` because `jq` is not installed.
- `git diff --check` passed for both preservation manifests before deletion.
- No model, application, training, inference, rollout, or evaluation entrypoint was run.

TEST safety was path-only. The two TEST talk IDs and their expected raw/parsed/aligned/timed/processed paths were taken from the authoritative audit and checked only with `Test-Path`. TEST directories were not enumerated, sized, read, parsed, summarized, evaluated, or hashed. The absent `data/policy/pseudo_labels/test/` path and any possible future `outputs/**/test/` paths remain protected.

## 2. Preservation Manifest Summary

The two manifests were created before cleanup:

- `docs/artifact-preservation-manifest.md`: human-readable preservation index.
- `docs/artifact-preservation-manifest.json`: machine-readable artifact metadata and validation state.

They cover:

- Canonical dataset manifests and the frozen split.
- Canonical TRAIN/DEV talks with TEST excluded from hash scope.
- Dataset review and QA evidence.
- Active V1 TRAIN/DEV supervision.
- The complete V1 `dev-frozen-complete` expanded archive and archive metadata.
- V2 P0/P1/P2 checkpoints and metadata.
- V2 frozen configuration, predictions, metrics, selection, provenance, and V1/V2 comparison.
- Prepared-context pools, manifest, and fingerprint.
- P3_GLOBAL checkpoint, metadata, and Kaggle publication provenance.
- P3 REAL_CONTEXT and ZERO_CONTEXT prediction threshold grids and metrics.
- Demo traces and bookmarks.
- Controlled-ablation reports, final report, figures, Overleaf source, and static demo.
- Source/config/schema contracts.
- Explicit path-only `DO NOT TOUCH` records for all known TEST areas.

Directory hashes use a documented deterministic tree-hash method. Important files use ordinary raw-byte SHA-256. Ignored scientific artifacts such as active V1 supervision and V2/P3 outputs are represented even though Git does not track them.

## 3. Frozen Artifact Verification

### Checkpoints and direct files

| Artifact | Verification result |
|---|---|
| P3_GLOBAL checkpoint | Unchanged; required SHA matched exactly |
| P3_GLOBAL metadata | Unchanged; SHA `e0a429acad54598897d39c9ee6cc1e601a15a02afa5c1bdfd482c5b91dfe2444` |
| V2 P0 checkpoint | Unchanged; SHA `f7f0c58d7ab4d3ec662aebc697a385c9747ab42ad0df77676aab7c962e03d299` |
| V2 P1 checkpoint | Unchanged; SHA `e1102f6c5245949a46335d61f9a054c1c9dbbdfc235f6946de1a5f3228413fd3` |
| V2 P2 checkpoint | Unchanged; SHA `4d531caf165175a4c8b5ef00b54ad09ef7effb3b5f453f0d3f28e1480263fbe7` |
| Prepared-context manifest | Unchanged; fingerprint `d9b910afd1941873826065bcf6e343be28cd850d339b356457daadbde60ad2eb` |
| REAL demo trace | Unchanged; SHA `ae39a327b0e50061bf2e6eb85126156b63dcc814c47cf54c393a69f9f86779fd` |
| ZERO demo trace | Unchanged; SHA `e10dd76c9ffbc6b3fef07d87b258c5711d3db4fc5ccc2d275558de6216d29a1e` |
| Demo bookmarks | Unchanged; SHA `3cf6b2a9911da7a6e2b4739038eab31988485e275fbdd2ec2805be1d80f4bb94` |
| Final report | Unchanged; SHA `c21aff6b9c4ac59e8a2e5908c269d07a783d8b24c28dff426885e543858261d3` |

### Directory trees

| Protected tree | Before/after result | Tree SHA-256 |
|---|---|---|
| Canonical non-TEST talk hash scope | Unchanged | `1c522cac5471804f97d8a0639f623728685ab89042d41ddaa10834bc26f677aa` |
| `data/review/` | Unchanged | `6e45672ebe3fbb5fe54ab0093ffa4cbfe73c8a9be23c0d0b6bf03bdfbbc7968d` |
| V1 TRAIN supervision | Unchanged | `1cdd1794ff715dea9f1030ddd1fe47f5fba6b0abc02bcbaa0c76b2a31dcbe82a` |
| V1 DEV supervision | Unchanged | `c2f051379aa31a8457a7db7b8a07395f5eab11e2fe791fa994c9785a7b4a30cf` |
| Expanded V1 frozen archive | Unchanged | `7db12d115496818601a96b9c4881f71775c5e21de457352dadf66a02573323d9` |
| Prepared-context pools and manifest | Unchanged | `b1a3dacae9befeb6ccda5d2f99b33ded53b51c44fd87ef3ef8a77828c39f60c3` |
| V2 predictions | Unchanged | `43dd5f7d1c780f50412227dffe6dcf78471d51efb77b9d4693b43bb5ab11f4ae` |
| V2 metrics | Unchanged | `8880d097bf3775f57ebb5a8ed8c5680f73dff38d8146dd9099681e1604ba5a5a` |
| P3 REAL/ZERO predictions | Unchanged | `10995cf1daeff8668d63a6e8e61ef7cd293c67686b1519d0a2c9d27764179b15` |
| P3 metrics | Unchanged | `df9167d2f85b7189ccef280676fd9cfad476cf28bc893f7582479c9d480b42ff` |
| `reports/` | Unchanged | `062dbfdc6e317ea62335a8f1e2458232808934b1e1e73bad08b09634d5abee1b` |
| `demo/` | Unchanged | `60070e4d9fecb90e076c6c8283c23c4d3938255fe0299f0ee452a2535f03473d` |
| `configs/` | Unchanged | `04fac899123b8b4d08d4f46f94da88d2d30084433a880110344161989b7d9afc` |
| `schemas/` | Unchanged | `934faec1d8b7addd03250dd91286cb836a0fc4bcb1e47ec7b8312bd10f88f0ec` |

Every pre-cleanup protected hash selected for post-cleanup comparison matched. Source, configs, schemas, tests, reports, demo, checkpoints, canonical data, frozen outputs, `.serena/`, `_kaggle_restore/`, and both archive ZIPs still exist.

## 4. Removed Local Artifacts

Only allowlisted generated/local state was removed:

| Removed path/category | Pre-cleanup evidence | Result |
|---|---|---|
| `venv/` | Ignored machine-local Python environment; about 4.43 GiB | Removed |
| `build/` | Ignored generated package build | Removed |
| `.pytest_cache/` | Ignored pytest cache | Removed |
| `.playwright-cli/` | Ignored browser snapshots/logs | Removed |
| `src/timelymt.egg-info/` | Ignored install metadata | Removed |
| Repository `**/__pycache__/` | 16 directories outside `venv/` | Removed |
| Repository `*.pyc` | Generated bytecode contained in allowlisted cache trees | Removed; zero remain outside `.git/` |

The selected allowlist inventory comprised **21 root/cache directories**, **33,264 files**, and **4,756,999,740 bytes**. This directory count covers the five explicit roots plus 16 separately discovered repository `__pycache__` directories; it does not attempt to count every nested subdirectory inside `venv/`.

Windows initially kept `venv/Scripts/python.exe` open through VS Code's isort language-server process. Only that disposable editor-tool process, PID 2824, was stopped; the remaining allowlisted 253,256-byte executable and `venv/` directory were then removed. No research/model process was stopped.

No embedding cache, translator cache, smoke artifact, archive, backup, notebook, scaffold, scientific output, or `.serena/` file was removed.

## 5. Disk Usage Before and After

Measurements include `.git/` and all ignored files. GiB values use 1,073,741,824 bytes.

| Measurement | Before | Final after cleanup and all three reports | Change |
|---|---:|---:|---:|
| Total repository bytes | 5,309,798,713 | 552,869,874 | -4,756,928,839 |
| Total repository GiB | 4.945135 | 0.514900 | -4.430235 |
| Working tree excluding `.git/` | 5,284,121,592 B | 527,192,753 B | -4,756,928,839 B |
| `.git/` | 25,677,121 B / 24.488 MiB | 25,677,121 B / 24.488 MiB | unchanged |
| Repository file count | 175,751 | 142,489 | net -33,262 |

The selected deletion inventory was 4,756,999,740 bytes and 33,264 files. The net values are lower by the bytes and files added for the three required Phase A documents.

## 6. Protected Areas Confirmed Untouched

The following were confirmed present after cleanup:

- `src/`, `configs/`, `schemas/`, and `tests/`.
- `data/streaming/processed/`, `data/review/`, and frozen manifests/split.
- Active V1 TRAIN/DEV supervision.
- `docs/archive/timelymt-checkpoint/`.
- V2 checkpoints, frozen config, predictions, metrics, selection, provenance, and comparison.
- P3 checkpoint, metadata, REAL/ZERO predictions, and metrics.
- `data/prepared_context/`.
- `_kaggle_restore/`.
- `outputs/demo_traces/`.
- `reports/`, `reports/figures/`, and `reports/overleaf/`.
- `demo/`.
- `docs/archive.zip` and `checkpoints/archive.zip`.
- Pseudo-label local backup and smoke artifacts.
- V2/P3 embedding caches and translator cache.
- `.serena/`.

TEST paths remained untouched. No TEST content was opened or modified, and TEST content was deliberately excluded from hashing.

## 7. Git Status

The repository was already dirty before Phase A. Phase A did not revert, stage, or modify pre-existing scientific/source changes. At final verification, the only non-ignored status entries were the three expected untracked Phase A documents; other working-tree changes observed during the earlier audit were no longer reported, apparently due to concurrent repository activity outside this task.

Final Git-visible additions are:

- `docs/artifact-preservation-manifest.md`
- `docs/artifact-preservation-manifest.json`
- `docs/cleanup-phase-a-report.md`

The deleted allowlist was ignored/generated state, so its removal does not appear as tracked deletions. `git diff --check` completed successfully with no output. `git status --short --ignored` showed the three documents above plus expected retained ignored data/output/tooling paths; it showed none of the deleted allowlist paths.

## 8. Gitignore Recommendations

`.gitignore` was not modified.

| Generated category | Current ignore coverage | Recommendation |
|---|---|---|
| `venv/` | Covered at `.gitignore:13` | Correct; retain |
| `build/` | Covered at `.gitignore:8` | Correct; retain |
| `.pytest_cache/` | Covered at `.gitignore:4` | Correct; retain |
| `.playwright-cli/` | Covered at `.gitignore:34` | Correct; retain |
| `__pycache__/` | Covered at `.gitignore:2` | Correct; retain |
| `*.pyc` | Covered by `*.py[cod]` at `.gitignore:3` | Correct; retain |
| `*.egg-info/` | Covered at `.gitignore:7` | Correct; retain |

No missing ignore pattern was found for the Phase A deletion categories. Phase B should focus instead on the distinction between ignored regenerable caches and ignored required frozen scientific outputs. In particular, V2/P3 frozen outputs need an explicit preservation/tracking or external-artifact policy without unignoring embedding/translator caches.

## 9. Deferred Cleanup Candidates

The following were explicitly deferred and confirmed retained:

- V2 embedding cache.
- P3 embedding cache.
- Translator cache.
- `_kaggle_restore/`.
- `docs/archive.zip` and `checkpoints/archive.zip`.
- `data/policy/pseudo_labels/_local_backup/`.
- Smoke labels/checkpoints/artifacts.
- Notebook variants.
- Legacy reports.
- Empty scaffolds.
- Data normalization or movement.
- Source-code reorganization/refactoring.
- V1 active/archive supervision deduplication.
- Any TEST-related cleanup or inspection.

## 10. Recommended Phase B

Phase B should be a separate, explicitly authorized Git-hygiene task. Recommended scope:

1. Decide whether canonical P3 checkpoints, final reports, demo traces, and final demo are normal Git artifacts, Git LFS artifacts, or externally registered immutable artifacts.
2. Add narrow `.gitignore` exceptions for required frozen V2/P3 predictions, metrics, selection, and provenance only after that storage decision.
3. Keep embedding caches, translator cache, local environments, build products, and bytecode ignored.
4. Record an external immutable backup location for ignored active V1 supervision and raw/intermediate Dataset v1 material.
5. Validate every proposed ignore rule with `git check-ignore -v` against both protected and regenerable examples.
6. Do not reorganize paths, deduplicate artifacts, or touch TEST in Phase B.

Phase B was not started.
