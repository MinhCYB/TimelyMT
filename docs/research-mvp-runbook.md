# Research MVP Full-Run Runbook

This runbook starts the expensive, publishable experiment. It is not a smoke
workflow. Run commands from the repository root in PowerShell. The verified
environment and large Hugging Face cache remain on `D:`.

```powershell
$PY = "D:\TimelyMT-runtime-cache\venv\Scripts\python.exe"
$env:HF_HOME = "D:\TimelyMT-runtime-cache\hf"
$env:HF_HUB_CACHE = "D:\TimelyMT-runtime-cache\hf\hub"
$env:TEMP = "D:\TimelyMT-runtime-cache\tmp"
$env:TMP = "D:\TimelyMT-runtime-cache\tmp"
```

All timing output is simulated source-clock timing, not acoustic latency.
Prediction generation is source-only. Only the explicit `evaluate` stages load
canonical Vietnamese references.

## STEP 1 - Preflight / Tests

```powershell
& $PY -m unittest tests.research.test_research_mvp tests.translator.test_translator tests.data.test_translation_artifacts -v
& $PY -m unittest discover -s tests -v
& $PY -m compileall -q src tests
git diff --check
```

Expected: all tests pass. Dataset and translator checksum assertions run as
part of the artifact tests and every experiment CLI stage.

## STEP 2 - Full TRAIN Pseudo-Label Generation

```powershell
& $PY -m timelymt.research.cli pseudo --split train --batch-size 3
```

Expected artifacts: one atomic JSONL per talk under
`data/policy/pseudo_labels/train/` and
`data/policy/pseudo_labels/train/manifest.json`. Existing valid talk JSONL files
are resume hits. Do not pass any limit or `--smoke`; the final manifest must say
`artifact_status: full` and `publishable: true`.

## STEP 3 - Full DEV Pseudo-Label Generation

```powershell
& $PY -m timelymt.research.cli pseudo --split dev --batch-size 3
```

Expected artifacts: `data/policy/pseudo_labels/dev/*.jsonl` and
`data/policy/pseudo_labels/dev/manifest.json` with full 3-talk coverage.

## STEP 4 - Validate Pseudo-Label Statistics

```powershell
& $PY -m timelymt.research.cli validate-pseudo --split train
& $PY -m timelymt.research.cli validate-pseudo --split dev
```

Expected output: state, LISTEN, COMMIT, and talk counts. Both manifests must be
`full`. TEST is rejected by this command and by pseudo-label generation.

## STEP 5 - Train P0

```powershell
& $PY -m timelymt.research.cli train --pseudo-labels data/policy/pseudo_labels/train/manifest.json --variant P0
```

Expected: `checkpoints/policy/P0.joblib` and
`checkpoints/policy/P0.metadata.json`.

## STEP 6 - Train P1

```powershell
& $PY -m timelymt.research.cli train --pseudo-labels data/policy/pseudo_labels/train/manifest.json --variant P1
```

Expected: `checkpoints/policy/P1.joblib` and
`checkpoints/policy/P1.metadata.json`.

## STEP 7 - Train P2

```powershell
& $PY -m timelymt.research.cli train --pseudo-labels data/policy/pseudo_labels/train/manifest.json --variant P2
```

Expected: `checkpoints/policy/P2.joblib` and
`checkpoints/policy/P2.metadata.json`. All final checkpoint metadata must say
`artifact_status: full` and include a SHA-256 checksum.

## STEP 8 - DEV Baseline Rollouts

```powershell
& $PY -m timelymt.research.cli rollout --split dev --batch-size 3 --strategies fixed_n_4 fixed_n_8 fixed_n_12 fixed_time_1600 fixed_time_3200 fixed_time_4800 local_agreement_style_k2 local_agreement_style_k3
```

Expected: source-only prediction artifacts under
`outputs/experiments/research-mvp/predictions/dev/<strategy>/`.

## STEP 9 - DEV Learned Rollouts at All Thresholds

```powershell
& $PY -m timelymt.research.cli rollout --split dev --batch-size 3 --strategies learned_P0_0.30 learned_P0_0.40 learned_P0_0.50 learned_P0_0.60 learned_P0_0.70 learned_P1_0.30 learned_P1_0.40 learned_P1_0.50 learned_P1_0.60 learned_P1_0.70 learned_P2_0.30 learned_P2_0.40 learned_P2_0.50 learned_P2_0.60 learned_P2_0.70
```

Expected: 15 learned strategy directories under
`outputs/experiments/research-mvp/predictions/dev/`. These are real sequential
rollouts; policy commits determine the next buffer and system history.

## STEP 10 - DEV Evaluation

```powershell
& $PY -m timelymt.research.cli evaluate --split dev --strategies fixed_n_4 fixed_n_8 fixed_n_12 fixed_time_1600 fixed_time_3200 fixed_time_4800 local_agreement_style_k2 local_agreement_style_k3 learned_P0_0.30 learned_P0_0.40 learned_P0_0.50 learned_P0_0.60 learned_P0_0.70 learned_P1_0.30 learned_P1_0.40 learned_P1_0.50 learned_P1_0.60 learned_P1_0.70 learned_P2_0.30 learned_P2_0.40 learned_P2_0.50 learned_P2_0.60 learned_P2_0.70
```

Expected: per-strategy files and
`outputs/experiments/research-mvp/metrics/dev/all.json`.

## STEP 11 - Deterministic DEV Selection

```powershell
& $PY -m timelymt.research.cli select
```

Expected: `outputs/experiments/research-mvp/dev-selection.json`. Selection uses
the preregistered fixed_n_8 Average Lagging constraint, then chrF2, BLEU, lower
AL, and higher threshold tie-breaks.

## STEP 12 - Freeze Final-Eval Config

```powershell
& $PY -m timelymt.research.cli freeze
```

Expected: `outputs/experiments/research-mvp/frozen-eval-config.json` and its
printed SHA-256. The command refuses incomplete pseudo manifests, smoke
checkpoints, missing selection, and overwrite attempts.

## STEP 13 - TEST Baseline Prediction Generation

```powershell
& $PY -m timelymt.research.cli rollout --split test --batch-size 3 --strategies fixed_n_4 fixed_n_8 fixed_n_12 fixed_time_1600 fixed_time_3200 fixed_time_4800 local_agreement_style_k2 local_agreement_style_k3
```

Expected: source-only TEST predictions under
`outputs/experiments/research-mvp/predictions/test/`. This command is gated on
the frozen final-eval config and does not load references.

## STEP 14 - Selected Learned TEST Prediction

```powershell
& $PY -m timelymt.research.cli rollout-selected --split test --batch-size 3
```

Expected: the DEV-selected learned strategy's TEST prediction directory. The
strategy and threshold are read from `dev-selection.json`, not chosen on TEST.

## STEP 15 - TEST Evaluation

Run this only after all STEP 13 and STEP 14 prediction files exist.

```powershell
& $PY -m timelymt.research.cli evaluate --split test --include-selected --strategies fixed_n_4 fixed_n_8 fixed_n_12 fixed_time_1600 fixed_time_3200 fixed_time_4800 local_agreement_style_k2 local_agreement_style_k3
```

Expected: per-strategy TEST metrics and
`outputs/experiments/research-mvp/metrics/test/all.json`. This is the first TEST
stage that loads Vietnamese references.

## STEP 16 - Report Table Generation

```powershell
& $PY -m timelymt.research.cli report --split test
```

Expected: `outputs/experiments/research-mvp/test-results.csv` with quality,
standard token-level Average Lagging, and mandatory simulated source-clock
latency fields, plus
`outputs/experiments/research-mvp/test-per-talk-results.csv` with preserved
per-talk BLEU and chrF2.

After the full run, rerun STEP 1. Preserve per-talk metrics from each strategy's
JSON. With two TEST talks, report a controlled project evaluation and do not
claim statistical significance.
