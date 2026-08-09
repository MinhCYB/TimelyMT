# Research MVP Full-Run Runbook

This is the later publishable Kaggle workflow, not the local smoke workflow.
Run from the repository root. Prediction stages are source-only; only
`evaluate` loads Vietnamese references. All clock values are simulated source
clock, not acoustic latency.

```powershell
$PY = "D:\TimelyMT-runtime-cache\venv\Scripts\python.exe"
$env:HF_HOME = "D:\TimelyMT-runtime-cache\hf"
$env:HF_HUB_CACHE = "D:\TimelyMT-runtime-cache\hf\hub"
```

## 1. Preflight

```powershell
& $PY -m unittest tests.research.test_literature_benchmarks tests.research.test_research_mvp tests.research.test_kaggle_runner tests.translator.test_translator -v
& $PY -m unittest discover -s tests -v
& $PY -m compileall -q src tests
git diff --check
```

## 2. TimelyMT TRAIN/DEV Pseudo Labels

```powershell
& $PY -m timelymt.research.cli pseudo --split train --batch-size 3
& $PY -m timelymt.research.cli pseudo --split dev --batch-size 3
& $PY -m timelymt.research.cli validate-pseudo --split train
& $PY -m timelymt.research.cli validate-pseudo --split dev
```

Existing valid per-talk files are resume hits. Full manifests must report
`artifact_status: full`. Do not pass limits or `--smoke`.

## 3. MU TRAIN/DEV Supervision

```powershell
& $PY -m timelymt.research.cli mu-supervision --split train --batch-size 3
& $PY -m timelymt.research.cli mu-supervision --split dev --batch-size 3
& $PY -m timelymt.research.cli validate-mu --split train
& $PY -m timelymt.research.cli validate-mu --split dev
```

These independent artifacts are written under `data/policy/mu_zhang2020/`.
The full admissible remaining-unit translation (at most 48 source tokens) is
training/dev oracle data only. Both
supervision commands reject TEST.

## 4. Train P0/P1/P2

```powershell
& $PY -m timelymt.research.cli train --pseudo-labels data/policy/pseudo_labels/train/manifest.json --variant P0
& $PY -m timelymt.research.cli train --pseudo-labels data/policy/pseudo_labels/train/manifest.json --variant P1
& $PY -m timelymt.research.cli train --pseudo-labels data/policy/pseudo_labels/train/manifest.json --variant P2
```

## 5. Train MU

```powershell
& $PY -m timelymt.research.cli train-mu --pseudo-labels data/policy/mu_zhang2020/train/manifest.json
```

Final training refuses smoke or partial inputs. Checkpoint metadata under
`checkpoints/policy/` must report `artifact_status: full`.

## 6. DEV Fixed Baselines

```powershell
& $PY -m timelymt.research.cli rollout --split dev --batch-size 3 --strategies fixed_n_4 fixed_n_8 fixed_n_12 fixed_time_1600 fixed_time_3200 fixed_time_4800
```

## 7. DEV TimelyMT Local-Agreement-Style Heuristics

```powershell
& $PY -m timelymt.research.cli rollout --split dev --batch-size 3 --strategies local_agreement_style_k2 local_agreement_style_k3
```

## 8. DEV Local Agreement LA-2 Adaptation

```powershell
& $PY -m timelymt.research.cli rollout --split dev --batch-size 3 --strategies local_agreement_la2
```

## 9. DEV MU Rollout

```powershell
& $PY -m timelymt.research.cli rollout --split dev --batch-size 3 --strategies mu_zhang2020
```

## 10. DEV P0/P1/P2 Rollouts

```powershell
& $PY -m timelymt.research.cli rollout --split dev --batch-size 3 --strategies learned_P0_0.30 learned_P0_0.40 learned_P0_0.50 learned_P0_0.60 learned_P0_0.70 learned_P1_0.30 learned_P1_0.40 learned_P1_0.50 learned_P1_0.60 learned_P1_0.70 learned_P2_0.30 learned_P2_0.40 learned_P2_0.50 learned_P2_0.60 learned_P2_0.70
```

These are sequential causal rollouts. Policy decisions determine subsequent
buffers and, for TimelyMT P1/P2 only, system history.

## 11. DEV Evaluation

```powershell
& $PY -m timelymt.research.cli evaluate --split dev --strategies fixed_n_4 fixed_n_8 fixed_n_12 fixed_time_1600 fixed_time_3200 fixed_time_4800 local_agreement_style_k2 local_agreement_style_k3 local_agreement_la2 mu_zhang2020 learned_P0_0.30 learned_P0_0.40 learned_P0_0.50 learned_P0_0.60 learned_P0_0.70 learned_P1_0.30 learned_P1_0.40 learned_P1_0.50 learned_P1_0.60 learned_P1_0.70 learned_P2_0.30 learned_P2_0.40 learned_P2_0.50 learned_P2_0.60 learned_P2_0.70
```

Per-strategy metrics include SacreBLEU, chrF2, AL, LAAL, commits per 100 source
tokens, mean/median source tokens per unit, mean simulated source-clock unit
duration, first-commit source tokens and simulated-clock latency, and forced
commit rate.

## 12. TimelyMT DEV Selection

```powershell
& $PY -m timelymt.research.cli select
```

The deterministic rule remains unchanged: only `learned_P0/P1/P2` candidates
participate, constrained against `fixed_n_8` AL and selected by the registered
quality/tie-break rule. MU and LA-2 are comparison baselines only.

## 13. Freeze Config

```powershell
& $PY -m timelymt.research.cli freeze
```

This requires full TimelyMT and MU TRAIN/DEV manifests, full P0/P1/P2 and MU
checkpoints, and full DEV selection. It refuses overwrite.

# STOP BEFORE TEST

Export `data/policy/pseudo_labels`, `data/policy/mu_zhang2020`,
`checkpoints/policy`, and `outputs/experiments/research-mvp`. Do not run TEST
pseudo labels, MU supervision, predictions, evaluation, selected rollout, or
reporting in this Kaggle execution.
