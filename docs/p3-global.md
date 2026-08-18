# P3-GLOBAL

P3-GLOBAL is a separate experimental policy variant. It does not change P0, P1, P2, V1 supervision, EnViT5 requests, or streaming semantics.

## Architecture

`PreparedContextPool -> independently encoded eligible sources -> equal mean + L2 normalization -> P3 feature vector -> P3 MLP -> p(COMMIT)`

Only `SAFE_PRETALK_CONFIRMED`, pre-talk, non-transcript, non-reference sources contribute. Sources are ordered by `source_id`, encoded independently with pinned multilingual MiniLM, then equally averaged and L2-normalized. One source is used directly. An empty eligible pool produces exactly `float32 zeros(384)`.

The fixed 1547-dimensional order is: current source 384, previous committed source 384, previous generated target 384, prepared global 384, then the unchanged scaled 11 numeric features. `has_eligible_context` is provenance only, never a feature.

P3 joins persisted V1 supervision through the existing explicit `talk_id` and `split` fields. One prepared embedding is built per talk and reused across its rows and all streaming decisions. Missing pools and talk/split mismatches fail; valid empty pools are supported.

## Execution Backend

Model semantics are unchanged; execution backend is accelerated. P3 resolves its independent `runtime.encoder_device` and `runtime.policy_device` values as `cuda` when available and `cpu` otherwise. The pinned frozen MiniLM remains FP32, in evaluation mode with gradients disabled, and encodes exact-text cache misses in batches. Cache values and the final 1547-dimensional P3 feature matrix remain `float32`.

P3 first gathers all eligible prepared-source and TRAIN causal texts, deduplicates them by the existing exact cache identity, materializes cache misses in encoder batches, then reuses those values to build one prepared-global-v0 embedding per talk and the feature matrix. Sources are never concatenated. CUDA changes can produce ordinary floating-point differences from CPU; numerical bitwise equivalence is not claimed.

## Commands

All commands below are researcher-operated and were **not executed** for this task.

```powershell
$env:PYTHONPATH='src'; python -m timelymt.research.cli validate-p3
$env:PYTHONPATH='src'; python -m timelymt.research.cli inspect-p3 --split dev --talk-id ted-sims-witherspoon-ai-climate
$env:PYTHONPATH='src'; python -m timelymt.research.cli train-p3
$env:PYTHONPATH='src'; python -m timelymt.research.cli inspect-p3-checkpoint
$env:PYTHONPATH='src'; python -m timelymt.research.cli rollout-p3 --split dev --talk-id ted-sims-witherspoon-ai-climate --thresholds 0.50 --batch-size 1
$env:PYTHONPATH='src'; python -m timelymt.research.cli rollout-p3 --split dev --thresholds 0.30 0.40 0.50 0.60 0.70 --batch-size 1
$env:PYTHONPATH='src'; python -m timelymt.research.cli evaluate-p3 --split dev
```

Evaluation uses the existing BLEU, chrF2, AL, LAAL, and commit metrics. P3 outputs are named `p3_global_0.30` through `p3_global_0.70`.

No TEST results exist. Do not run TEST. Current prepared-context coverage is sparse: five TRAIN talks and one DEV talk have eligible sources; valid empty pools are intentional.
