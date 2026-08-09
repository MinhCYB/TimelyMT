# TimelyMT

**Nghiên cứu mô hình quyết định thời điểm dịch cho hệ thống dịch cabin Anh–Việt**

TimelyMT nghiên cứu quyết định **LISTEN** (tiếp tục nhận source) hay **COMMIT** (chốt translation hypothesis hiện tại) trên luồng transcript tiếng Anh đến tăng dần. Translator Anh-Việt là một thành phần pretrained/frozen; trọng tâm nghiên cứu là streaming commit policy, không phải huấn luyện mô hình dịch.

Dataset trung tâm là các bilingual talk liên tục theo phong cách TED. Dữ liệu streaming được dùng để tạo source prefixes, translation hypotheses, pseudo-labels, và đánh giá các heuristic cùng learned policy.

**Current milestone:** M1.3 - Dataset-safe causal translation requests and hypothesis artifacts

The resumable high-level dataset commands are `make prepare-dataset`, `make validate-dataset`, and `make dataset-summary`. M0.8 gate status and limitations are recorded in [`docs/dataset-quality-report.md`](docs/dataset-quality-report.md).

Translator fine-tuning là một extension tùy chọn trong tương lai, không thuộc scope hiện tại.

The M1 translator contract and frozen EnViT5 integration are documented in
[`docs/translator.md`](docs/translator.md).
The leakage-safe Dataset v1 request and derived-hypothesis contracts are
documented in [`docs/translation-artifacts.md`](docs/translation-artifacts.md).

## Repository Structure

```text
configs/                 Future configuration by research concern
data/streaming/          Raw, aligned, and processed continuous talk data
data/policy/             Derived prefixes, hypotheses, and pseudo-labels
src/timelymt/data/       Future acquisition, parsing, alignment, and validation modules
src/timelymt/translator/ Frozen translator abstraction
training/                Future pseudo-labeling and policy training entrypoints
experiments/             Baseline, model comparison, and policy ablation definitions
checkpoints/policy/      Learned policy checkpoints only
outputs/                 Generated runs, logs, predictions, metrics, and figures
demo/                    Future visualization surfaces
tests/                   Tests organized by research area
```

## Roadmap

1. M0 - Streaming Dataset: acquisition, parsing, EN-VI alignment, timing normalization, talk-level splits, and validation.
2. M1 - Frozen Translator Wrapper.
3. M2 - Streaming Baselines: Fixed-N, Fixed-Time, and LocalAgreement.
4. M3 - Pseudo-label Generation.
5. M4 - Learned LISTEN/COMMIT Policy.
6. M5 - History + Context Ablation.
7. M6 - Demo + Final Evaluation.
