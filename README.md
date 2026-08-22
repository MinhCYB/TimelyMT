# TimelyMT

## 1. Overview

TimelyMT nghiên cứu khi nào hệ thống dịch streaming Anh–Việt nên phát hành một
đơn vị dịch đã được chốt và không thể chỉnh sửa lại. Mục tiêu dài hạn là dịch
cabin tự động Anh–Việt với độ trễ thấp; phạm vi hiện tại là policy trên luồng
token quyết định **WAIT**, **LISTEN** hoặc **COMMIT**. EnViT5 được giữ cố định
và chỉ nhận nguồn; prepared context chỉ tác động đến policy, không đi trực tiếp
vào EnViT5. Dự án hiện chưa phải hệ thống speech-to-speech đầu cuối.

```text
source tokens
    -> current source buffer
    -> frozen EnViT5
    -> candidate translation
    -> timing policy
    -> WAIT / LISTEN / COMMIT
```

## 2. Repository structure

| Path | Vai trò |
|---|---|
| `src/timelymt/research/` | V1/V2/P3 policy, rollout, evaluation và research CLI |
| `src/timelymt/data/` | Chuẩn bị, kiểm tra và nạp dataset streaming |
| `scripts/` | Bootstrap V2, báo cáo chỉ đọc và kiểm tra trace |
| `configs/` | Cấu hình dataset, frozen EnViT5 và thí nghiệm |
| `data/` | Dataset v1, split, V1 supervision và prepared-context pools |
| `checkpoints/` | Checkpoint V2 và frozen `P3_GLOBAL` được cung cấp sẵn |
| `outputs/` | DEV predictions, metrics, cache và demo traces |
| `demo/` | Static artifact-replay viewer |
| `reports/` | Báo cáo và hình đã sinh; không phải hướng dẫn chạy |
| `tests/` | Kiểm thử offline |
| `docs/` | Đặc tả, runbook và frozen V1 archive |

## 3. Running the full reproduction pipeline

Chạy từ repository root bằng Windows PowerShell. Workflow thông thường sử dụng
các artifact đã được freeze; không cần huấn luyện lại V1, V2 hoặc P3. Các bước
rollout và inference sẽ tải model nếu cache chưa có; GPU CUDA được khuyến nghị
để giảm thời gian chạy.

### Environment setup

Purpose:
Cài package Python 3.10+ và các dependency khai báo trong repository.

Command:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install --editable .
```

Output:
`.venv/` và editable package `timelymt`.

### Frozen dataset validation

Purpose:
Kiểm tra dataset streaming hiện có mà không tải hoặc dựng lại dữ liệu.

Command:

```powershell
python -m timelymt.data.pipeline.cli validate
```

Output:
Kết quả validation trên stdout cho `data/manifests/streaming-dataset.json`.

### V1 frozen supervision import

Purpose:
Khôi phục và xác minh V1 baseline/supervision đã freeze thay vì tái tạo pseudo labels hoặc huấn luyện lại V1.

Command:

```powershell
python -m timelymt.research.cli import-v1 --source docs/archive/timelymt-checkpoint
```

Output:
`data/policy/pseudo_labels/{train,dev}/` và `outputs/experiments/policy-v2/v1-source/`.

### V2 semantic policy

Purpose:
Chạy coordinator DEV resume-aware cho frozen MiniLM + MLP P0/P1/P2; checkpoint hợp lệ được cung cấp sẵn sẽ được bỏ qua.

Command:

```powershell
python scripts/policy_v2_bootstrap.py
```

Output:
`checkpoints/policy_v2/` và `outputs/experiments/policy-v2/`.

### P3_GLOBAL checkpoint validation

Purpose:
Xác minh prepared-context pools và xem metadata của frozen P3 checkpoint được cung cấp sẵn; không chạy `train-p3` trong workflow khuyến nghị.

Command:

```powershell
python -m timelymt.research.cli validate-p3
python -m timelymt.research.cli inspect-p3-checkpoint
```

Output:
Kết quả validation của `data/prepared_context/manifest.json` và metadata của
`checkpoints/policy_p3_global/P3_GLOBAL.pt` trên stdout.

### P3_GLOBAL DEV rollout and evaluation

Purpose:
Chạy grid REAL_CONTEXT trên ba DEV talks rồi tính quality/latency metrics.

Command:

```powershell
python -m timelymt.research.cli rollout-p3 --split dev --thresholds 0.30 0.40 0.50 0.60 0.70 --batch-size 1
python -m timelymt.research.cli evaluate-p3 --split dev
```

Output:
`outputs/experiments/policy-p3-global/predictions/dev/p3_global_*/` và `outputs/experiments/policy-p3-global/metrics/dev/all.json`.

### REAL_CONTEXT vs ZERO_CONTEXT ablation

Purpose:
Giữ nguyên checkpoint P3 và chỉ thay prepared-global policy slice bằng vector 0
để tạo controlled DEV ablation. Prepared context có thể làm thay đổi hành vi
COMMIT của P3, nhưng `prepared-global-v0` chưa cho thấy cải thiện ổn định về
chất lượng hoặc độ trễ qua các ngưỡng.

Command:

```powershell
python -m timelymt.research.cli rollout-p3 --split dev --thresholds 0.30 0.40 0.50 0.60 0.70 --batch-size 1 --prepared-context-mode zero
python -m timelymt.research.cli evaluate-p3 --split dev --strategies p3_global_zeroctx_0.30 p3_global_zeroctx_0.40 p3_global_zeroctx_0.50 p3_global_zeroctx_0.60 p3_global_zeroctx_0.70
python scripts/report_p3_prepared_context_ablation.py
```

Output:
`outputs/experiments/policy-p3-global/metrics/dev/all-zeroctx.json` và `reports/p3_prepared_context_ablation.{md,json}`.

### Stored trace artifacts

Purpose:
Kiểm tra hai trace DEV Sims 0.60 được cung cấp sẵn; workflow khuyến nghị không chạy lại model inference để tái tạo chúng.

Command:

```powershell
python scripts/validate_demo_trace.py outputs/demo_traces/sims-real-0.60.json
python scripts/validate_demo_trace.py outputs/demo_traces/sims-zero-0.60.json
```

Output:
Các artifact đã xác minh tại `outputs/demo_traces/sims-{real,zero}-0.60.json`; bookmarks nằm tại `outputs/demo_traces/sims-0.60-bookmarks.json`.

TEST chỉ được mở đúng một lần sau khi thiết kế nghiên cứu đã được cố định và
không có tuning nào được thực hiện sau đó. TEST không thuộc workflow tái lập
thông thường và không phải bước để người dùng chạy lặp lại.

## 4. Demo

Demo phát lại các DEV trace đã lưu cho ví dụ Sims 0.60 và hiển thị source stream,
candidate translation, `p(COMMIT)`, quyết định của policy cùng bản dịch tiếng
Việt đã COMMIT.

Phân kỳ trực tiếp bắt đầu tại Event 131 và làm hai buffer khác nhau từ Event 132
trở đi. Demo không chạy training, rollout hay model inference.

Từ repository root:

```powershell
python -m http.server 8000
```

Mở [http://localhost:8000/demo/](http://localhost:8000/demo/). Dừng server bằng
`Ctrl+C` trong terminal đang chạy lệnh.
