# TimelyMT

TimelyMT nghiên cứu dịch tăng dần Anh-Việt cho các bài nói kỹ thuật. Hệ thống nhận
luồng tiếng Anh theo thời gian và liên tục cập nhật giả thuyết dịch tiếng Việt.

Vấn đề trung tâm là quyết định khi nào hệ thống nên **LISTEN** để chờ thêm thông tin
nguồn, hoặc **COMMIT** để chốt và công bố bản dịch hiện tại. EnViT5 được giữ cố định;
nghiên cứu tập trung vào policy LISTEN / COMMIT thay vì huấn luyện lại mô hình dịch.

## Mục tiêu

Mục tiêu là đánh giá liệu causal learned commit policy có đạt cân bằng chất lượng-độ
trễ tốt hơn các policy cố định và thích nghi dựa trên nghiên cứu trước hay không. Phạm
vi so sánh gồm Fixed baselines, Local Agreement, Meaningful Unit adaptation của
Zhang et al. 2020 và các biến thể TimelyMT P0/P1/P2.

## Kiến trúc

```text
Dataset
  ↓
Causal English stream
  ↓
Frozen EnViT5
  ↓
Translation hypotheses
  ↓
LISTEN / COMMIT policy
  ↓
Committed Vietnamese translation
  ↓
Evaluation
```

Tham chiếu tiếng Việt chỉ được dùng cho đánh giá, không được đưa vào đầu vào causal
của policy.

## Cài đặt

Dự án yêu cầu Python từ 3.10; Python 3.10 là môi trường đã được dự án xác minh.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e .
```

GPU/CUDA được khuyến nghị khi suy luận EnViT5, nhưng không bắt buộc cho mọi phần của
dự án.

## Kiểm tra nhanh

Kiểm tra package sau khi cài đặt:

```powershell
python -c "import timelymt; print('timelymt import OK')"
```

Chạy toàn bộ kiểm thử offline:

```powershell
python -m unittest discover -s tests -p "test_*.py" -v
```

Chạy translator smoke CLI; lần đầu có thể tải model nếu cache cục bộ chưa có:

```powershell
python -m timelymt.translator.cli --text "Artificial intelligence is changing the world." --device auto
```

## Cấu trúc repository

```text
TimelyMT/
├── configs/          # cấu hình dữ liệu, translator và thí nghiệm
├── data/             # dataset, manifest, split và artifact nghiên cứu
├── schemas/          # schema dữ liệu JSON
├── src/timelymt/
│   ├── data/         # pipeline chuẩn bị dataset
│   ├── translator/   # wrapper frozen EnViT5
│   └── research/     # baseline, policy và evaluation
├── tests/            # kiểm thử offline
├── notebooks/        # Kaggle runner
├── checkpoints/      # checkpoint policy sinh khi chạy
├── outputs/          # output và cache sinh khi chạy
├── pyproject.toml    # package và dependency Python
└── README.md         # hướng dẫn ngắn của dự án
```

## Chạy thí nghiệm trên Kaggle

Workflow hiện có nằm tại `notebooks/kaggle-research-mvp.ipynb`:

1. Upload repository này dưới dạng Kaggle Dataset.
2. Tạo hoặc mở một Kaggle notebook.
3. Thêm Dataset chứa repository vào input.
4. Bật GPU và Internet.
5. Mở và chạy `notebooks/kaggle-research-mvp.ipynb`.
6. Chạy các cell tuần tự.
7. Dừng tại cell `STOP BEFORE TEST`.
8. Tải archive artifact nghiên cứu đã được notebook export.

Có thể giảm `INFERENCE_BATCH_SIZE` xuống 2 hoặc 1 nếu bộ nhớ GPU không đủ.

## Demo dự kiến

Demo chưa được triển khai. Giao diện dự kiến trình bày luồng xử lý:

```text
English stream
→ current translation hypothesis
→ LISTEN / COMMIT decision
→ committed Vietnamese subtitle
```

Thông tin có thể hiển thị gồm văn bản tiếng Anh đang đến, giả thuyết tiếng Việt hiện
tại, quyết định policy, `P(COMMIT)` khi áp dụng và bản dịch đã được chốt.

## Trạng thái

- Dataset v1 và frozen translator đã có trong repository.
- Research scaffold, Fixed baselines, Local Agreement, Meaningful Unit và P0/P1/P2 đã có.
- Kaggle notebook đã sẵn sàng để chạy workflow đầy đủ đến trước TEST.
- Full Kaggle experiment còn chờ chạy; held-out TEST chưa được thực thi.
