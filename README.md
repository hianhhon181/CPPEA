# CPPEA

# CPPEA – Counterfactual Probing of Persona Evidence Attribution

**Mô hình ngôn ngữ có biết thông tin thuộc về ai? Kiểm thử phản thực khả năng quy thuộc bằng chứng cá nhân trong hội thoại đa phiên**

Hoàng Hải Anh – ISL Lab, Viện Trí tuệ Nhân tạo, Trường Đại học Công nghệ – ĐHQGHN

---

## Giới thiệu

Repo này chứa mã nguồn để xây dựng một tập kiểm thử phản thực (counterfactual) trên bộ dữ liệu
**Multi-Session Chat (MSC)** và đánh giá xem mô hình ngôn ngữ có:

- dùng đúng bằng chứng persona trong bộ nhớ hội thoại hay không, và
- xác định đúng **người sở hữu** thông tin đó (Speaker 1 hay Speaker 2) hay không.

Mỗi probe là một câu hỏi trắc nghiệm 3 lựa chọn (fact đúng, fact gây nhiễu, `No information`).
Mỗi mẫu gốc sinh ra 6 biến thể:

| Biến thể | Cách tạo | Đáp án đúng |
|---|---|---|
| `original` | Giữ nguyên bộ nhớ | fact đúng |
| `removal` | Xóa fact đúng | No information |
| `subst_content` | Thay fact đúng bằng một fact mới | fact mới |
| `subst_speaker` | Chuyển fact đúng sang người nói còn lại | No information |
| `distractor_irrelevant` | Thêm 3 fact không liên quan | fact đúng |
| `distractor_other` | Thêm fact gây nhiễu vào bộ nhớ người kia | fact đúng |

Các chỉ số: **PA** (độ chính xác ban đầu), **EN** (tính cần thiết của bằng chứng),
**EI** (tính bất biến trước nhiễu), **SAA** (quy thuộc đúng chủ thể) và
**khoảng cách quy thuộc** = acc(`removal`) − acc(`subst_speaker`), có khoảng tin cậy 95% bằng bootstrap.

## Cấu trúc thư mục

```
CPPEA/
├── build_probes.py       # sinh tập probe + lọc rò rỉ (Jaccard + lọc ngữ nghĩa)
├── run_model.py          # chạy mô hình HuggingFace, chấm bằng xác suất token A/B/C
├── compute_metrics.py    # tính PA, EN, EI, SAA, khoảng cách quy thuộc, phân loại lỗi
├── export_errors.py      # xuất lỗi ra CSV để kiểm tra tay
├── calibrate_leak.py     # hiệu chỉnh ngưỡng lọc ngữ nghĩa từ file lỗi đã chấm
├── inspect_msc.py        # kiểm tra nhanh cấu trúc dữ liệu MSC
├── peek_msc.py           # xem chi tiết một bản ghi MSC
├── requirements.txt
├── data/                 # (không đưa lên git) dữ liệu MSC
│   └── msc/msc_dialogue/session_2..5/{train,valid,test}.txt
├── probes/               # tập probe bản thường (sinh ra)
├── probes_shuffled/      # tập probe bản xáo vị trí (sinh ra)
├── predictions/          # dự đoán của mô hình (sinh ra)
└── metrics/              # kết quả chỉ số (sinh ra)
```

## Cài đặt

Yêu cầu Python 3.10+ và GPU NVIDIA (đã thử trên RTX 3050 Laptop 4 GB và Kaggle T4 16 GB).

```bash
# 1. Cài PyTorch có CUDA (chọn đúng phiên bản CUDA tại https://pytorch.org)
pip install torch --index-url https://download.pytorch.org/whl/cu121

# 2. Cài các thư viện còn lại
pip install -r requirements.txt
```

Kiểm tra GPU:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## Dữ liệu

Tải MSC (bản phát hành ParlAI) và giải nén vào `data/msc/`:

```bash
mkdir -p data
curl -L -o data/msc_v0.1.tar.gz http://parl.ai/downloads/msc/msc_v0.1.tar.gz
tar -xzf data/msc_v0.1.tar.gz -C data
```

Sau khi giải nén phải có `data/msc/msc_dialogue/session_2/test.txt`. Kiểm tra bằng:

```bash
python inspect_msc.py data/msc
```

## Chạy lại thí nghiệm

### Bước 1 – Sinh tập probe (v3)

```bash
# Bản thường ("No information" luôn ở C)
python build_probes.py --semantic --sem-thr 0.45 --sem-dup 0.9 --max-base 1500 --seed 13 --out probes

# Bản xáo vị trí (cả 3 lựa chọn được xáo)
python build_probes.py --semantic --sem-thr 0.45 --sem-dup 0.9 --max-base 1500 --seed 13 --shuffle-all --out probes_shuffled
```

Đầu ra: `probes.jsonl` (9.000 probe), `manual_check.csv` (60 probe để đọc tay), `stats.txt`.

### Bước 2 – Chạy mô hình

```bash
# Chạy thử 50 probe
python run_model.py --model Qwen/Qwen2.5-1.5B-Instruct --load-4bit --limit 50

# Chạy đầy đủ, bản thường
python run_model.py --model Qwen/Qwen2.5-1.5B-Instruct --load-4bit --batch-size 4 \
    --probes probes/probes.jsonl --out-dir predictions

# Chạy đầy đủ, bản xáo vị trí
python run_model.py --model Qwen/Qwen2.5-1.5B-Instruct --load-4bit --batch-size 4 \
    --probes probes_shuffled/probes.jsonl --out-dir predictions_shuffled
```

Lặp lại với `Qwen/Qwen2.5-0.5B-Instruct`, `Qwen/Qwen2.5-3B-Instruct`, `Qwen/Qwen2.5-7B-Instruct`
(7B nên chạy trên GPU ≥ 16 GB, ví dụ Kaggle T4, với `--batch-size 8`).
Nếu bị ngắt, chạy lại đúng lệnh cũ: script sẽ bỏ qua các probe đã có kết quả.

### Bước 3 – Tính chỉ số

```bash
python compute_metrics.py "predictions/*.jsonl" --probes probes/probes.jsonl --out-dir metrics
python compute_metrics.py "predictions_shuffled/*.jsonl" --probes probes_shuffled/probes.jsonl --out-dir metrics_shuffled
```

Kết quả tổng hợp nằm ở `metrics/summary.csv`, chi tiết từng mô hình ở `metrics/<tên mô hình>.json`.

### Bước 4 (tùy chọn) – Kiểm tra rò rỉ bằng tay

```bash
# Xuất 30 lỗi removal để đọc tay, điền cột "leak (y/n)"
python export_errors.py predictions/Qwen__Qwen2.5-1.5B-Instruct.jsonl --variant removal --n 30

# Dùng file đã chấm để chọn ngưỡng lọc ngữ nghĩa
python calibrate_leak.py errors_Qwen__Qwen2.5-1.5B-Instruct_removal.csv
```

## Kết quả chính (bản thường)

| Mô hình | PA | EN | EI | SAA | removal | subst_speaker | Khoảng cách [95% CI] |
|---|---|---|---|---|---|---|---|
| Qwen2.5-0.5B | 0.785 | 0.647 | 0.832 | 0.419 | 0.596 | 0.055 | 0.540 [0.511, 0.570] |
| Qwen2.5-1.5B | 0.963 | 0.934 | 0.969 | 0.657 | 0.909 | 0.357 | 0.551 [0.527, 0.577] |
| Qwen2.5-3B | 0.709 | 0.860 | 0.934 | 0.970 | 0.960 | 0.932 | 0.027 [0.010, 0.044] |
| Qwen2.5-7B | 0.891 | 0.911 | 0.971 | 0.989 | 0.935 | 0.971 | −0.036 [−0.049, −0.023] |

Tỷ lệ gán nhầm fact đã bị chuyển sang người kia (`subst_speaker`): 93,7% / 63,0% / 6,0% / 2,2%
(bản thường) và 95,4% / 68,7% / 11,7% / 4,1% (bản xáo vị trí).
Chi tiết xem trong báo cáo.

## Hạn chế

- Ngữ cảnh là persona tóm tắt của MSC, không phải toàn bộ lời thoại.
- Còn khoảng 3,5% probe `removal` có thể bị rò rỉ sau khi lọc (ước lượng từ 30 lỗi kiểm tra tay).
- Chỉ đánh giá họ Qwen2.5, lượng tử hóa 4-bit, zero-shot, trên dữ liệu tiếng Anh.

## Trích dẫn dữ liệu và mô hình

- Xu, J., Szlam, A., Weston, J.: Beyond goldfish memory: Long-term open-domain conversation. ACL 2022.
- Qwen Team: Qwen2.5 technical report. arXiv:2412.15115 (2024).
- Reimers, N., Gurevych, I.: Sentence-BERT. EMNLP-IJCNLP 2019.
