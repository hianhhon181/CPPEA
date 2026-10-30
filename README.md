# CoSA: Counterfactual Speaker Attribution

**Speaker Attribution in Multi-Session Dialogue Memory: A Counterfactual Evaluation**

Hoàng Hải Anh – ISL Lab, Viện Trí tuệ Nhân tạo, Trường Đại học Công nghệ – ĐHQGHN

---

## Giới thiệu

**CoSA** là một tập kiểm thử phản thực xây dựng từ bộ dữ liệu **Multi-Session Chat (MSC)**. Tập này dùng để đánh giá
xem mô hình ngôn ngữ có gán đúng thông tin trong bộ nhớ hội thoại cho **người sở hữu** nó (Speaker 1 hay Speaker 2) hay không.

Mỗi probe là một câu hỏi trắc nghiệm 3 lựa chọn (fact đúng, fact gây nhiễu, `No information`).
Mỗi mẫu gốc sinh ra 6 biến thể:

| Biến thể | Cách tạo | Đáp án đúng | Chỉ số |
|---|---|---|---|
| `original` | Giữ nguyên bộ nhớ | fact đúng | PA |
| `removal` | Xóa fact đúng | No information | EN |
| `subst_content` | Thay fact đúng bằng một fact mới | fact mới | EN |
| `subst_speaker` | Chuyển fact đúng sang người nói còn lại | No information | SAA |
| `distractor_irrelevant` | Thêm 3 fact không liên quan | fact đúng | EI |
| `distractor_other` | Thêm fact gây nhiễu vào bộ nhớ người kia | fact đúng | EI, SAA |

Các chỉ số:
- **PA:** độ chính xác trên `original`, tính trên toàn bộ mẫu.
- **EN, EI, SAA:** tính **có điều kiện trên tập S**, tức các mẫu mô hình trả lời đúng ở `original`.
- **Khoảng cách quy thuộc:** acc(`removal`) − acc(`subst_speaker`), ghép cặp theo mẫu gốc, kèm khoảng tin cậy 95% bằng bootstrap.

## Cấu trúc thư mục

```
CoSA/
├── build_probes.py       # sinh tập probe + lọc rò rỉ (Jaccard + lọc ngữ nghĩa)
├── run_model.py          # chạy mô hình HuggingFace, chấm bằng xác suất token A/B/C
├── compute_metrics.py    # tính PA, EN, EI, SAA, khoảng cách quy thuộc, phân loại lỗi
├── reformat_memory.py    # viết lại bộ nhớ theo định dạng khác (thực nghiệm định dạng)
├── compare_formats.py    # so sánh kết quả giữa các định dạng bộ nhớ
├── sample_audit.py       # lấy mẫu ngẫu nhiên và tổng hợp rà soát ground truth
├── export_errors.py      # xuất lỗi của mô hình ra CSV để kiểm tra tay
├── calibrate_leak.py     # hiệu chỉnh ngưỡng lọc ngữ nghĩa từ file lỗi đã chấm
├── inspect_msc.py        # kiểm tra nhanh cấu trúc dữ liệu MSC
├── peek_msc.py           # xem chi tiết một bản ghi MSC
├── requirements.txt
│
│   (các thư mục dưới đây được sinh ra khi chạy, không đưa lên git)
├── data/                 # dữ liệu MSC
├── probes/               # tập probe, bản thường
├── probes_shuffled/      # tập probe, bản xáo vị trí đáp án
├── predictions/          # dự đoán của mô hình, bản thường
├── predictions_shuffled/ # dự đoán của mô hình, bản xáo
├── metrics/              # chỉ số, bản thường
├── metrics_shuffled/     # chỉ số, bản xáo
├── probes_fmt_*/  predictions_fmt_*/  metrics_formats/   # thực nghiệm định dạng bộ nhớ
├── audit/                # mẫu rà soát ground truth
└── old_v1/               # phiên bản cũ (chỉ lọc theo từ), giữ để đối chiếu
```

## Cài đặt

Yêu cầu Python 3.10+ và GPU NVIDIA (đã thử trên RTX 3050 Laptop 4 GB và Kaggle T4 16 GB).

```bash
# 1. PyTorch có CUDA (chọn đúng phiên bản CUDA tại https://pytorch.org)
pip install torch --index-url https://download.pytorch.org/whl/cu121

# 2. Các thư viện còn lại
pip install -r requirements.txt

# 3. Kiểm tra GPU
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

## Dữ liệu

Tải MSC (bản phát hành ParlAI) và giải nén vào `data/`:

```bash
mkdir data
curl -L -o data/msc_v0.1.tar.gz http://parl.ai/downloads/msc/msc_v0.1.tar.gz
tar -xzf data/msc_v0.1.tar.gz -C data
python inspect_msc.py data/msc
```

Sau khi giải nén phải có `data/msc/msc_dialogue/session_2/test.txt`.

## Chạy lại thí nghiệm

Các lệnh dưới đây chạy được cả trên PowerShell (Windows) lẫn bash. Mọi script mặc định đọc và ghi vào
các thư mục trong cấu trúc ở trên, nên không cần thêm đường dẫn.

### Bước 1 – Sinh tập probe

```bash
# Bản thường ("No information" luôn ở C)
python build_probes.py --semantic --sem-thr 0.45 --sem-dup 0.9 --max-base 1500 --seed 13 --out probes

# Bản xáo vị trí (cả 3 lựa chọn được xáo)
python build_probes.py --semantic --sem-thr 0.45 --sem-dup 0.9 --max-base 1500 --seed 13 --shuffle-all --out probes_shuffled
```

Kiểm tra: `probes/stats.txt` phải ghi 1.500 mẫu gốc, 9.000 probe, độ dài ngữ cảnh tối đa 553 từ.

### Bước 2 – Chạy mô hình

Chạy với từng mô hình `Qwen/Qwen2.5-0.5B-Instruct`, `Qwen/Qwen2.5-1.5B-Instruct`, `Qwen/Qwen2.5-3B-Instruct`,
`Qwen/Qwen2.5-7B-Instruct` (7B cần GPU ≥ 16 GB, dùng `--batch-size 8`):

```bash
python run_model.py --model Qwen/Qwen2.5-1.5B-Instruct --load-4bit --batch-size 4
python run_model.py --model Qwen/Qwen2.5-1.5B-Instruct --load-4bit --batch-size 4 --probes probes_shuffled/probes.jsonl --out-dir predictions_shuffled
```

Thêm `--limit 50` để chạy thử. Nếu bị ngắt, chạy lại đúng lệnh cũ: script sẽ bỏ qua các probe đã có kết quả.

### Bước 3 – Tính chỉ số

```bash
python compute_metrics.py "predictions/*.jsonl"
python compute_metrics.py "predictions_shuffled/*.jsonl" --probes probes_shuffled/probes.jsonl --out-dir metrics_shuffled
```

Kết quả tổng hợp ở `metrics/summary.csv`, chi tiết từng mô hình ở `metrics/<tên mô hình>.json`.

### Bước 4 – Thực nghiệm định dạng bộ nhớ

So sánh định dạng gốc (`Speaker 1: I was brought up in Alaska.`) với hai định dạng ghi người sở hữu trong từng câu:
`sentence_prefix` (nhãn trước mỗi câu) và `third_person` (`Speaker 1 was brought up in Alaska.`).
Chỉ bộ nhớ thay đổi; câu hỏi và lựa chọn giữ nguyên. Dùng 3 biến thể `original`, `removal`, `subst_speaker`.

```bash
python reformat_memory.py --format sentence_prefix
python reformat_memory.py --format third_person      # tạo thêm conversion_check.csv để kiểm tra ngữ pháp bằng tay

python run_model.py --model Qwen/Qwen2.5-1.5B-Instruct --load-4bit --batch-size 4 --probes probes_fmt_sentence_prefix/probes.jsonl --out-dir predictions_fmt_sentence_prefix
python run_model.py --model Qwen/Qwen2.5-1.5B-Instruct --load-4bit --batch-size 4 --probes probes_fmt_third_person/probes.jsonl --out-dir predictions_fmt_third_person
# lặp lại hai lệnh trên với Qwen/Qwen2.5-3B-Instruct

python compare_formats.py --models Qwen__Qwen2.5-1.5B-Instruct Qwen__Qwen2.5-3B-Instruct
```

Định dạng gốc dùng lại kết quả trong `predictions/`. Ở dòng `line_prefix`, PA phải khớp với kết quả chính
(0.963 với 1.5B, 0.709 với 3B).

### Bước 5 – Rà soát ground truth trên mẫu ngẫu nhiên

```bash
python sample_audit.py sample --n 50          # 50 removal + 50 subst_speaker, chọn ngẫu nhiên (seed 2026)
# chấm cột "valid (y/n)" và ghi lý do vào cột "issue" trong audit/audit_sample.csv
python sample_audit.py summarize audit/audit_sample.csv
```

## Kết quả chính

**Kết quả chính (thứ tự đáp án thường; EN, EI, SAA, removal, subst_speaker tính trên S):**

| Mô hình | PA | EN | EI | SAA | removal | subst_speaker | Khoảng cách [95% CI] |
|---|---|---|---|---|---|---|---|
| Qwen2.5-0.5B | 0.785 | 0.647 | 0.832 | 0.419 | 0.596 | 0.055 | 0.540 [0.511, 0.570] |
| Qwen2.5-1.5B | 0.963 | 0.934 | 0.969 | 0.657 | 0.909 | 0.357 | 0.551 [0.527, 0.577] |
| Qwen2.5-3B | 0.709 | 0.860 | 0.934 | 0.970 | 0.960 | 0.932 | 0.027 [0.010, 0.044] |
| Qwen2.5-7B | 0.891 | 0.911 | 0.971 | 0.989 | 0.935 | 0.971 | −0.036 [−0.049, −0.023] |

Tỷ lệ gán nhầm fact đã bị chuyển sang người kia (`subst_speaker`): 93,7% / 63,0% / 6,0% / 2,2% (bản thường),
95,4% / 68,7% / 11,7% / 4,1% (bản xáo vị trí).

**Định dạng bộ nhớ (tỷ lệ gán nhầm ở `subst_speaker`, trên S):**

| Mô hình | Gốc | Nhãn trước mỗi câu | Ngôi thứ ba |
|---|---|---|---|
| Qwen2.5-1.5B | 63,0% | 53,8% | 43,9% |
| Qwen2.5-3B | 6,0% | 6,5% | 4,6% |

**Rà soát ground truth:** 94/100 probe lấy ngẫu nhiên được chấm hợp lệ (47/50 `removal`, 47/50 `subst_speaker`).

Chi tiết xem trong bài báo.

## Hạn chế

- Ngữ cảnh là persona tóm tắt của MSC, không phải toàn bộ lời thoại.
- Chỉ đánh giá họ Qwen2.5-Instruct, lượng tử hóa 4-bit, zero-shot, trên dữ liệu tiếng Anh.
- Khoảng 6% probe `removal`/`subst_speaker` có nhãn không hợp lệ trên mẫu rà soát (một người chấm).

## Trích dẫn dữ liệu và mô hình

- Xu, J., Szlam, A., Weston, J.: Beyond goldfish memory: Long-term open-domain conversation. ACL 2022.
- Qwen Team: Qwen2.5 technical report. arXiv:2412.15115 (2024).
- Reimers, N., Gurevych, I.: Sentence-BERT. EMNLP-IJCNLP 2019.
