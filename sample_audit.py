"""
Rà soát ground truth trên một mẫu NGẪU NHIÊN các probe removal và subst_speaker.

Khác với export_errors.py (chỉ lấy các lỗi của mô hình, nên thiên về trường hợp rò rỉ),
script này lấy mẫu ngẫu nhiên không phụ thuộc mô hình, để ước lượng tỷ lệ nhãn hợp lệ của cả tập.

Bước 1 – lấy mẫu và xuất file chấm tay:
    python sample_audit.py sample --n 50
    -> audit/audit_sample.csv  (50 probe removal + 50 probe subst_speaker, seed cố định)

Bước 2 – mở audit/audit_sample.csv (Excel/WPS), điền cột "valid (y/n)" cho từng dòng:
    removal       : y nếu bộ nhớ CÒN LẠI của người được hỏi không nêu và không cho phép suy ra fact đã xóa
                    (tức "No information" là đáp án đúng); n nếu còn câu gợi ý (rò rỉ).
    subst_speaker : y nếu (1) bộ nhớ của người được hỏi không nêu/không gợi ý fact, VÀ
                    (2) fact xuất hiện dưới tên người nói còn lại; n nếu một trong hai điều kiện sai.
    Ghi lý do vào cột "issue" khi chấm n (ví dụ: paraphrase, lỗi chính tả, mâu thuẫn persona).
    Lưu lại dạng CSV, giữ nguyên tên cột.

Bước 3 – tổng hợp:
    python sample_audit.py summarize audit/audit_sample.csv
    -> tỷ lệ hợp lệ theo biến thể, khoảng tin cậy 95% (Wilson), danh sách lý do.
"""
import argparse
import csv
import json
import math
import os
import random
import re
from collections import Counter, defaultdict

SPEAKER_LINE = re.compile(r"^(Speaker [12]): (.*)$")


def memory_of(context, speaker):
    out = []
    for line in context.split("\n"):
        m = SPEAKER_LINE.match(line)
        if m and m.group(1) == speaker:
            out.append(m.group(2))
    return " | ".join(out)


def sample(args):
    by_var = defaultdict(list)
    with open(args.probes, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                p = json.loads(line)
                if p["variant"] in ("removal", "subst_speaker"):
                    by_var[p["variant"]].append(p)
    rnd = random.Random(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, "audit_sample.csv")
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "variant", "n_prev_sessions", "target_speaker", "fact_bi_xoa_hoac_chuyen",
                    "bo_nho_nguoi_duoc_hoi", "bo_nho_nguoi_con_lai", "valid (y/n)", "issue"])
        for var in ("removal", "subst_speaker"):
            pool = sorted(by_var[var], key=lambda p: p["id"])
            for p in rnd.sample(pool, min(args.n, len(pool))):
                spk = p["target_speaker"]
                other = "Speaker 2" if spk == "Speaker 1" else "Speaker 1"
                w.writerow([p["id"], var, p["n_prev_sessions"], spk, p["gold_evidence"],
                            memory_of(p["context"], spk), memory_of(p["context"], other), "", ""])
    total = {v: len(by_var[v]) for v in by_var}
    print(f"Đã ghi {path}: {min(args.n, total.get('removal', 0))} removal + "
          f"{min(args.n, total.get('subst_speaker', 0))} subst_speaker "
          f"(lấy ngẫu nhiên từ {total}, seed={args.seed})")


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def summarize(args):
    counts = defaultdict(Counter)
    issues = defaultdict(Counter)
    with open(args.csv, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            lab = (r.get("valid (y/n)") or "").strip().lower()
            if lab not in ("y", "n"):
                counts[r["variant"]]["chưa chấm"] += 1
                continue
            counts[r["variant"]][lab] += 1
            if lab == "n":
                issues[r["variant"]][(r.get("issue") or "").strip() or "(không ghi lý do)"] += 1
    for var in ("removal", "subst_speaker"):
        c = counts[var]
        n = c["y"] + c["n"]
        lo, hi = wilson(c["y"], n)
        msg = f"{var:14s}: hợp lệ {c['y']}/{n}"
        if n:
            msg += f" = {c['y']/n:.1%}  [95% CI {lo:.1%}, {hi:.1%}]"
        if c["chưa chấm"]:
            msg += f"  (còn {c['chưa chấm']} dòng chưa chấm)"
        print(msg)
        for k, v in issues[var].most_common():
            print(f"    - {k}: {v}")
    allc = Counter()
    for c in counts.values():
        allc.update(c)
    n = allc["y"] + allc["n"]
    if n:
        lo, hi = wilson(allc["y"], n)
        print(f"{'tổng':14s}: hợp lệ {allc['y']}/{n} = {allc['y']/n:.1%}  [95% CI {lo:.1%}, {hi:.1%}]")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample")
    s.add_argument("--probes", default=os.path.join("probes", "probes.jsonl"))
    s.add_argument("--n", type=int, default=50, help="số probe mỗi biến thể")
    s.add_argument("--seed", type=int, default=2026)
    s.add_argument("--out-dir", default="audit")
    t = sub.add_parser("summarize")
    t.add_argument("csv")
    args = ap.parse_args()
    sample(args) if args.cmd == "sample" else summarize(args)


if __name__ == "__main__":
    main()
