"""
Hiệu chỉnh ngưỡng lọc ngữ nghĩa (--sem-thr) cho build_probes.py, dựa trên file lỗi đã chấm tay.

Với mỗi dòng đã chấm (leak y/n), tính cosine lớn nhất giữa fact đã bị xóa và từng câu
còn lại trong ngữ cảnh. Một ngưỡng tốt sẽ bắt được hầu hết dòng "y" (rò rỉ).
Đồng thời ước lượng: với mỗi ngưỡng, bao nhiêu % mẫu gốc hiện tại sẽ bị loại.

Cần: pip install sentence-transformers

Cách chạy:
    python calibrate_leak.py errors_Qwen__Qwen2.5-7B-Instruct_removal.csv
"""
import argparse
import csv
import json
import os
import random
import re


def split_sentences(text):
    parts = []
    for line in re.split(r"[\r\n]+", text):
        parts.extend(x.strip() for x in re.split(r"(?<=[.!?])\s+", line) if x.strip())
    return parts


def context_sentences(context, speaker=None):
    """Lấy các câu trong ngữ cảnh; nếu có speaker thì chỉ lấy câu của người đó."""
    out = []
    for line in context.splitlines():
        m = re.match(r"(Speaker [12]):\s*(.*)", line)
        if m and (speaker is None or m.group(1) == speaker):
            out.extend(split_sentences(m.group(2)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("labeled_csv")
    ap.add_argument("--probes", default=os.path.join("probes", "probes.jsonl"))
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--n-ref", type=int, default=1000, help="số mẫu gốc dùng để ước lượng tỷ lệ bị loại")
    args = ap.parse_args()

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(args.model)

    def max_cos(fact, sents):
        if not sents:
            return 0.0, ""
        vecs = model.encode([fact] + sents, normalize_embeddings=True, convert_to_numpy=True)
        sims = vecs[1:] @ vecs[0]
        i = int(sims.argmax())
        return float(sims[i]), sents[i]

    # 1. các dòng đã chấm
    rows = []
    with open(args.labeled_csv, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            label = (r.get("leak (y/n)") or "").strip().lower()
            if label not in ("y", "n"):
                continue
            fact = r["fact_bi_xoa_hoac_thay"]
            score, best = max_cos(fact, context_sentences(r["ngu_canh_sau_can_thiep"]))
            rows.append((label, score, fact, best))

    print("=" * 90)
    print("1. COSINE LỚN NHẤT GIỮA FACT ĐÃ XÓA VÀ NGỮ CẢNH CÒN LẠI (sắp xếp giảm dần)")
    print("=" * 90)
    for label, score, fact, best in sorted(rows, key=lambda x: -x[1]):
        print(f"[{label}] {score:.2f}  {fact[:45]:45s} <- {best[:60]}")

    # 2. ước lượng tỷ lệ mẫu gốc bị loại với từng ngưỡng
    ref_scores = []
    if os.path.exists(args.probes):
        originals = [json.loads(line) for line in open(args.probes, encoding="utf-8")
                     if '"variant": "original"' in line]
        random.Random(0).shuffle(originals)
        for p in originals[: args.n_ref]:
            sents = [s for s in context_sentences(p["context"]) if s != p["gold_evidence"]]
            ref_scores.append(max_cos(p["gold_evidence"], sents)[0])

    print()
    print("=" * 90)
    print("2. CHỌN NGƯỠNG --sem-thr")
    print("=" * 90)
    ys = [s for l, s, _, _ in rows if l == "y"]
    ns = [s for l, s, _, _ in rows if l == "n"]
    print(f"{'ngưỡng':>7} | {'bắt được rò rỉ (y)':>20} | {'chặn nhầm (n)':>14} | {'% mẫu gốc hiện tại bị loại':>27}")
    for t in [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:
        cy = sum(s >= t for s in ys)
        cn = sum(s >= t for s in ns)
        rem = (f"{100 * sum(s >= t for s in ref_scores) / len(ref_scores):.0f}%" if ref_scores else "n/a")
        print(f"{t:>7.2f} | {cy:>3}/{len(ys):<3} ({100 * cy / max(1, len(ys)):3.0f}%)     | "
              f"{cn:>3}/{len(ns):<3}       | {rem:>27}")
    print("\nGợi ý: chọn ngưỡng bắt được khoảng 80-90% dòng 'y' mà vẫn giữ đủ mẫu gốc"
          " (cần >= 1.500 mẫu gốc đủ điều kiện sau khi lọc).")


if __name__ == "__main__":
    main()
