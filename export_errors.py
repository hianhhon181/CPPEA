"""
Xuất các lỗi của một mô hình ra CSV để đọc tay.

Mặc định: lỗi ở biến thể removal mà mô hình vẫn chọn fact đã bị xóa,
chỉ tính trên các mẫu mô hình trả lời đúng ở bản original (tập S).

Cách chạy:
    python export_errors.py predictions\\Qwen__Qwen2.5-7B-Instruct.jsonl
    python export_errors.py predictions\\Qwen__Qwen2.5-7B-Instruct.jsonl --variant subst_content --n 30

Đầu ra: errors_<mô-hình>_<biến-thể>.csv (mở bằng Excel), gồm ngữ cảnh SAU can thiệp,
fact đã bị xóa/thay, lựa chọn của mô hình, và 2 cột để điền:
    leak (y/n)  : ngữ cảnh còn lại có cho phép suy ra fact đó không?
    note        : ghi câu nào trong ngữ cảnh gợi ý fact đó
"""
import argparse
import csv
import json
import os
import random


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("predictions")
    ap.add_argument("--probes", default=os.path.join("probes", "probes.jsonl"))
    ap.add_argument("--variant", default="removal")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    preds = {}
    for line in open(args.predictions, encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            preds[r["id"]] = r
    probes = [json.loads(line) for line in open(args.probes, encoding="utf-8") if line.strip()]

    ok_original = {p["base_id"] for p in probes
                   if p["variant"] == "original" and p["id"] in preds and preds[p["id"]]["pred"] == p["label"]}

    errors = []
    for p in probes:
        if p["variant"] != args.variant or p["id"] not in preds or p["base_id"] not in ok_original:
            continue
        r = preds[p["id"]]
        if r["pred"] == p["label"]:
            continue
        chosen = p["options"][r["pred"]]
        # removal: chỉ lấy trường hợp chọn lại fact đã bị xóa; subst_content: chọn fact cũ
        if args.variant == "removal" and chosen != p["gold_evidence"]:
            continue
        if args.variant == "subst_content" and chosen != p.get("replaced_fact"):
            continue
        errors.append((p, chosen, r.get("probs")))

    print(f"Số lỗi phù hợp: {len(errors)}")
    random.Random(args.seed).shuffle(errors)
    name = os.path.splitext(os.path.basename(args.predictions))[0]
    out = f"errors_{name}_{args.variant}.csv"
    with open(out, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "target_speaker", "fact_bi_xoa_hoac_thay", "mo_hinh_chon", "xac_suat_ABC",
                    "ngu_canh_sau_can_thiep", "leak (y/n)", "note"])
        for p, chosen, probs in errors[: args.n]:
            removed = p.get("replaced_fact", p["gold_evidence"])
            w.writerow([p["id"], p["target_speaker"], removed, chosen, probs, p["context"], "", ""])
    print(f"Đã ghi {min(args.n, len(errors))} lỗi vào {out}")


if __name__ == "__main__":
    main()
