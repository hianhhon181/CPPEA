"""
So sánh kết quả giữa các định dạng bộ nhớ (line_prefix / sentence_prefix / third_person).

Với mỗi mô hình và mỗi định dạng, tính trên tập S (các mẫu gốc trả lời đúng ở original):
    PA, |S|, acc(removal), acc(subst_speaker), tỷ lệ gán nhầm ở subst_speaker,
    khoảng cách quy thuộc acc(removal) - acc(subst_speaker) kèm khoảng tin cậy 95% (bootstrap).
Ngoài ra so sánh ghép cặp giữa định dạng gốc và từng định dạng mới trên các mẫu gốc
thuộc S của CẢ HAI định dạng: chênh lệch acc(subst_speaker) và acc(removal), kèm khoảng tin cậy 95%.

Cách chạy (trong thư mục dự án CoSA):
    python compare_formats.py --models Qwen__Qwen2.5-1.5B-Instruct Qwen__Qwen2.5-3B-Instruct

Đường dẫn mặc định:
    line_prefix:     probes/probes.jsonl                      + predictions/<model>.jsonl  (dùng lại lần chạy chính)
    <định dạng khác>: probes_fmt_<định dạng>/probes.jsonl     + predictions_fmt_<định dạng>/<model>.jsonl
Đầu ra: bảng in ra màn hình + metrics_formats/format_comparison.csv
"""
import argparse
import csv
import json
import os
import random

FORMATS = ["line_prefix", "sentence_prefix", "third_person"]


ARGS = None


def paths(fmt, model):
    if fmt == "line_prefix" and not os.path.exists(os.path.join("predictions_fmt_line_prefix", model + ".jsonl")):
        return ARGS.base_probes, os.path.join(ARGS.base_pred_dir, model + ".jsonl")
    return (os.path.join(f"probes_fmt_{fmt}", "probes.jsonl"),
            os.path.join(f"predictions_fmt_{fmt}", model + ".jsonl"))


def load(probe_path, pred_path):
    preds = {}
    with open(pred_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                preds[r["id"]] = r["pred"]
    rows = {}
    with open(probe_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            p = json.loads(line)
            if p["variant"] not in ("original", "removal", "subst_speaker") or p["id"] not in preds:
                continue
            pred = preds[p["id"]]
            rows[(p["base_id"], p["variant"])] = {
                "correct": int(pred == p["label"]),
                "misattr": int(p["variant"] == "subst_speaker" and p["options"][pred] == p["gold_evidence"]),
            }
    return rows


def ci(values, n_boot=1000, seed=0):
    if not values:
        return float("nan"), float("nan")
    rnd = random.Random(seed)
    n = len(values)
    means = sorted(sum(rnd.choice(values) for _ in range(n)) / n for _ in range(n_boot))
    return means[int(0.025 * n_boot)], means[int(0.975 * n_boot) - 1]


def mean(v):
    return sum(v) / len(v) if v else float("nan")


def stats(rows):
    bases = {b for (b, v) in rows if v == "original"}
    S = {b for b in bases if rows[(b, "original")]["correct"]}
    full = [b for b in S if (b, "removal") in rows and (b, "subst_speaker") in rows]
    rem = [rows[(b, "removal")]["correct"] for b in full]
    ss = [rows[(b, "subst_speaker")]["correct"] for b in full]
    mis = [rows[(b, "subst_speaker")]["misattr"] for b in full]
    gap = [a - b for a, b in zip(rem, ss)]
    return {"PA": mean([rows[(b, "original")]["correct"] for b in bases]), "n_base": len(bases),
            "S": set(full), "n_S": len(full), "removal": mean(rem), "subst_speaker": mean(ss),
            "misattribution": mean(mis), "gap": mean(gap), "gap_ci": ci(gap)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True, help="tên file dự đoán, ví dụ Qwen__Qwen2.5-1.5B-Instruct")
    ap.add_argument("--formats", nargs="+", default=FORMATS)
    ap.add_argument("--out-dir", default="metrics_formats")
    ap.add_argument("--base-probes", default=os.path.join("probes", "probes.jsonl"),
                    help="tập probe gốc (định dạng line_prefix), phải là tập đã dùng để sinh các định dạng mới")
    ap.add_argument("--base-pred-dir", default="predictions", help="thư mục dự đoán của lần chạy chính trên tập gốc")
    args = ap.parse_args()
    global ARGS
    ARGS = args
    os.makedirs(args.out_dir, exist_ok=True)

    table, paired = [], []
    for model in args.models:
        loaded = {}
        for fmt in args.formats:
            pp, pr = paths(fmt, model)
            if not (os.path.exists(pp) and os.path.exists(pr)):
                print(f"[bỏ qua] thiếu {pp} hoặc {pr}")
                continue
            loaded[fmt] = load(pp, pr)
            n_orig = sum(1 for (b, v) in loaded[fmt] if v == "original")
            print(f"[{model} | {fmt}] {pp} + {pr}: {len(loaded[fmt])} probe khớp id ({n_orig} mẫu gốc)")
            st = stats(loaded[fmt])
            table.append({"model": model, "format": fmt, "PA": st["PA"], "n_base": st["n_base"], "n_S": st["n_S"],
                          "acc_removal_S": st["removal"], "acc_subst_speaker_S": st["subst_speaker"],
                          "misattribution_S": st["misattribution"], "gap": st["gap"],
                          "gap_ci_low": st["gap_ci"][0], "gap_ci_high": st["gap_ci"][1]})
            loaded[fmt + "_S"] = st["S"]
        base = "line_prefix"
        for fmt in args.formats:
            if fmt == base or fmt not in loaded or base not in loaded:
                continue
            common = sorted(loaded[base + "_S"] & loaded[fmt + "_S"])
            for var in ("subst_speaker", "removal"):
                d = [loaded[fmt][(b, var)]["correct"] - loaded[base][(b, var)]["correct"] for b in common]
                lo, hi = ci(d)
                paired.append({"model": model, "compare": f"{fmt} - {base}", "variant": var, "n_common_S": len(common),
                               "acc_base": mean([loaded[base][(b, var)]["correct"] for b in common]),
                               "acc_new": mean([loaded[fmt][(b, var)]["correct"] for b in common]),
                               "diff": mean(d), "ci_low": lo, "ci_high": hi})

    print(f"\n{'model':32s} {'format':16s} {'PA':>6s} {'|S|':>5s} {'rem':>6s} {'ss':>6s} {'misattr':>8s} {'gap [95% CI]':>24s}")
    for r in table:
        print(f"{r['model']:32s} {r['format']:16s} {r['PA']:6.3f} {r['n_S']:5d} {r['acc_removal_S']:6.3f} "
              f"{r['acc_subst_speaker_S']:6.3f} {r['misattribution_S']:8.3f} "
              f"{r['gap']:7.3f} [{r['gap_ci_low']:.3f}, {r['gap_ci_high']:.3f}]")
    if paired:
        print("\nSo sánh ghép cặp (trên giao của hai tập S):")
        for r in paired:
            print(f"{r['model']:32s} {r['compare']:30s} {r['variant']:14s} n={r['n_common_S']:4d} "
                  f"{r['acc_base']:.3f} -> {r['acc_new']:.3f}  diff {r['diff']:+.3f} [{r['ci_low']:+.3f}, {r['ci_high']:+.3f}]")

    with open(os.path.join(args.out_dir, "format_comparison.csv"), "w", encoding="utf-8-sig", newline="") as f:
        if table:
            w = csv.DictWriter(f, fieldnames=list(table[0].keys()))
            w.writeheader()
            w.writerows(table)
    with open(os.path.join(args.out_dir, "format_paired.csv"), "w", encoding="utf-8-sig", newline="") as f:
        if paired:
            w = csv.DictWriter(f, fieldnames=list(paired[0].keys()))
            w.writeheader()
            w.writerows(paired)
    print(f"\nĐã ghi {args.out_dir}/format_comparison.csv và {args.out_dir}/format_paired.csv")


if __name__ == "__main__":
    main()
