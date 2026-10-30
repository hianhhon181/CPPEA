"""
Tính PA, EN, EI, SAA (kèm khoảng tin cậy 95% bootstrap) và phân loại lỗi.

Cách chạy:
    python compute_metrics.py predictions/*.jsonl
    python compute_metrics.py predictions/model_a.jsonl predictions/model_b.jsonl --probes probes/probes.jsonl

Định nghĩa (S = các mẫu gốc mà mô hình trả lời đúng ở bản original):
    PA  = độ chính xác trên original
    EN  = độ chính xác trên removal + subst_content, chỉ tính trên S
    EI  = độ chính xác trên distractor_irrelevant + distractor_other, chỉ tính trên S
    SAA = tỷ lệ KHÔNG gán nhầm fact của người kia cho người được hỏi,
          trên subst_speaker (lỗi = chọn fact đã bị chuyển đi)
          và distractor_other (lỗi = chọn fact của người kia), chỉ tính trên S
Đầu ra: bảng in ra màn hình + metrics/<mô-hình>.json + metrics/summary.csv
"""
import argparse
import csv
import glob
import json
import os
import random
from collections import Counter, defaultdict

ERROR_NAMES = {
    ("removal", "gold"): "Không phụ thuộc bằng chứng (vẫn chọn fact đã bị xóa)",
    ("removal", "foil"): "Chọn fact không có trong ngữ cảnh",
    ("subst_content", "old"): "Dùng thông tin cũ (chọn fact đã bị thay)",
    ("subst_content", "none"): "Không nhận ra fact mới",
    ("subst_speaker", "gold"): "Nhầm chủ thể (gán fact của người kia)",
    ("subst_speaker", "foil"): "Chọn fact không có trong ngữ cảnh",
    ("distractor_irrelevant", "foil"): "Bị nhiễu kéo theo",
    ("distractor_irrelevant", "none"): "Bỏ sót bằng chứng khi có nhiễu",
    ("distractor_other", "foil"): "Nhầm chủ thể (chọn fact của người kia)",
    ("distractor_other", "none"): "Bỏ sót bằng chứng khi có nhiễu",
}


def bootstrap_ci(values, n_boot=1000, seed=0):
    if not values:
        return (float("nan"), float("nan"))
    rnd = random.Random(seed)
    n = len(values)
    means = sorted(sum(rnd.choice(values) for _ in range(n)) / n for _ in range(n_boot))
    return (means[int(0.025 * n_boot)], means[int(0.975 * n_boot) - 1])


def summarize(values):
    if not values:
        return {"value": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n": 0}
    lo, hi = bootstrap_ci(values)
    return {"value": sum(values) / len(values), "ci_low": lo, "ci_high": hi, "n": len(values)}


def chosen_kind(p, pred_text):
    """Loại lựa chọn mà mô hình đã chọn, để phân loại lỗi."""
    if pred_text == "No information":
        return "none"
    if p["variant"] == "subst_content":
        return "new" if pred_text == p["gold_evidence"] else "old"
    if pred_text == p["foil"]:
        return "foil"
    return "gold"


def evaluate(probes, preds):
    rows = [p for p in probes if p["id"] in preds]
    for p in rows:
        p["pred"] = preds[p["id"]]
        p["pred_text"] = p["options"][p["pred"]]
        p["correct"] = int(p["pred"] == p["label"])

    original_ok = {p["base_id"]: p["correct"] for p in rows if p["variant"] == "original"}
    S = {b for b, ok in original_ok.items() if ok}

    def on_S(variants, fn=lambda p: p["correct"]):
        return [fn(p) for p in rows if p["variant"] in variants and p["base_id"] in S]

    def no_misattribution(p):
        if p["variant"] == "subst_speaker":
            return int(p["pred_text"] != p["gold_evidence"])
        return int(p["pred_text"] != p["foil"])  # distractor_other

    metrics = {
        "PA": summarize([p["correct"] for p in rows if p["variant"] == "original"]),
        "EN": summarize(on_S({"removal", "subst_content"})),
        "EI": summarize(on_S({"distractor_irrelevant", "distractor_other"})),
        "SAA": summarize(on_S({"subst_speaker", "distractor_other"}, no_misattribution)),
    }

    per_variant = {}
    for v in ["original", "removal", "subst_content", "subst_speaker", "distractor_irrelevant", "distractor_other"]:
        per_variant[v] = {
            "acc_all": summarize([p["correct"] for p in rows if p["variant"] == v])["value"],
            "acc_on_S": summarize(on_S({v}))["value"],
        }

    by_sessions = defaultdict(dict)
    for k in sorted({p["n_prev_sessions"] for p in rows}):
        sub = [p for p in rows if p["n_prev_sessions"] == k]
        sub_S = {p["base_id"] for p in sub if p["variant"] == "original" and p["correct"]}
        pick = lambda vs: [p["correct"] for p in sub if p["variant"] in vs and p["base_id"] in sub_S]
        by_sessions[k] = {
            "PA": summarize([p["correct"] for p in sub if p["variant"] == "original"])["value"],
            "EN": summarize(pick({"removal", "subst_content"}))["value"],
            "EI": summarize(pick({"distractor_irrelevant", "distractor_other"}))["value"],
            "SAA": summarize([no_misattribution(p) for p in sub
                              if p["variant"] in {"subst_speaker", "distractor_other"} and p["base_id"] in sub_S])["value"],
            "removal": summarize(pick({"removal"}))["value"],
            "subst_speaker": summarize(pick({"subst_speaker"}))["value"],
        }
        by_sessions[k]["gap"] = by_sessions[k]["removal"] - by_sessions[k]["subst_speaker"]

    # Phân bố lựa chọn trên S: mô hình chọn loại đáp án nào ở từng biến thể
    choice_dist = {}
    for v in ["original", "removal", "subst_content", "subst_speaker", "distractor_irrelevant", "distractor_other"]:
        kinds = Counter(chosen_kind(p, p["pred_text"]) for p in rows if p["variant"] == v and p["base_id"] in S)
        total = sum(kinds.values()) or 1
        choice_dist[v] = {k: round(c / total, 4) for k, c in kinds.items()}

    # Khoảng cách quy thuộc: acc(removal) - acc(subst_speaker), ghép cặp theo mẫu gốc, bootstrap 95%
    by_base = defaultdict(dict)
    for p in rows:
        if p["base_id"] in S and p["variant"] in ("removal", "subst_speaker"):
            by_base[p["base_id"]][p["variant"]] = p["correct"]
    diffs = [d["removal"] - d["subst_speaker"] for d in by_base.values() if len(d) == 2]
    gap = summarize(diffs)
    flip = Counter((d["removal"], d["subst_speaker"]) for d in by_base.values() if len(d) == 2)

    # Thiên lệch vị trí: tỷ lệ chọn A/B/C so với tỷ lệ đáp án đúng
    position = {
        "pred": {"ABC"[i]: round(c / len(rows), 4) for i, c in sorted(Counter(p["pred"] for p in rows).items())},
        "label": {"ABC"[i]: round(c / len(rows), 4) for i, c in sorted(Counter(p["label"] for p in rows).items())},
    }

    errors = Counter()
    for p in rows:
        if p["variant"] != "original" and p["base_id"] in S and not p["correct"]:
            errors[ERROR_NAMES.get((p["variant"], chosen_kind(p, p["pred_text"])), f"{p['variant']}: khác")] += 1

    return {"n_probes": len(rows), "n_S": len(S), "metrics": metrics, "per_variant": per_variant,
            "attribution_gap": gap,
            "gap_contingency": {"removal_ok&speaker_ok": flip[(1, 1)], "removal_ok&speaker_wrong": flip[(1, 0)],
                                "removal_wrong&speaker_ok": flip[(0, 1)], "both_wrong": flip[(0, 0)]},
            "choice_distribution_on_S": choice_dist, "position_bias": position,
            "by_prev_sessions": by_sessions, "errors": dict(errors.most_common())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("predictions", nargs="+")
    ap.add_argument("--probes", default=os.path.join("probes", "probes.jsonl"))
    ap.add_argument("--out-dir", default="metrics")
    args = ap.parse_args()

    # PowerShell không tự mở rộng dấu *, nên script tự làm
    paths = []
    for pattern in args.predictions:
        matches = sorted(glob.glob(pattern))
        if not matches and not os.path.exists(pattern):
            print(f"Không tìm thấy file nào khớp với: {pattern}")
        paths.extend(matches or ([pattern] if os.path.exists(pattern) else []))
    if not paths:
        raise SystemExit("Không có file kết quả nào để tính. Kiểm tra lại thư mục predictions.")
    print("Các file sẽ tính:", ", ".join(paths))

    probes = [json.loads(line) for line in open(args.probes, encoding="utf-8") if line.strip()]
    probe_ids = {p["id"] for p in probes}
    os.makedirs(args.out_dir, exist_ok=True)
    summary = []
    for path in paths:
        preds = {}
        for line in open(path, encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                preds[r["id"]] = r["pred"]
        name = os.path.splitext(os.path.basename(path))[0]
        matched = len(probe_ids & preds.keys())
        if matched < len(probe_ids):
            print(f"\n[Cảnh báo] {name}: chỉ có {matched}/{len(probe_ids)} probe khớp với {args.probes}."
                  " Kết quả có thể chưa chạy xong hoặc lấy từ tập probe khác.")
        res = evaluate([dict(p) for p in probes], preds)
        with open(os.path.join(args.out_dir, name + ".json"), "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=2)

        m = res["metrics"]
        print(f"\n===== {name}  (probe: {res['n_probes']}, |S| = {res['n_S']})")
        for k in ["PA", "EN", "EI", "SAA"]:
            print(f"  {k:4s} = {m[k]['value']:.3f}  [95% CI {m[k]['ci_low']:.3f}–{m[k]['ci_high']:.3f}]  n={m[k]['n']}")
        print("  Độ chính xác từng biến thể (toàn bộ / trên S):")
        for v, d in res["per_variant"].items():
            print(f"    {v:22s} {d['acc_all']:.3f} / {d['acc_on_S']:.3f}")
        g = res["attribution_gap"]
        print(f"  Khoảng cách quy thuộc (removal - subst_speaker, ghép cặp) = {g['value']:.3f}"
              f"  [95% CI {g['ci_low']:.3f}–{g['ci_high']:.3f}]  n={g['n']}")
        c = res["gap_contingency"]
        print(f"    removal đúng & subst_speaker sai: {c['removal_ok&speaker_wrong']},"
              f"  removal sai & subst_speaker đúng: {c['removal_wrong&speaker_ok']}")
        print("  Mô hình chọn gì ở từng biến thể (trên S; gold=fact gốc, foil=fact gây nhiễu, none=No information, new/old=subst_content):")
        for v, d in res["choice_distribution_on_S"].items():
            print(f"    {v:22s} " + "  ".join(f"{k}={x:.3f}" for k, x in sorted(d.items())))
        pb = res["position_bias"]
        print("  Vị trí được chọn / vị trí đáp án đúng: "
              + "  ".join(f"{k}: {pb['pred'].get(k, 0):.3f}/{pb['label'].get(k, 0):.3f}" for k in "ABC"))
        print("  Theo số phiên trước (PA / EN / EI / SAA | removal / subst_speaker / gap):")
        for k, d in res["by_prev_sessions"].items():
            print(f"    {k} phiên: {d['PA']:.3f} / {d['EN']:.3f} / {d['EI']:.3f} / {d['SAA']:.3f}"
                  f" | {d['removal']:.3f} / {d['subst_speaker']:.3f} / {d['gap']:.3f}")
        print("  Các lỗi thường gặp (trên S):")
        for e, c in list(res["errors"].items())[:8]:
            print(f"    {c:5d}  {e}")
        pv = res["per_variant"]
        summary.append([name] + [round(m[k]["value"], 4) for k in ["PA", "EN", "EI", "SAA"]]
                       + [round(pv["removal"]["acc_on_S"], 4), round(pv["subst_speaker"]["acc_on_S"], 4),
                          round(g["value"], 4), round(g["ci_low"], 4), round(g["ci_high"], 4), res["n_S"]])

    with open(os.path.join(args.out_dir, "summary.csv"), "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "PA", "EN", "EI", "SAA", "removal_S", "subst_speaker_S",
                    "gap", "gap_ci_low", "gap_ci_high", "n_S"])
        w.writerows(summary)
    print(f"\nĐã ghi {args.out_dir}/summary.csv và chi tiết từng mô hình.")


if __name__ == "__main__":
    main()
