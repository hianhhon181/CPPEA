"""
Sinh tập probe phản thực từ MSC (bản ParlAI, thư mục msc_dialogue).

Mỗi probe là một câu hỏi trắc nghiệm 3 lựa chọn về persona của một người nói:
    A/B: hai câu persona (một đúng, một gây nhiễu), C: "No information".
Ngữ cảnh là BỘ NHỚ persona theo từng phiên trước (previous_dialogs[i].personas),
có nhãn người nói, nên mọi phép can thiệp đều thao tác trực tiếp trên bộ nhớ.

Sáu biến thể cho mỗi mẫu gốc:
    original               - giữ nguyên                 -> đáp án: fact đúng
    removal                - xóa fact đúng (mọi bản sao) -> đáp án: No information
    subst_content          - thay fact đúng bằng fact mới -> đáp án: fact mới
    subst_speaker          - chuyển fact đúng sang người kia -> đáp án: No information
    distractor_irrelevant  - thêm fact ngẫu nhiên không liên quan -> đáp án: fact đúng
    distractor_other       - thêm fact gây nhiễu vào bộ nhớ người kia -> đáp án: fact đúng

Cách chạy (trong thư mục CPPEA):
    python build_probes.py
    python build_probes.py --max-base 1500     # lấy mẫu con 1500 mẫu gốc x 6 biến thể
    python build_probes.py --max-base 1500 --shuffle-all --out probes_shuffled   # bản kiểm tra vị trí

Đầu ra (thư mục probes/):
    probes.jsonl         - toàn bộ probe, mỗi dòng một JSON
    manual_check.csv     - 60 probe ngẫu nhiên để đọc tay
    stats.txt            - thống kê tập dữ liệu
"""
import argparse
import hashlib
import csv
import json
import os
import random
import re
from collections import Counter

SPEAKERS = ["Speaker 1", "Speaker 2"]
NONE_OPTION = "No information"
STOP = set("""a an the i i'm im my me am is are was were be been to of in on at for and or but
with have has had do does did not no so very really just like love enjoy also it its that this
there they them their you your we our he she his her as by from up out about than then""".split())

SHUFFLE_ALL = False  # True: xáo cả vị trí của "No information" (kiểm tra thiên lệch vị trí)

VARIANTS = ["original", "removal", "subst_content", "subst_speaker",
            "distractor_irrelevant", "distractor_other"]


# ---------------------------------------------------------------- tiện ích văn bản
def norm(text):
    return re.sub(r"\s+", " ", text.strip().lower())


def stem(word):
    """Rút gọn hậu tố đơn giản để work/working/worked được coi là một."""
    for suf in ("ings", "ing", "ers", "er", "ies", "ied", "ed", "es", "s"):
        if suf == "s" and word.endswith("ss"):  # class, glass: không bỏ s
            continue
        if word.endswith(suf) and len(word) - len(suf) >= 3:
            return word[: -len(suf)]
    return word


def content_words(text):
    return {stem(w) for w in re.findall(r"[a-z']+", text.lower()) if w not in STOP and len(w) > 2}


def clean(text):
    return re.sub(r"\s+", " ", text).strip()


def is_option_like(text):
    """Chỉ dùng làm lựa chọn: một câu đơn, 4-20 từ, không xuống dòng."""
    sentences = [x for x in re.split(r"(?<=[.!?])\s+", text.strip()) if x]
    return len(sentences) == 1 and 4 <= len(text.split()) <= 20 and len(content_words(text)) >= 2


def jaccard(a, b):
    wa, wb = content_words(a), content_words(b)
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


def similar(a, b, thr):
    return norm(a) == norm(b) or jaccard(a, b) >= thr


def split_sentences(text):
    """Tách một mục persona (có thể gồm nhiều câu, có xuống dòng) thành từng câu."""
    parts = []
    for line in re.split(r"[\r\n]+", text):
        parts.extend(x.strip() for x in re.split(r"(?<=[.!?])\s+", line) if x.strip())
    return parts


# ---------------------------------------------------------------- lọc theo ngữ nghĩa (tùy chọn)
SEM = None  # dict câu -> vector đã chuẩn hóa, khi bật --semantic
SEM_THR = 0.55
SEM_DUP = 0.9


def build_semantic_index(sentences, model_name, batch_size=256):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    sentences = sorted(set(sentences))
    print(f"Đang tính embedding cho {len(sentences)} câu bằng {model_name} ...")
    vecs = model.encode(sentences, batch_size=batch_size, normalize_embeddings=True,
                        show_progress_bar=True, convert_to_numpy=True)
    return dict(zip(sentences, vecs))


def cos(a, b):
    if SEM is None or a not in SEM or b not in SEM:
        return 0.0
    return float(SEM[a] @ SEM[b])


def is_dup(a, b, args):
    """Hai câu là bản sao của cùng một fact."""
    return similar(a, b, args.dup) or (SEM is not None and cos(a, b) >= SEM_DUP)


def is_related(a, b, args):
    """Hai câu liên quan đủ để câu này gợi ý câu kia."""
    return norm(a) == norm(b) or jaccard(a, b) >= args.related or (SEM is not None and cos(a, b) >= SEM_THR)


# ---------------------------------------------------------------- đọc dữ liệu
def load_episodes(data_dir, split):
    """Đọc mọi session_k/<split>.txt; trả về list episode có bộ nhớ theo phiên."""
    episodes = []
    for sess in sorted(os.listdir(data_dir)):
        path = os.path.join(data_dir, sess, f"{split}.txt")
        if not os.path.isfile(path):
            continue
        with open(path, encoding="utf-8") as f:
            for line_no, line in enumerate(f):
                if not line.strip():
                    continue
                rec = json.loads(line)
                memory = []  # list các entry {session, speaker, fact}
                for s_idx, prev in enumerate(rec.get("previous_dialogs", []), start=1):
                    for spk_idx, facts in enumerate(prev.get("personas", [])[:2]):
                        for fact in facts:
                            for sent in split_sentences(fact):  # mỗi câu là một mục bộ nhớ
                                sent = clean(sent)
                                memory.append({"session": s_idx, "speaker": SPEAKERS[spk_idx],
                                               "fact": sent, "option_ok": is_option_like(sent)})
                if not memory:
                    continue
                meta = rec.get("metadata", {})
                episodes.append({
                    "episode_id": f"{sess}/{split}/{line_no}",
                    "initial_data_id": meta.get("initial_data_id", ""),
                    "n_prev_sessions": len(rec.get("previous_dialogs", [])),
                    "memory": memory,
                })
    return episodes


# ---------------------------------------------------------------- dựng probe
def render_context(memory):
    """Chuyển bộ nhớ thành văn bản, gom theo phiên và người nói."""
    lines = []
    for s in sorted({m["session"] for m in memory}):
        lines.append(f"[Session {s}]")
        for spk in SPEAKERS:
            facts = [m["fact"] for m in memory if m["session"] == s and m["speaker"] == spk]
            if facts:
                lines.append(f"{spk}: " + " ".join(facts))
    return "\n".join(lines)


def render_prompt(context_text, speaker, options):
    letters = "ABC"
    opts = "\n".join(f"({letters[i]}) {o}" for i, o in enumerate(options))
    return (
        "Below is the memory of a multi-session conversation between Speaker 1 and Speaker 2.\n\n"
        f"{context_text}\n\n"
        f"Question: According to the memory, which statement is true about {speaker}?\n"
        f"{opts}\n"
        "Answer with A, B or C."
    )


def short_hash(text):
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:8]


def make_probe(ep, variant, speaker, memory, gold, foil, answer_text, extra=None, key_fact=None):
    # A/B xáo trộn ổn định theo id, C luôn là No information
    pair = [gold, foil]
    rnd = random.Random(f"{ep['episode_id']}|{gold}")
    rnd.shuffle(pair)
    options = pair + [NONE_OPTION]
    if SHUFFLE_ALL:
        random.Random(f"{ep['episode_id']}|{gold}|all").shuffle(options)
    context_text = render_context(memory)
    base_id = f"{ep['episode_id']}|{SPEAKERS.index(speaker)}|{short_hash(key_fact or gold)}"
    probe = {
        "id": f"{base_id}|{variant}",
        "base_id": base_id,
        "variant": variant,
        "episode_id": ep["episode_id"],
        "n_prev_sessions": ep["n_prev_sessions"],
        "target_speaker": speaker,
        "gold_evidence": gold,
        "owner": speaker,
        "foil": foil,
        "options": options,
        "label": options.index(answer_text),
        "label_text": answer_text,
        "context": context_text,
        "prompt": render_prompt(context_text, speaker, options),
    }
    if extra:
        probe.update(extra)
    return probe


def build_for_episode(ep, all_facts, args, rnd):
    probes = []
    memory = ep["memory"]
    for speaker in SPEAKERS:
        other = SPEAKERS[1 - SPEAKERS.index(speaker)]
        own = [m["fact"] for m in memory if m["speaker"] == speaker]
        others = [m["fact"] for m in memory if m["speaker"] == other]

        # fact đúng phải "cô lập": không có fact liên quan nào khác trong episode,
        # của cả hai người nói (trừ các bản sao gần như y hệt của chính nó, sẽ bị xóa cùng).
        own_ok = [m["fact"] for m in memory if m["speaker"] == speaker and m["option_ok"]]

        def isolated(f):
            for m in memory:
                if is_dup(m["fact"], f, args):     # bản sao của chính nó
                    if m["speaker"] != speaker:
                        return False                # người kia cũng có fact này
                    continue
                if is_related(m["fact"], f, args):  # câu liên quan còn sót lại -> rò rỉ
                    return False
            return True

        candidates = list(dict.fromkeys(f for f in own_ok if isolated(f)))
        rnd.shuffle(candidates)
        for gold in candidates[: args.per_episode]:
            ep_facts = own + others

            def pick_outside(exclude):
                """Lấy một fact từ episode khác, không giống bất kỳ fact nào trong episode."""
                for _ in range(200):
                    cand = rnd.choice(all_facts)
                    if not any(is_related(cand, e, args) for e in exclude):
                        return cand
                return None

            foil = pick_outside(ep_facts)
            new_fact = pick_outside(ep_facts + [foil] if foil else ep_facts)
            if foil is None or new_fact is None:
                continue

            def is_gold(m):
                return m["speaker"] == speaker and is_dup(m["fact"], gold, args)

            gold_entries = [m for m in memory if is_gold(m)]
            last_session = max(m["session"] for m in memory)

            # 1. original
            probes.append(make_probe(ep, "original", speaker, memory, gold, foil, gold))

            # 2. removal: xóa mọi bản sao của fact đúng
            mem_rm = [m for m in memory if not is_gold(m)]
            probes.append(make_probe(ep, "removal", speaker, mem_rm, gold, foil, NONE_OPTION))

            # 3. subst_content: thay fact đúng bằng fact mới; lựa chọn là fact cũ và fact mới
            mem_sc, replaced = [], False
            for m in memory:
                if is_gold(m):
                    if not replaced:  # thay bản sao đầu tiên, bỏ các bản sao còn lại
                        mem_sc.append(dict(m, fact=new_fact))
                        replaced = True
                else:
                    mem_sc.append(m)
            p = make_probe(ep, "subst_content", speaker, mem_sc, new_fact, gold, new_fact,
                           extra={"gold_evidence": new_fact, "replaced_fact": gold}, key_fact=gold)
            probes.append(p)

            # 4. subst_speaker: chuyển fact đúng sang người kia, giữ nguyên nội dung
            mem_ss = [dict(m, speaker=other) if is_gold(m) else m for m in memory]
            probes.append(make_probe(ep, "subst_speaker", speaker, mem_ss, gold, foil, NONE_OPTION,
                                     extra={"owner": other}))

            # 5. distractor_irrelevant: thêm k fact ngẫu nhiên cho cả hai người
            mem_di = list(memory)
            added = []
            for _ in range(args.n_distractors):
                d = pick_outside(ep_facts + [foil, new_fact] + added)
                if d:
                    added.append(d)
                    mem_di.append({"session": rnd.randint(1, last_session),
                                   "speaker": rnd.choice(SPEAKERS), "fact": d})
            probes.append(make_probe(ep, "distractor_irrelevant", speaker, mem_di, gold, foil, gold,
                                     extra={"added_facts": added}))

            # 6. distractor_other: đưa fact gây nhiễu (chính là lựa chọn sai) vào bộ nhớ người kia
            mem_do = list(memory) + [{"session": gold_entries[-1]["session"], "speaker": other, "fact": foil}]
            probes.append(make_probe(ep, "distractor_other", speaker, mem_do, gold, foil, gold,
                                     extra={"added_facts": [foil]}))
    return probes


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join("data", "msc", "msc_dialogue"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", default="probes")
    ap.add_argument("--per-episode", type=int, default=2, help="số fact đúng lấy cho mỗi người nói")
    ap.add_argument("--n-distractors", type=int, default=3)
    ap.add_argument("--dup", type=float, default=0.6, help="ngưỡng coi là bản sao của cùng một fact")
    ap.add_argument("--related", type=float, default=0.2, help="ngưỡng coi hai fact là có liên quan")
    ap.add_argument("--shuffle-all", action="store_true",
                    help="xáo cả vị trí của 'No information' thay vì luôn đặt ở C")
    ap.add_argument("--max-base", type=int, default=0,
                    help="lấy tối đa N mẫu gốc, chia đều theo số phiên trước (0 = lấy hết)")
    ap.add_argument("--semantic", action="store_true",
                    help="bật lọc theo ngữ nghĩa bằng sentence-transformers (cần: pip install sentence-transformers)")
    ap.add_argument("--sem-model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--sem-thr", type=float, default=0.55, help="cosine >= ngưỡng này coi là liên quan")
    ap.add_argument("--sem-dup", type=float, default=0.9, help="cosine >= ngưỡng này coi là bản sao")
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    global SHUFFLE_ALL
    SHUFFLE_ALL = args.shuffle_all
    rnd = random.Random(args.seed)
    episodes = load_episodes(args.data, args.split)
    if not episodes:
        raise SystemExit(f"Không đọc được episode nào trong {args.data} (split={args.split}).")

    global SEM, SEM_THR, SEM_DUP
    if args.semantic:
        SEM_THR, SEM_DUP = args.sem_thr, args.sem_dup
        SEM = build_semantic_index([m["fact"] for ep in episodes for m in ep["memory"]], args.sem_model)

    all_facts = sorted({m["fact"] for ep in episodes for m in ep["memory"] if m["option_ok"]})
    probes = []
    for ep in episodes:
        probes.extend(build_for_episode(ep, all_facts, args, rnd))

    n_eligible = sum(p["variant"] == "original" for p in probes)
    eligible_by_sess = Counter(p["n_prev_sessions"] for p in probes if p["variant"] == "original")
    if args.max_base:
        # lấy mẫu con theo mẫu gốc, chia đều giữa các mức số phiên trước
        groups = {}
        for p in probes:
            if p["variant"] == "original":
                groups.setdefault(p["n_prev_sessions"], []).append(p["base_id"])
        quota = args.max_base // len(groups)
        keep = set()
        for k in sorted(groups):
            ids = sorted(groups[k])
            keep.update(rnd.sample(ids, min(quota, len(ids))))
        probes = [p for p in probes if p["base_id"] in keep]

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "probes.jsonl"), "w", encoding="utf-8") as f:
        for p in probes:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    # mẫu để kiểm tra tay: 10 mẫu gốc ngẫu nhiên x 6 biến thể
    base_ids = sorted({p["base_id"] for p in probes})
    chosen = set(rnd.sample(base_ids, min(10, len(base_ids))))
    with open(os.path.join(args.out, "manual_check.csv"), "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["base_id", "variant", "target_speaker", "options", "label_text", "context", "valid (y/n)", "note"])
        for p in probes:
            if p["base_id"] in chosen:
                w.writerow([p["base_id"], p["variant"], p["target_speaker"], " | ".join(p["options"]),
                            p["label_text"], p["context"], "", ""])

    # thống kê
    var_counts = Counter(p["variant"] for p in probes)
    sess_counts = Counter(p["n_prev_sessions"] for p in probes if p["variant"] == "original")
    ctx_words = [len(p["context"].split()) for p in probes if p["variant"] == "original"]
    label_pos = Counter(p["label"] for p in probes)
    lines = [
        f"Dữ liệu: {args.data}  split={args.split}  seed={args.seed}",
        f"Số episode đọc được: {len(episodes)}",
        f"Lọc ngữ nghĩa: {'bật, ' + args.sem_model + f', sem_thr={args.sem_thr}, sem_dup={args.sem_dup}' if args.semantic else 'tắt'}",
        f"Số mẫu gốc đủ điều kiện (trước khi lấy mẫu con): {n_eligible} "
        + str(dict(sorted(eligible_by_sess.items()))),
        f"Số mẫu gốc: {var_counts['original']}",
        f"Tổng số probe: {len(probes)}",
        "Số probe theo biến thể: " + ", ".join(f"{v}={var_counts[v]}" for v in VARIANTS),
        "Số mẫu gốc theo số phiên trước: " + ", ".join(f"{k} phiên={v}" for k, v in sorted(sess_counts.items())),
        f"Độ dài ngữ cảnh (từ): trung bình {sum(ctx_words)/len(ctx_words):.0f}, "
        f"min {min(ctx_words)}, max {max(ctx_words)}",
        "Vị trí đáp án đúng (0=A,1=B,2=C): " + str(dict(sorted(label_pos.items()))),
    ]
    with open(os.path.join(args.out, "stats.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))
    print(f"\nĐã ghi: {args.out}/probes.jsonl, {args.out}/manual_check.csv, {args.out}/stats.txt")


if __name__ == "__main__":
    main()