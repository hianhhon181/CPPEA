"""
Thực nghiệm định dạng bộ nhớ: viết lại ngữ cảnh của các probe theo cách ghi người sở hữu khác nhau.

Ba định dạng:
    line_prefix      (gốc)  Speaker 1: I'm an athlete. I was brought up in Alaska.
    sentence_prefix         Speaker 1: I'm an athlete. Speaker 1: I was brought up in Alaska.
    third_person            Speaker 1 is an athlete. Speaker 1 was brought up in Alaska.

Chỉ phần BỘ NHỚ thay đổi; câu hỏi và các lựa chọn giữ nguyên (mặc định),
nên mọi khác biệt về kết quả đến từ cách biểu diễn người sở hữu trong bộ nhớ.
Tùy chọn --options third viết lại cả các lựa chọn ở ngôi thứ ba theo người được hỏi.

Cách chạy (trong thư mục dự án CoSA):
    python reformat_memory.py --format third_person
    python reformat_memory.py --format sentence_prefix
    python reformat_memory.py --format line_prefix        # bản đối chứng, cùng tập con

Mặc định chỉ giữ 3 biến thể original, removal, subst_speaker (original cần để xác định tập S).
Đầu ra: probes_fmt_<format>/probes.jsonl và conversion_check.csv (50 câu để kiểm tra ngữ pháp bằng tay).
"""
import argparse
import csv
import json
import os
import random
import re

SPEAKER_LINE = re.compile(r"^(Speaker [12]): (.*)$")
SESSION_LINE = re.compile(r"^\[Session \d+\]$")

ADVERBS = {"also", "really", "just", "always", "never", "often", "usually", "still", "actually",
           "recently", "sometimes", "currently", "only", "definitely", "absolutely", "mostly",
           "even", "already", "finally", "truly", "rarely", "occasionally", "generally", "totally",
           "honestly", "personally", "certainly", "probably", "both", "all"}
MODALS = {"can", "could", "will", "would", "should", "may", "might", "must", "shall",
          "can't", "cannot", "won't", "wouldn't", "couldn't", "shouldn't", "did", "didn't",
          "used", "had", "hadn't", "was", "wasn't"}
IRREGULAR_PAST = set("""ate became began bit blew broke brought built bought caught chose came cost
cut dealt did drew drank drove fell fed felt fought found flew forgot forgave froze got gave went
grew hung heard hid hit held hurt kept knew laid led left lent let lay lost made meant met paid put
quit read rode rang rose ran said saw sold sent set shook shot showed shut sang sank sat slept slid
spoke spent spun split spread stood stole stuck stung struck swore swept swam took taught tore told
thought threw understood woke wore won wrote""".split())
THIRD_SG = {"have": "has", "do": "does", "go": "goes", "be": "is", "don't": "doesn't",
            "haven't": "hasn't", "am": "is", "'m": "is"}


def split_sentences(text):
    return [x.strip() for x in re.split(r"(?<=[.!?])\s+", text.strip()) if x.strip()]


def conjugate(verb):
    """Chia động từ hiện tại ngôi thứ ba số ít (heuristic)."""
    low = verb.lower()
    if low in THIRD_SG:
        out = THIRD_SG[low]
    elif low in MODALS or low in IRREGULAR_PAST or low.endswith("ed") or not low.isalpha():
        return verb
    elif re.search(r"[^aeiou]y$", low):
        out = low[:-1] + "ies"
    elif re.search(r"(s|x|z|ch|sh|o)$", low):
        out = low + "es"
    else:
        out = low + "s"
    return out.capitalize() if verb[:1].isupper() else out


def to_third_person(sentence, speaker):
    """Viết lại một câu persona ngôi thứ nhất thành câu có người sở hữu trong câu."""
    s = sentence
    # Các dạng rút gọn và cụm cố định
    rules = [
        (r"\b[Ii]'m\b", f"{speaker} is"), (r"\b[Ii] am\b", f"{speaker} is"),
        (r"\b[Ii]'ve\b", f"{speaker} has"), (r"\b[Ii]'ll\b", f"{speaker} will"),
        (r"\b[Ii]'d\b", f"{speaker} would"),
        (r"\b[Ii] don't\b", f"{speaker} doesn't"), (r"\b[Ii] do not\b", f"{speaker} does not"),
        (r"\bmyself\b", "themself"), (r"\bMy\b", f"{speaker}'s"), (r"\bmy\b", f"{speaker}'s"),
        (r"\bmine\b", f"{speaker}'s"), (r"\bme\b", speaker), (r"\bMe\b", speaker),
    ]
    for pat, rep in rules:
        s = re.sub(pat, rep, s)

    # "I <verb>" hoặc "I <adverb> <verb>": chia động từ, trừ khi "I" đứng sau "and"/"or" (chủ ngữ ghép)
    tokens = s.split(" ")
    out = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok in ("I", "i"):
            prev = out[-1].lower() if out else ""
            compound = prev in {"and", "or", "nor"}
            out.append(speaker)
            j = i + 1
            if not compound:
                while j < len(tokens) and tokens[j].lower().strip(",") in ADVERBS:
                    out.append(tokens[j])
                    j += 1
                if j < len(tokens):
                    word = tokens[j]
                    core = re.match(r"^([A-Za-z']+)(.*)$", word)
                    if core:
                        out.append(conjugate(core.group(1)) + core.group(2))
                        j += 1
            i = j
            continue
        out.append(tok)
        i += 1
    return " ".join(out)


def reformat_context(context, fmt):
    lines = []
    converted = []
    for line in context.split("\n"):
        m = SPEAKER_LINE.match(line)
        if not m or fmt == "line_prefix":
            lines.append(line)
            continue
        spk, body = m.group(1), m.group(2)
        sents = split_sentences(body)
        if fmt == "sentence_prefix":
            lines.append(" ".join(f"{spk}: {x}" for x in sents))
        else:  # third_person
            new = [to_third_person(x, spk) for x in sents]
            converted.extend(zip(sents, new))
            lines.append(" ".join(new))
    return "\n".join(lines), converted


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probes", default=os.path.join("probes", "probes.jsonl"))
    ap.add_argument("--format", required=True, choices=["line_prefix", "sentence_prefix", "third_person"])
    ap.add_argument("--variants", default="original,removal,subst_speaker")
    ap.add_argument("--options", default="keep", choices=["keep", "third"],
                    help="keep: giữ lựa chọn ngôi thứ nhất; third: viết lại lựa chọn theo người được hỏi")
    ap.add_argument("--out", default=None)
    ap.add_argument("--n-check", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    keep = set(args.variants.split(","))
    out_dir = args.out or f"probes_fmt_{args.format}" + ("_opt3" if args.options == "third" else "")
    os.makedirs(out_dir, exist_ok=True)

    n, pool = 0, []
    with open(args.probes, encoding="utf-8") as fin, \
            open(os.path.join(out_dir, "probes.jsonl"), "w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            p = json.loads(line)
            if p["variant"] not in keep:
                continue
            ctx, conv = reformat_context(p["context"], args.format)
            pool.extend(conv)
            options = list(p["options"])
            if args.options == "third":
                options = [o if o == "No information" else to_third_person(o, p["target_speaker"])
                           for o in options]
            p.update({"context": ctx, "options": options, "memory_format": args.format,
                      "prompt": render_prompt(ctx, p["target_speaker"], options)})
            if args.options == "third":
                # giữ nhãn văn bản khớp với lựa chọn mới để compute_metrics/compare_formats vẫn chạy
                for key in ("gold_evidence", "foil", "label_text", "replaced_fact"):
                    if key in p and p[key] != "No information":
                        p[key] = to_third_person(p[key], p["target_speaker"])
            fout.write(json.dumps(p, ensure_ascii=False) + "\n")
            n += 1

    if pool:
        uniq = list(dict.fromkeys(pool))
        random.Random(args.seed).shuffle(uniq)
        with open(os.path.join(out_dir, "conversion_check.csv"), "w", encoding="utf-8-sig", newline="") as f:
            w = csv.writer(f)
            w.writerow(["cau_goc", "cau_ngoi_thu_ba", "dung_ngu_phap_va_nghia (y/n)", "ghi_chu"])
            for a, b in uniq[: args.n_check]:
                w.writerow([a, b, "", ""])
    print(f"Đã ghi {n} probe vào {out_dir}/probes.jsonl"
          + (f"; {min(len(set(pool)), args.n_check)} câu cần kiểm tra tay trong {out_dir}/conversion_check.csv" if pool else ""))


if __name__ == "__main__":
    main()
