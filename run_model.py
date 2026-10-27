"""
Chạy một mô hình ngôn ngữ (HuggingFace) trên tập probe, chấm bằng xác suất token A/B/C.

Với mỗi probe, mô hình đọc prompt (qua chat template nếu có), rồi so sánh xác suất
token tiếp theo là "A", "B" hay "C". Không sinh văn bản tự do nên nhanh và chấm được tự động.

Cách chạy (Kaggle/Colab có GPU):
    pip install -q transformers accelerate bitsandbytes
    python run_model.py --model <tên-mô-hình-trên-HuggingFace> --load-4bit
Chạy thử nhanh 50 probe:
    python run_model.py --model <tên-mô-hình> --load-4bit --limit 50

Đầu ra: predictions/<tên-mô-hình>.jsonl, mỗi dòng gồm id, pred (0/1/2), xác suất A/B/C.
Chạy lại cùng lệnh sẽ bỏ qua các probe đã có kết quả (tiếp tục khi bị ngắt).
"""
import argparse
import json
import os
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def build_inputs(tokenizer, prompt):
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
        ) + "Answer: ("
    return prompt + "\nAnswer: ("


def letter_ids(tokenizer):
    """Lấy id token cho A/B/C (có và không có khoảng trắng phía trước)."""
    ids = []
    for letter in "ABC":
        variants = set()
        for text in (letter, " " + letter):
            toks = tokenizer.encode(text, add_special_tokens=False)
            if toks:
                variants.add(toks[-1])
        ids.append(sorted(variants))
    return ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--probes", default=os.path.join("probes", "probes.jsonl"))
    ap.add_argument("--out-dir", default="predictions")
    ap.add_argument("--load-4bit", action="store_true", help="lượng tử hóa 4-bit để vừa GPU 16 GB")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-length", type=int, default=2048)
    ap.add_argument("--limit", type=int, default=0, help="chỉ chạy N probe đầu (để thử)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, args.model.replace("/", "__") + ".jsonl")
    done = set()
    if os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as f:
            done = {json.loads(line)["id"] for line in f if line.strip()}

    probes = [json.loads(line) for line in open(args.probes, encoding="utf-8") if line.strip()]
    if args.limit:
        probes = probes[: args.limit]
    todo = [p for p in probes if p["id"] not in done]
    print(f"Tổng {len(probes)} probe, đã có {len(done)}, cần chạy {len(todo)}")
    if not todo:
        return

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"  # nếu quá dài, cắt phần đầu ngữ cảnh chứ không cắt câu hỏi
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    kwargs = {"device_map": "auto", "dtype": torch.float16}
    if args.load_4bit:
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type="nf4")
    model = AutoModelForCausalLM.from_pretrained(args.model, **kwargs)
    model.eval()

    cand = letter_ids(tokenizer)
    t0 = time.time()
    with open(out_path, "a", encoding="utf-8") as out, torch.no_grad():
        for start in range(0, len(todo), args.batch_size):
            batch = todo[start: start + args.batch_size]
            texts = [build_inputs(tokenizer, p["prompt"]) for p in batch]
            enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True,
                            max_length=args.max_length, add_special_tokens=False).to(model.device)
            logits = model(**enc).logits[:, -1, :].float()
            logprobs = torch.log_softmax(logits, dim=-1)
            for p, lp in zip(batch, logprobs):
                scores = [torch.logsumexp(lp[ids], dim=0).item() for ids in cand]
                probs = torch.softmax(torch.tensor(scores), dim=0).tolist()
                out.write(json.dumps({"id": p["id"], "pred": int(max(range(3), key=lambda i: scores[i])),
                                      "probs": [round(x, 4) for x in probs]}) + "\n")
            out.flush()
            n_done = start + len(batch)
            if (start // args.batch_size) % 20 == 0 or n_done == len(todo):
                rate = n_done / (time.time() - t0)
                eta = (len(todo) - n_done) / rate / 60
                print(f"{n_done}/{len(todo)}  {rate:.1f} probe/giây  còn khoảng {eta:.0f} phút")
    elapsed = time.time() - t0
    print(f"Xong: {out_path}  ({len(todo)} probe trong {elapsed:.0f} giây, {len(todo)/elapsed:.1f} probe/giây)")


if __name__ == "__main__":
    main()