"""
Kiểm tra nhanh cấu trúc tập MSC trên máy.

Cách chạy (trong terminal của VS Code):
    python inspect_msc.py <đường_dẫn_tới_thư_mục_MSC>

Ví dụ:
    python inspect_msc.py D:/data/msc

Script chỉ đọc file, không sửa gì. Hãy copy toàn bộ phần in ra và gửi lại.
"""
import json
import os
import sys
from collections import Counter

DATA_EXT = (".json", ".jsonl", ".txt", ".csv", ".parquet", ".arrow", ".pkl")


def short(value, limit=200):
    """Rút gọn giá trị để in cho dễ đọc."""
    text = json.dumps(value, ensure_ascii=False) if not isinstance(value, str) else value
    return text if len(text) <= limit else text[:limit] + f"... (+{len(text) - limit} ký tự)"


def describe(obj, indent=0, max_depth=4):
    """In cấu trúc lồng nhau: tên trường, kiểu dữ liệu, ví dụ giá trị."""
    pad = "  " * indent
    if indent > max_depth:
        print(f"{pad}...")
        return
    if isinstance(obj, dict):
        for key, val in obj.items():
            kind = type(val).__name__
            extra = f" (len={len(val)})" if isinstance(val, (list, dict, str)) else ""
            if isinstance(val, (dict, list)):
                print(f"{pad}- {key}: {kind}{extra}")
                describe(val, indent + 1, max_depth)
            else:
                print(f"{pad}- {key}: {kind}{extra} = {short(val, 120)}")
    elif isinstance(obj, list) and obj:
        print(f"{pad}[phần tử đầu tiên, kiểu {type(obj[0]).__name__}]")
        if isinstance(obj[0], (dict, list)):
            describe(obj[0], indent + 1, max_depth)
        else:
            print(f"{pad}  = {short(obj[0], 120)}")


def read_first_records(path, n=2):
    """Đọc n bản ghi đầu tiên của file .json hoặc .jsonl."""
    with open(path, encoding="utf-8") as f:
        first_char = f.read(1)
        f.seek(0)
        if first_char == "[":  # một mảng JSON
            data = json.load(f)
            return data[:n], len(data)
        records, count = [], 0
        for line in f:  # JSON Lines
            line = line.strip()
            if not line:
                continue
            count += 1
            if len(records) < n:
                records.append(json.loads(line))
        return records, count


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    root = sys.argv[1]

    # 1. Liệt kê các file dữ liệu
    print("=" * 70)
    print("1. CÁC FILE DỮ LIỆU")
    print("=" * 70)
    files = []
    for dirpath, _, filenames in os.walk(root):
        for name in sorted(filenames):
            if name.lower().endswith(DATA_EXT):
                full = os.path.join(dirpath, name)
                size_mb = os.path.getsize(full) / 1e6
                files.append(full)
                print(f"{os.path.relpath(full, root)}  ({size_mb:.1f} MB)")
    if not files:
        print("Không tìm thấy file dữ liệu nào. Kiểm tra lại đường dẫn.")
        return

    # 2. Xem cấu trúc của file JSON/JSONL đầu tiên trong mỗi thư mục con
    print()
    print("=" * 70)
    print("2. CẤU TRÚC BẢN GHI")
    print("=" * 70)
    seen_dirs = set()
    for path in files:
        folder = os.path.dirname(path)
        if folder in seen_dirs or not path.lower().endswith((".json", ".jsonl", ".txt")):
            continue
        seen_dirs.add(folder)
        try:
            records, count = read_first_records(path)
        except Exception as err:  # file không phải JSON
            print(f"\n[{os.path.relpath(path, root)}] không đọc được dạng JSON: {err}")
            continue
        print(f"\n[{os.path.relpath(path, root)}]  tổng số bản ghi: {count}")
        if records:
            describe(records[0])

    # 3. Thống kê nhanh nếu có cấu trúc giống MSC gốc của ParlAI
    print()
    print("=" * 70)
    print("3. THỐNG KÊ NHANH (nếu nhận ra cấu trúc)")
    print("=" * 70)
    for path in files:
        if not path.lower().endswith((".json", ".jsonl", ".txt")):
            continue
        try:
            records, _ = read_first_records(path, n=10**9)
        except Exception:
            continue
        if not records or not isinstance(records[0], dict):
            continue
        keys = Counter(k for r in records for k in r.keys())
        print(f"\n[{os.path.relpath(path, root)}]")
        print("  Các trường xuất hiện:", dict(keys))
        if "dialog" in records[0]:
            turns = [len(r["dialog"]) for r in records if isinstance(r.get("dialog"), list)]
            if turns:
                print(f"  Số lượt mỗi phiên: trung bình {sum(turns)/len(turns):.1f}, "
                      f"min {min(turns)}, max {max(turns)}")
        if "previous_dialogs" in records[0]:
            prev = [len(r["previous_dialogs"]) for r in records]
            print("  Số phiên trước:", dict(Counter(prev)))
        if "personas" in records[0]:
            p = records[0]["personas"]
            print("  Ví dụ personas:", short(p, 300))


if __name__ == "__main__":
    main()
