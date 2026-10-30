"""
Xem chi tiết một bản ghi của msc_dialogue và ghi ra file peek_output.txt.

Cách chạy (trong thư mục dự án CoSA):
    python peek_msc.py
Nếu dữ liệu nằm chỗ khác:
    python peek_msc.py "đường\\dẫn\\tới\\msc_dialogue\\session_2\\test.txt"
"""
import json
import sys

DEFAULT = r"data\msc\msc_dialogue\session_2\test.txt"
path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT


def cut(value, n=3):
    """Giữ n phần tử đầu của list để bản in gọn."""
    if isinstance(value, list):
        return [cut(v, n) for v in value[:n]] + ([f"... (+{len(value) - n} phần tử)"] if len(value) > n else [])
    if isinstance(value, dict):
        return {k: cut(v, n) for k, v in value.items()}
    return value


with open(path, encoding="utf-8") as f:
    record = json.loads(f.readline())
    total = 1 + sum(1 for line in f if line.strip())

lines = [
    f"File: {path}",
    f"Số bản ghi trong file: {total}",
    f"Các trường cấp 1: {list(record.keys())}",
    "",
    "=== Nội dung bản ghi đầu tiên (đã rút gọn) ===",
    json.dumps(cut(record), ensure_ascii=False, indent=2),
]
text = "\n".join(lines)

with open("peek_output.txt", "w", encoding="utf-8") as out:
    out.write(text)

print(text[:6000])
print("\nĐã ghi đầy đủ vào peek_output.txt")
