#!/usr/bin/env python
"""
prepare_dataset.py
------------------
Очистка коду BigCloneBench (версія Hugging Face) і побудова metadata.csv
"""

import csv
import re
from pathlib import Path

from datasets import load_from_disk
from tqdm import tqdm

RAW_DIR = Path("data/raw/bigclonebench/java")
PROC_DIR = Path("data/processed/java")
PROC_DIR.mkdir(parents=True, exist_ok=True)
META_PATH = Path("metadata.csv")

# ──────────────────────────────────────────────────────────────────────────────
# 1. Завантажуємо датасет
print("📖  Loading dataset from", RAW_DIR)
ds = load_from_disk(str(RAW_DIR))
print(f"   → {len(ds):,} pairs")

# ──────────────────────────────────────────────────────────────────────────────
# 2. Регулярки для видалення коментарів
LINE_COMMENT = re.compile(r"//.*?$", re.MULTILINE)
BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.DOTALL)


def strip_comments(code: str) -> str:
    """Прибираємо // та /* */ коментарі, тримаємо структуру коду."""
    without_block = BLOCK_COMMENT.sub("", code)
    without_line = LINE_COMMENT.sub("", without_block)
    return without_line.strip()


# ──────────────────────────────────────────────────────────────────────────────
# 3. Проходимо датасет і записуємо результати
rows = []
for item in tqdm(ds, desc="Processing pairs"):
    pair_id = item["id"]
    label = int(item["label"])  # 1 = clone, 0 = non-clone

    # func1
    code1 = strip_comments(item["func1"])
    path1 = PROC_DIR / f"{pair_id}_1.java"
    path1.write_text(code1, encoding="utf-8")
    rows.append({"path": str(path1), "language": "java", "label": label})

    # func2
    code2 = strip_comments(item["func2"])
    path2 = PROC_DIR / f"{pair_id}_2.java"
    path2.write_text(code2, encoding="utf-8")
    rows.append({"path": str(path2), "language": "java", "label": label})

# ──────────────────────────────────────────────────────────────────────────────
# 4. Записуємо metadata.csv
with META_PATH.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["path", "language", "label"])
    writer.writeheader()
    writer.writerows(rows)

print(f"✅  Saved {len(rows):,} rows to {META_PATH}")
