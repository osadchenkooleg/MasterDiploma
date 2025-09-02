#!/usr/bin/env python
"""
Завантажує BigCloneBench (Hugging Face) по сплітах
і зберігає у data/raw/bigclonebench/java/<split>/.
"""
from pathlib import Path

from datasets import load_dataset

TARGET_ROOT = Path("data/raw/bigclonebench/java")
TARGET_ROOT.mkdir(parents=True, exist_ok=True)
DATASET = "google/code_x_glue_cc_clone_detection_big_clone_bench"

for split in ("train", "validation", "test"):
    print(f"⬇  downloading {split} …")
    ds = load_dataset(DATASET, split=split)
    out_dir = TARGET_ROOT / split
    ds.save_to_disk(out_dir)
    print(f"   saved → {out_dir}")
