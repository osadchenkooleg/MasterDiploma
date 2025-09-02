#!/usr/bin/env python
"""
Завантажує датасет BigCloneBench (версія Hugging Face / CodeXGLUE)
і зберігає у data/raw/bigclonebench/java.
"""
from datasets import load_dataset

TARGET_DIR = "data/raw/bigclonebench/java"

print("⬇  Downloading BigCloneBench (Hugging Face)…")
ds = load_dataset(
    "google/code_x_glue_cc_clone_detection_big_clone_bench",
    split="train",  # у цьому датасеті train містить усі 892К пар
)

print(ds)  # швидка перевірка: кількість прикладів
ds.save_to_disk(TARGET_DIR)  # ≈ 6 GB на диску
print(f"✅ Saved to {TARGET_DIR}")
