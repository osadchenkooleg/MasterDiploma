#!/usr/bin/env python
import argparse
import re
from pathlib import Path

from datasets import DatasetDict, Features, Value, load_from_disk

STRIP_LINE = re.compile(r"//.*?$", re.MULTILINE)
STRIP_BLOCK = re.compile(r"/\*.*?\*/", re.DOTALL)


def strip_comments(code: str) -> str:
    return STRIP_LINE.sub("", STRIP_BLOCK.sub("", code)).strip()


def process_split(src_dir: Path, out_dir: Path):
    ds_raw = load_from_disk(src_dir)

    def _clean(batch):  # batched=True
        return {k: [strip_comments(c) for c in batch[k]] for k in ("func1", "func2")}

    ds_clean = ds_raw.map(_clean, batched=True, batch_size=10_000, num_proc=4)
    ds_clean.save_to_disk(out_dir)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True)  # bigclonebench
    p.add_argument("--lang", required=True)  # java / python
    p.add_argument("--split", default="all")  # train / validation / test / all
    args = p.parse_args()

    raw_base = Path("data/raw") / args.source / args.lang
    out_base = Path("data/cleaned") / args.lang
    splits = ["train", "validation", "test"] if args.split == "all" else [args.split]

    for sp in splits:
        process_split(raw_base / sp, out_base / sp)
        print(f"✔ cleaned {args.lang}/{sp}")
