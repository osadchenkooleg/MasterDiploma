#!/usr/bin/env python
import argparse
from pathlib import Path

from datasets import load_dataset

SUPPORTED = ("python", "javascript", "java", "go", "php", "ruby")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--langs",
        default="python,javascript",
        help="comma-separated: python,javascript,java,go,php,ruby",
    )
    ap.add_argument(
        "--splits", default="train,validation", help="train,validation,test"
    )
    args = ap.parse_args()

    langs = [x.strip() for x in args.langs.split(",") if x.strip()]
    splits = [x.strip() for x in args.splits.split(",") if x.strip()]

    for lang in langs:
        if lang not in SUPPORTED:
            print(f"⛔ unsupported: {lang}")
            continue
        for sp in splits:
            print(f"⬇ CodeSearchNet {lang}/{sp}")
            ds = load_dataset("code_search_net", lang, split=sp, trust_remote_code=True)
            out = Path(f"data/raw/codesearchnet/{lang}/{sp}")
            out.mkdir(parents=True, exist_ok=True)
            ds.save_to_disk(out)
            print(f"   → saved to {out}")
