"""
Build a per-language stop-list of frequent token shingles (boilerplate).

Usage examples:
  python tools/build_stoplist.py --src data/train/go --lang go --n 5 --df 0.005 --out data/stoplists/go.jsonl
  python tools/build_stoplist.py --src data/train/java --lang java --topk 80000 --out data/stoplists/java.jsonl

Notes:
- Works over plain source files in a folder tree. It will scan recursively.
- Language affects only default file extensions; tokenization is language-agnostic (assumes prior light normalization).
- Output is JSONL with fields: {"lang","n","hash","shingle":[...],"df"}
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Iterable, List, Tuple

LANG_EXT = {
    "go": [".go"],
    "java": [".java"],
    "python": [".py"],
    "py": [".py"],
    "js": [".js"],
    "ts": [".ts"],
    "cpp": [".cpp", ".cc", ".hpp", ".h"],
}

TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z_0-9]*|==|!=|<=|>=|&&|\|\||[{}();,\[\].:+\-*/]")


def tokenize(code: str) -> List[str]:
    return TOKEN_RE.findall(code)


def shingles(tokens: List[str], n: int) -> Iterable[Tuple[str, ...]]:
    L = len(tokens)
    if L < n:
        return []
    return (tuple(tokens[i : i + n]) for i in range(L - n + 1))


def shingle_hash(s: Tuple[str, ...]) -> str:
    # 64-bit (hex) for compactness
    h = hashlib.blake2b("\u0001".join(s).encode("utf-8"), digest_size=8)
    return h.hexdigest()


def iter_source_files(root: Path, lang: str | None) -> Iterable[Path]:
    if lang and lang in LANG_EXT:
        exts = set(LANG_EXT[lang])
    else:
        exts = None  # take all
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if exts is None or p.suffix in exts:
            yield p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--src",
        type=Path,
        required=True,
        help="Root directory with code files (train split)",
    )
    ap.add_argument(
        "--lang",
        type=str,
        required=True,
        help="Language tag for the stop-list (go/java/python/...)",
    )
    ap.add_argument("--n", type=int, default=5, help="Shingle length (tokens)")
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--df",
        type=float,
        help="Document frequency threshold as a fraction (e.g. 0.005 = 0.5%)",
    )
    group.add_argument(
        "--topk",
        type=int,
        help="Take top-K most frequent shingles by document frequency",
    )
    ap.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output JSONL path (will be overwritten)",
    )
    args = ap.parse_args()

    files = list(iter_source_files(args.src, args.lang))
    if not files:
        raise SystemExit(f"No source files found under {args.src}")

    df = Counter()
    for p in files:
        try:
            code = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        toks = tokenize(code)
        uniq = set(shingles(toks, args.n))
        df.update(uniq)

    total_docs = len(files)
    if args.df is not None:
        cutoff = max(1, int(total_docs * args.df))
        chosen = [(s, c) for s, c in df.items() if c >= cutoff]
    else:
        chosen = df.most_common(args.topk)

    with args.out.open("w", encoding="utf-8") as f:
        for s, c in chosen:
            rec = {
                "lang": args.lang,
                "n": args.n,
                "hash": shingle_hash(s),
                "shingle": list(s),
                "df": int(c),
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Wrote {len(chosen)} shingles to {args.out} | docs={total_docs}")


if __name__ == "__main__":
    main()
