# python scripts/search_faiss.py --index_dir index/global --query_file sample.java --k 10
# python scripts/search_faiss.py --index_dir index/global --query_file sample.py --k 10 --lang_filter java

"""
.venv/bin/python scripts/search_faiss.py \
  --index_dir index/global \
  --query_file samples/similar.java \
  --k 5 --lang_filter java \
  --show_top1_code --save_top1_to samples/similar_top1_match_reranked.java \
  --alpha 0.7 --min_jaccard 0.10 --len_lo 0.5 --len_hi 2.0
"""


#!/usr/bin/env python
import argparse
import os
import re
from pathlib import Path

# Tidy up tokenizer forks warning and OpenMP clash on macOS
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import os

import faiss
import numpy as np

# Import order matters: torch first, then faiss
import torch
from datasets import load_from_disk
from transformers import AutoModel, AutoTokenizer

import duckdb

DIM = 768

JAVA_KEYWORDS = {
    "abstract",
    "assert",
    "boolean",
    "break",
    "byte",
    "case",
    "catch",
    "char",
    "class",
    "const",
    "continue",
    "default",
    "do",
    "double",
    "else",
    "enum",
    "extends",
    "final",
    "finally",
    "float",
    "for",
    "goto",
    "if",
    "implements",
    "import",
    "instanceof",
    "int",
    "interface",
    "long",
    "native",
    "new",
    "package",
    "private",
    "protected",
    "public",
    "return",
    "short",
    "static",
    "strictfp",
    "super",
    "switch",
    "synchronized",
    "this",
    "throw",
    "throws",
    "transient",
    "try",
    "void",
    "volatile",
    "while",
    "true",
    "false",
    "null",
    "var",
    "record",
    "sealed",
    "permits",
    "non-sealed",
}

IDENT_RE = re.compile(r"[A-Za-z_]\w+")


def load_meta(meta_path: Path):
    meta = []
    with meta_path.open("r", encoding="utf-8") as f:
        next(f)
        for line in f:
            row_idx, uid, label, split, lang = line.rstrip("\n").split("\t")
            meta.append((int(row_idx), uid, int(label), split, lang))
    return meta


def pick_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def embed(code: str, tok, model, device):
    with torch.no_grad():
        t = tok(code, return_tensors="pt", truncation=True, max_length=512).to(device)
        v = model(**t).last_hidden_state.mean(1)
        v = torch.nn.functional.normalize(v, p=2, dim=1)
        return v.cpu().numpy().astype("float32")


def fetch_code_by_uid(lang: str, split: str, uid: str) -> str | None:
    p = f"parquet/cleaned/{lang}/{split}/*.parquet"
    con = duckdb.connect()
    row = con.execute(
        "SELECT code FROM read_parquet(?) WHERE uid = ? LIMIT 1", [p, uid]
    ).fetchone()
    return row[0] if row else None


def identifiers(code: str, lang: str):
    toks = set(x for x in IDENT_RE.findall(code))
    if lang == "java":
        toks = {t for t in toks if t not in JAVA_KEYWORDS}
    return toks


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / max(1, len(a | b))


if __name__ == "__main__":
    os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
    ap = argparse.ArgumentParser()
    ap.add_argument("--index_dir", default="index/global")
    ap.add_argument("--model", default="microsoft/codebert-base")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument(
        "--headroom", type=int, default=50, help="search top-N then re-rank"
    )
    ap.add_argument("--lang_filter", default="any", help="any|java|python|...")
    ap.add_argument("--query_file", required=True)
    ap.add_argument("--show_top1_code", action="store_true")
    ap.add_argument("--save_top1_to", default="")
    # re-rank params
    ap.add_argument("--alpha", type=float, default=0.7, help="weight for embedding sim")
    ap.add_argument("--min_jaccard", type=float, default=0.10, help="discard if below")
    ap.add_argument("--len_lo", type=float, default=0.5, help="min length ratio")
    ap.add_argument("--len_hi", type=float, default=2.0, help="max length ratio")
    args = ap.parse_args()

    # load index + meta
    index = faiss.read_index(str(Path(args.index_dir) / "index.faiss"))
    meta = load_meta(Path(args.index_dir) / "meta.tsv")

    # load model
    device = pick_device()
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model).eval().to(device)

    # read query, embed
    query = Path(args.query_file).read_text(encoding="utf-8")
    qv = embed(query, tok, model, device)

    # stage-1: ANN search with headroom
    D, I = index.search(qv, max(args.k * 5, args.headroom))
    raw_results = []
    for score, idx in zip(D[0], I[0]):
        row_idx, uid, label, sp, lang = meta[idx]
        if args.lang_filter != "any" and lang != args.lang_filter:
            continue
        raw_results.append((float(score), uid, label, sp, lang))

    # stage-2: re-rank by lexical overlap + length ratio
    q_ids = identifiers(
        query, args.lang_filter if args.lang_filter != "any" else "java"
    )
    q_len = len(query)
    reranked = []
    for emb_score, uid, label, sp, lang in raw_results:
        code = fetch_code_by_uid(lang, sp, uid)
        if not code:
            continue
        cand_len = len(code)
        ratio = cand_len / max(1, q_len)
        if ratio < args.len_lo or ratio > args.len_hi:
            continue
        c_ids = identifiers(code, lang)
        jac = jaccard(q_ids, c_ids)
        if jac < args.min_jaccard:
            continue
        final = args.alpha * emb_score + (1 - args.alpha) * jac
        reranked.append((final, emb_score, jac, uid, label, sp, lang))

    reranked.sort(key=lambda x: x[0], reverse=True)
    results = reranked[: args.k] if reranked else []

    if not results:
        print(
            f"No duplication was found"
            f"(min_jaccard={args.min_jaccard}, len_ratio∈[{args.len_lo},{args.len_hi}], k={args.k})."
        )
        raise SystemExit(0)

    print("Top results (final_score | emb | jaccard):")
    for final, emb, jac, uid, lab, sp, lang in results:
        print(f"{final:.4f}\t{emb:.4f}\t{jac:.3f}\t{uid}\t{lang}\t{sp}\tlabel={lab}")

    if results and (args.show_top1_code or args.save_top1_to):
        final, emb, jac, uid, lab, sp, lang = results[0]
        code = fetch_code_by_uid(lang, sp, uid)
        if code is None:
            print(f"\n[warn] could not fetch code for uid={uid} {lang}/{sp}")
        else:
            header = (
                f"\n----- TOP-1 CODE -----\n"
                f"final={final:.4f} emb={emb:.4f} jaccard={jac:.3f} uid={uid} {lang}/{sp}\n"
            )
            if args.show_top1_code:
                print(header)
                print(code)
                print("----- END -----")
            if args.save_top1_to:
                out = Path(args.save_top1_to)
                out.write_text(code, encoding="utf-8")
                print(f"[saved] top-1 code → {out}")
