#!/usr/bin/env python
import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.searching.lib_search import (
    embed,
    fetch_code_by_uid,
    identifiers,
    jaccard,
    load_index_and_meta,
    load_model,
    normalize_code,
    set_search_params,
    sha256,
)

# швидкий сценарій з раннім виходом (якщо дубль «дуже близький»)
# .venv/bin/python scripts/progressive_search.py \
#   --index_dir index/global \
#   --query_file samples/similar.java \
#   --lang_filter java \
#   --k 5 --early_emb 0.985 \
#   --show_top1_code --save_top1_to samples/similar_early.java
#
# # той самий для different.java (побачиш deep-режим і re-rank)
# .venv/bin/python scripts/progressive_search.py \
#   --index_dir index/global \
#   --query_file samples/different.java \
#   --lang_filter java \
#   --k 5 --early_emb 0.985 \
#   --alpha 0.7 --min_jaccard 0.15 --len_lo 0.7 --len_hi 1.4 \
#   --show_top1_code --save_top1_to samples/different_reranked.java


os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def stage_search(index, meta, qv, lang_filter: str, headroom: int):
    D, I = index.search(qv, headroom)
    out = []
    for score, idx in zip(D[0], I[0]):
        row_idx, uid, label, split, lang = meta[idx]
        if lang_filter != "any" and lang != lang_filter:
            continue
        out.append(
            {
                "emb": float(score),
                "uid": uid,
                "label": label,
                "split": split,
                "lang": lang,
            }
        )
    # сортуємо за embedding score
    out.sort(key=lambda x: x["emb"], reverse=True)
    return out


if __name__ == "__main__":
    os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
    ap = argparse.ArgumentParser()
    ap.add_argument("--index_dir", default="index/global")
    ap.add_argument("--model", default="microsoft/codebert-base")
    ap.add_argument("--query_file", required=True)
    ap.add_argument("--lang_filter", default="any", help="any|java|python|...")
    ap.add_argument("--k", type=int, default=5)

    # пороги раннього виходу
    ap.add_argument(
        "--early_emb", type=float, default=0.985, help="emb score to early-return"
    )
    ap.add_argument("--fast_head", type=int, default=30, help="headroom for fast probe")
    ap.add_argument(
        "--deep_head", type=int, default=120, help="headroom for deep probe"
    )

    # re-rank (друга стадія)
    ap.add_argument(
        "--alpha", type=float, default=0.7, help="blend weight for emb vs lexical"
    )
    ap.add_argument("--min_jaccard", type=float, default=0.10)
    ap.add_argument("--len_lo", type=float, default=0.5)
    ap.add_argument("--len_hi", type=float, default=2.0)
    ap.add_argument("--show_top1_code", action="store_true")
    ap.add_argument("--save_top1_to", default="")
    args = ap.parse_args()

    # 0) завантаження
    index, meta = load_index_and_meta(args.index_dir)
    tok, model, dev = load_model(args.model)

    # 1) нормалізація + (опційно) exact hash (місце для кеша, поки лише обчислюємо)
    q_text = Path(args.query_file).read_text(encoding="utf-8")
    q_norm = normalize_code(q_text)
    q_hash = sha256(q_norm)  # нині не використовуємо, але залишено для майбутнього кеша
    qv = embed(q_norm, tok, model, dev)

    # 2) швидкий зонд
    set_search_params(index, fast=True)
    fast = stage_search(index, meta, qv, args.lang_filter, headroom=args.fast_head)
    if fast and fast[0]["emb"] >= args.early_emb:
        hits = fast[: args.k]
        print("[mode] early-fast")
        for h in hits:
            print(
                f"{h['emb']:.4f}\t{h['uid']}\t{h['lang']}\t{h['split']}\tlabel={h['label']}"
            )
        # опція показати/зберегти top-1 код
        if args.show_top1_code or args.save_top1_to:
            top = hits[0]
            code = fetch_code_by_uid(top["lang"], top["split"], top["uid"])
            if code:
                header = f"\n----- TOP-1 CODE (emb={top['emb']:.4f}, uid={top['uid']}, {top['lang']}/{top['split']}) -----"
                print(header) if args.show_top1_code else None
                if args.show_top1_code:
                    print(code)
                    print("----- END -----")
                if args.save_top1_to:
                    Path(args.save_top1_to).write_text(code, encoding="utf-8")
                    print(f"[saved] → {args.save_top1_to}")
        raise SystemExit(0)

    # 3) поглиблення + re-rank
    set_search_params(index, fast=False)
    deep = stage_search(index, meta, qv, args.lang_filter, headroom=args.deep_head)

    # lexical signals
    lang_for_ids = args.lang_filter if args.lang_filter != "any" else "java"
    q_ids = identifiers(q_norm, lang_for_ids)
    q_len = len(q_norm)

    reranked = []
    for h in deep:
        code = fetch_code_by_uid(h["lang"], h["split"], h["uid"])
        if not code:
            continue
        ratio = len(code) / max(1, q_len)
        if ratio < args.len_lo or ratio > args.len_hi:
            continue
        c_ids = identifiers(code, h["lang"])
        jac = jaccard(q_ids, c_ids)
        if jac < args.min_jaccard:
            continue
        final = args.alpha * h["emb"] + (1 - args.alpha) * jac
        reranked.append((final, h["emb"], jac, h))

    reranked.sort(key=lambda x: x[0], reverse=True)
    hits = [h for _, _, _, h in reranked[: args.k]]

    print("[mode] deep")
    print("final\temb\tjacc\tuid\tlang\tsplit\tlabel")
    for final, emb, jac, h in reranked[: args.k]:
        print(
            f"{final:.4f}\t{emb:.4f}\t{jac:.3f}\t{h['uid']}\t{h['lang']}\t{h['split']}\t{h['label']}"
        )

    if hits and (args.show_top1_code or args.save_top1_to):
        top = hits[0]
        code = fetch_code_by_uid(top["lang"], top["split"], top["uid"])
        if code:
            header = f"\n----- TOP-1 CODE (emb={top['emb']:.4f}, uid={top['uid']}, {top['lang']}/{top['split']}) -----"
            print(header) if args.show_top1_code else None
            if args.show_top1_code:
                print(code)
                print("----- END -----")
            if args.save_top1_to:
                Path(args.save_top1_to).write_text(code, encoding="utf-8")
                print(f"[saved] → {args.save_top1_to}")
