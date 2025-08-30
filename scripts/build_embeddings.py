#!/usr/bin/env python
"""
build_embeddings.py
Генерує embeddings для cleaned-датасетів (Arrow) батчами.
Підтримує обидва формати:
- pair:   {id, func1, func2, label}
- single: {id, code}
Вивід: embeddings/<lang>/<split>/{embeddings.memmap, ids.tsv}
"""
# .venv/bin/python scripts/build_embeddings.py --lang python --split train --batch 256
# .venv/bin/python scripts/build_embeddings.py --lang javascript --split train --batch 256
import argparse
from pathlib import Path

import numpy as np
import torch
from datasets import load_from_disk
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

DIM = 768  # CodeBERT hidden size


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def embed_texts(texts, tok, model, device):
    with torch.no_grad():
        toks = tok(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        toks = {k: v.to(device) for k, v in toks.items()}
        out = model(**toks).last_hidden_state  # [B, L, H]
        vec = out.mean(dim=1)  # mean-pooling -> [B, H]
        vec = torch.nn.functional.normalize(vec, p=2, dim=1)  # L2
        return vec.cpu().numpy().astype("float32")  # [B, 768]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", required=True, help="java | python | javascript | ...")
    ap.add_argument("--split", default="train", help="train|validation|test|all")
    ap.add_argument("--model", default="microsoft/codebert-base")
    ap.add_argument("--batch", type=int, default=256, help="розмір батчу (записів)")
    args = ap.parse_args()

    device = pick_device()
    print(f"🧠 device: {device}")

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model).to(device).eval()

    splits = ["train", "validation", "test"] if args.split == "all" else [args.split]

    for sp in splits:
        src_dir = Path(f"data/cleaned/{args.lang}/{sp}")
        if not src_dir.exists():
            print(f"⛔️ skip {args.lang}/{sp}: {src_dir} not found")
            continue

        ds = load_from_disk(str(src_dir))
        cols = set(ds.column_names)

        # ---- режим SINGLE (CodeSearchNet) ----
        if "code" in cols and "func1" not in cols:
            n_rows = len(ds)
            out_dir = Path(f"embeddings/{args.lang}/{sp}")
            out_dir.mkdir(parents=True, exist_ok=True)
            mem_path = out_dir / "embeddings.memmap"
            ids_path = out_dir / "ids.tsv"

            emmap = np.memmap(mem_path, mode="w+", dtype="float32", shape=(n_rows, DIM))
            with ids_path.open("w", encoding="utf-8") as f_ids:
                f_ids.write("row_idx\tuid\tlabel\tsplit\tlang\n")

                cursor = 0
                step = args.batch
                print(f"➡️  {args.lang}/{sp} (single): {n_rows:,} items")

                for i in tqdm(
                    range(0, n_rows, step),
                    total=(n_rows + step - 1) // step,
                    desc=f"{sp}:{args.lang}",
                ):
                    batch = ds[i : min(i + step, n_rows)]
                    texts = batch["code"]
                    vecs = embed_texts(texts, tok, model, device)  # [B, 768]
                    B = vecs.shape[0]
                    emmap[cursor : cursor + B] = vecs

                    ids = batch["id"]
                    for j, pid in enumerate(ids):
                        uid = f"{int(pid)}_1"
                        f_ids.write(f"{cursor+j}\t{uid}\t{-1}\t{sp}\t{args.lang}\n")

                    cursor += B

            emmap.flush()
            print(f"✅ saved: {mem_path}  ({n_rows} x {DIM}),  map: {ids_path}")
            continue

        # ---- режим PAIR (BigCloneBench) ----
        if "func1" in cols and "func2" in cols:
            n_pairs = len(ds)
            n_rows = n_pairs * 2

            out_dir = Path(f"embeddings/{args.lang}/{sp}")
            out_dir.mkdir(parents=True, exist_ok=True)
            mem_path = out_dir / "embeddings.memmap"
            ids_path = out_dir / "ids.tsv"

            emmap = np.memmap(mem_path, mode="w+", dtype="float32", shape=(n_rows, DIM))
            with ids_path.open("w", encoding="utf-8") as f_ids:
                f_ids.write("row_idx\tuid\tlabel\tsplit\tlang\n")

                cursor = 0
                step = args.batch
                print(
                    f"➡️  {args.lang}/{sp} (pair): {n_pairs:,} pairs ({n_rows:,} funcs)"
                )

                for i in tqdm(
                    range(0, n_pairs, step),
                    total=(n_pairs + step - 1) // step,
                    desc=f"{sp}:{args.lang}",
                ):
                    batch = ds[i : min(i + step, n_pairs)]
                    texts = batch["func1"] + batch["func2"]
                    vecs = embed_texts(texts, tok, model, device)  # [2*B, 768]
                    B = vecs.shape[0]
                    emmap[cursor : cursor + B] = vecs

                    ids1 = [f"{pid}_1" for pid in batch["id"]]
                    ids2 = [f"{pid}_2" for pid in batch["id"]]
                    labels = [int(x) for x in batch["label"]]
                    for j, uid in enumerate(ids1):
                        f_ids.write(
                            f"{cursor+j}\t{uid}\t{labels[j]}\t{sp}\t{args.lang}\n"
                        )
                    off = len(ids1)
                    for j, uid in enumerate(ids2):
                        f_ids.write(
                            f"{cursor+off+j}\t{uid}\t{labels[j]}\t{sp}\t{args.lang}\n"
                        )

                    cursor += B

            emmap.flush()
            print(f"✅ saved: {mem_path}  ({n_rows} x {DIM}),  map: {ids_path}")
            continue

        print(f"⛔️ unknown dataset schema at {src_dir}: columns={cols}")


if __name__ == "__main__":
    main()
