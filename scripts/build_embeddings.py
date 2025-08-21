#!/usr/bin/env python
"""
build_embeddings.py
Генерує embeddings для cleaned-датасетів (Arrow) батчами:
- входи: data/cleaned/<lang>/<split>/
- виходи: embeddings/<lang>/<split>/embeddings.memmap (float32, L2-нормовані)
          embeddings/<lang>/<split>/ids.tsv (row_idx,uid,label,split,lang)
UID = "<pair_id>_1" або "<pair_id>_2" (для func1/func2)
"""

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
    ap.add_argument("--lang", required=True, help="java | python | ...")
    ap.add_argument("--split", default="train", help="train|validation|test|all")
    ap.add_argument("--model", default="microsoft/codebert-base")
    ap.add_argument(
        "--batch_pairs", type=int, default=256, help="пар у батчі (x2 функцій)"
    )
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

        ds = load_from_disk(str(src_dir))  # має колонки: id, func1, func2, label
        n_pairs = len(ds)
        n_rows = n_pairs * 2  # func1 + func2

        out_dir = Path(f"embeddings/{args.lang}/{sp}")
        out_dir.mkdir(parents=True, exist_ok=True)
        mem_path = out_dir / "embeddings.memmap"
        ids_path = out_dir / "ids.tsv"

        # Попередньо виділяємо memmap під усі рядки
        emmap = np.memmap(mem_path, mode="w+", dtype="float32", shape=(n_rows, DIM))
        ids_f = ids_path.open("w", encoding="utf-8")
        ids_f.write("row_idx\tuid\tlabel\tsplit\tlang\n")

        cursor = 0
        step = args.batch_pairs
        print(f"➡️  {args.lang}/{sp}: {n_pairs:,} pairs ({n_rows:,} funcs)")

        for i in tqdm(
            range(0, n_pairs, step), total=(n_pairs + step - 1) // step, desc=f"{sp}"
        ):
            batch = ds[i : min(i + step, n_pairs)]
            # тексти: послідовно func1, потім func2
            texts = batch["func1"] + batch["func2"]
            # uid-и і лейбли
            ids1 = [f"{pid}_1" for pid in batch["id"]]
            ids2 = [f"{pid}_2" for pid in batch["id"]]
            labels = [int(x) for x in batch["label"]]
            labels2 = labels  # дублюємо для другої половини

            # ембедимо
            vecs = embed_texts(texts, tok, model, device)  # [2*B, 768]
            B = vecs.shape[0]

            # записуємо блок у memmap
            emmap[cursor : cursor + B] = vecs

            # лог у TSV (мінімум даних — швидкий append)
            for j, uid in enumerate(ids1):
                ids_f.write(f"{cursor+j}\t{uid}\t{labels[j]}\t{sp}\t{args.lang}\n")
            off = len(ids1)
            for j, uid in enumerate(ids2):
                ids_f.write(f"{cursor+off+j}\t{uid}\t{labels2[j]}\t{sp}\t{args.lang}\n")

            cursor += B

        ids_f.close()
        emmap.flush()
        print(f"✅ saved: {mem_path}  ({n_rows} x {DIM}),  map: {ids_path}")


if __name__ == "__main__":
    main()
