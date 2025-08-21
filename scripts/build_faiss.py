#!/usr/bin/env python
"""
build_faiss.py
--------------
Будує FAISS-індекс із підготовлених ембеддингів.

Вхідні дані (створені build_embeddings.py):
  embeddings/<lang>/<split>/embeddings.memmap   # float32, L2-нормовані, [N,768]
  embeddings/<lang>/<split>/ids.tsv             # row_idx(uid локальний), uid, label, split, lang

Вихід:
  index/<...>/index.faiss
  index/<...>/meta.tsv                          # глобальна мапа row_idx -> uid,label,split,lang

Підтримує:
  - глобальний індекс (кілька мов/сплітів)
  - пер-мовні індекси (передати один lang)
  - типи індексу: flat / hnsw / ivfpq
"""

import argparse
from pathlib import Path

import faiss
import numpy as np
from tqdm import tqdm

DIM = 768  # розмір вектора CodeBERT (або GraphCodeBERT)


# ---------- утиліти роботи з memmap ----------


def count_rows(memmap_path: Path) -> int:
    """Порахувати кількість рядків у memmap за розміром файлу."""
    size = memmap_path.stat().st_size  # байти
    return size // (DIM * 4)  # float32 = 4 байти


def open_memmap(memmap_path: Path) -> tuple[np.memmap, int]:
    """Відкрити memmap раз і отримати (memmap, N)."""
    n = count_rows(memmap_path)
    if n <= 0:
        raise ValueError(f"memmap seems empty: {memmap_path}")
    X = np.memmap(memmap_path, mode="r", dtype="float32", shape=(n, DIM))
    return X, n


# ---------- будівники індексу ----------


def build_hnsw(dim: int, M: int = 32) -> faiss.Index:
    index = faiss.IndexHNSWFlat(dim, M, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = 200
    index.hnsw.efSearch = 128
    return index


def build_flat(dim: int) -> faiss.Index:
    return faiss.IndexFlatIP(dim)


def build_ivfpq(
    dim: int, nlist: int = 4096, m: int = 64, nbits: int = 8
) -> faiss.Index:
    quantizer = faiss.IndexFlatIP(dim)
    index = faiss.IndexIVFPQ(
        quantizer, dim, nlist, m, nbits, faiss.METRIC_INNER_PRODUCT
    )
    return index


def train_ivfpq(
    index: faiss.Index, langs: list[str], splits: list[str], per_part: int
) -> None:
    """Потренувати IVF-PQ на випадковій підвибірці з усіх мов/сплітів."""
    samples = []
    for lang in langs:
        for sp in splits:
            mem = Path(f"embeddings/{lang}/{sp}/embeddings.memmap")
            if not mem.exists():
                continue
            X, n = open_memmap(mem)
            k = min(per_part, n)
            idx = np.random.choice(n, size=k, replace=False)
            samples.append(np.asarray(X[idx]))
    if not samples:
        raise RuntimeError("Немає даних для тренування IVF-PQ (перевірте шляхи).")
    train_vecs = np.vstack(samples)
    print(f"🎯 training IVF-PQ on {train_vecs.shape[0]:,} samples …")
    index.train(train_vecs)


# ---------- додавання векторів + метаданих ----------


def add_vectors(index: faiss.Index, memmap_path: Path, batch: int = 100_000) -> int:
    """Додати всі вектори з memmap у FAISS батчами; повертає скільки додано."""
    X, total = open_memmap(memmap_path)  # X.shape = (total, DIM)
    cursor = 0
    pbar = tqdm(total=total, desc=f"add {memmap_path.parent.name}", unit="vec")
    while cursor < total:
        end = min(cursor + batch, total)
        xb = np.asarray(X[cursor:end])  # виділяємо RAM тільки під блок
        index.add(xb)
        pbar.update(end - cursor)
        cursor = end
    pbar.close()
    return total


def append_meta(ids_path: Path, meta_out, global_offset: int) -> int:
    """
    Переписуємо локальний ids.tsv у глобальний meta.tsv,
    замінюючи row_idx на (global_offset + локальний_рядок).
    ВАЖЛИВО: уникаємо f-string з backslash у виразі.
    """
    n = 0
    with ids_path.open("r", encoding="utf-8") as f_in:
        _ = f_in.readline()  # skip header
        for i, line in enumerate(f_in):
            # розділяємо 1-й раз по табу поза f-рядком:
            tail = line.split("\t", 1)[1]
            meta_out.write(str(global_offset + i) + "\t" + tail)
            n += 1
    return n


# ---------- CLI ----------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--langs", default="java", help="кома-сепарований список мов: java,python"
    )
    ap.add_argument(
        "--splits", default="train,validation", help="спліти: train,validation,test"
    )
    ap.add_argument("--index_type", choices=["flat", "hnsw", "ivfpq"], default="hnsw")
    ap.add_argument(
        "--out", default="index/global", help="каталог для збереження індексу"
    )
    # HNSW
    ap.add_argument("--hnsw_M", type=int, default=32)
    # IVF-PQ
    ap.add_argument("--nlist", type=int, default=4096)
    ap.add_argument("--pq_m", type=int, default=64)
    ap.add_argument("--pq_nbits", type=int, default=8)
    ap.add_argument(
        "--train_per_part",
        type=int,
        default=200_000,
        help="скільки зразків брати на тренування IVF-PQ з кожної частини",
    )
    # Додавання
    ap.add_argument("--add_batch", type=int, default=100_000)
    args = ap.parse_args()

    langs = [x.strip() for x in args.langs.split(",") if x.strip()]
    splits = [x.strip() for x in args.splits.split(",") if x.strip()]

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    index_path = out_dir / "index.faiss"
    meta_path = out_dir / "meta.tsv"

    # 1) створюємо індекс
    if args.index_type == "flat":
        index = build_flat(DIM)
    elif args.index_type == "hnsw":
        index = build_hnsw(DIM, M=args.hnsw_M)
    else:
        index = build_ivfpq(DIM, nlist=args.nlist, m=args.pq_m, nbits=args.pq_nbits)

    # 2) якщо IVF-PQ — тренуємо
    if isinstance(index, faiss.IndexIVFPQ):
        train_ivfpq(index, langs, splits, args.train_per_part)

    # 3) додаємо вектори і будуємо глобальну мета-мапу
    with meta_path.open("w", encoding="utf-8", newline="") as meta_out:
        meta_out.write("row_idx\tuid\tlabel\tsplit\tlang\n")
        global_offset = 0

        for lang in langs:
            for sp in splits:
                base = Path(f"embeddings/{lang}/{sp}")
                mem = base / "embeddings.memmap"
                ids = base / "ids.tsv"
                if not mem.exists() or not ids.exists():
                    print(f"⏭ skip {lang}/{sp} (missing)")
                    continue

                added = add_vectors(index, mem, batch=args.add_batch)
                wrote = append_meta(ids, meta_out, global_offset)

                if wrote != added:
                    raise RuntimeError(
                        f"meta lines ({wrote}) != vectors added ({added}) for {lang}/{sp}"
                    )
                global_offset += added

    # 4) зберігаємо індекс
    faiss.write_index(index, str(index_path))
    print(f"✅ saved index → {index_path}")
    print(f"🗂  metadata   → {meta_path}")


if __name__ == "__main__":
    # ←←← це і є «точка входу» скрипта в Python.
    # Якщо файл запускають напряму (python build_faiss.py),
    # викликається main(). Якщо імпортують як модуль — ні.
    main()
