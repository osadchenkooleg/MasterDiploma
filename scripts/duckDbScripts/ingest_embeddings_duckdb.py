#!/usr/bin/env python
"""
Ingest memmap embeddings into DuckDB.

Expects per language/split:
  embeddings/<lang>/<split>/embeddings.memmap   # float32 [N, DIM], L2-normalized
  embeddings/<lang>/<split>/ids.tsv             # row_idx  uid  label  split  lang

Writes into duckdb/plag.db → table `embeddings` (Phase 2 schema).

Notes
- Works whether `emb` column is FLOAT[] or FLOAT[DIM].
- Uses Arrow FixedSizeList to match FLOAT[DIM] if present (HNSW-ready).
- ON CONFLICT(uid) DO NOTHING → safe to rerun.

# example: all four languages, train+validation
.venv/bin/python scripts/duckdb/ingest_embeddings_duckdb.py \
  --langs java,python,javascript,go \
  --splits train,validation \
  --batch 50000 \
  --db duckdb/plag.db

"""

import argparse
import uuid
from pathlib import Path
from typing import Iterator, List, Tuple

import numpy as np
import pyarrow as pa

import duckdb

DIM_DEFAULT = 768


def open_memmap(memmap_path: Path, dim: int) -> Tuple[np.memmap, int]:
    if not memmap_path.exists():
        raise FileNotFoundError(memmap_path)
    size_bytes = memmap_path.stat().st_size
    if size_bytes % (dim * 4) != 0:
        raise ValueError(f"File size not multiple of dim*4: {size_bytes} vs {dim}*4")
    n = size_bytes // (dim * 4)
    X = np.memmap(memmap_path, mode="r", dtype="float32", shape=(n, dim))
    return X, n


def ids_iter(ids_path: Path) -> Iterator[Tuple[str, int, str, str]]:
    """
    Yields: (uid, label, split, lang)
    Header is: row_idx\tuid\tlabel\tsplit\tlang
    """
    with ids_path.open("r", encoding="utf-8") as f:
        _ = next(f, None)  # header
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 5:  # skip malformed
                continue
            uid = parts[1]
            try:
                label = int(parts[2])
            except Exception:
                label = -1
            split = parts[3]
            lang = parts[4]
            yield (uid, label, split, lang)


def take_n(it: Iterator, n: int) -> List:
    out = []
    try:
        for _ in range(n):
            out.append(next(it))
    except StopIteration:
        pass
    return out


def make_arrow_table(
    uids, labels, splits, langs, vecs: np.ndarray, model: str, dim: int
) -> pa.Table:
    # FixedSizeList ensures compatibility with FLOAT[dim]
    flat = pa.array(vecs.reshape(-1), type=pa.float32())
    emb = pa.FixedSizeListArray.from_arrays(flat, dim)
    return pa.table(
        {
            "uid": pa.array(uids, type=pa.string()),
            "lang": pa.array(langs, type=pa.string()),
            "split": pa.array(splits, type=pa.string()),
            "label": pa.array(labels, type=pa.int32()),
            "model": pa.array([model] * len(uids), type=pa.string()),
            "dim": pa.array([dim] * len(uids), type=pa.int16()),
            "emb": emb,
        }
    )


def column_type(con: duckdb.DuckDBPyConnection, table: str, col: str) -> str:
    return con.execute(
        "SELECT data_type FROM information_schema.columns WHERE table_name=? AND column_name=?",
        [table, col],
    ).fetchone()[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--langs", required=True, help="comma-separated: java,python,javascript,go"
    )
    ap.add_argument("--splits", default="train,validation", help="comma-separated")
    ap.add_argument("--dim", type=int, default=DIM_DEFAULT)
    ap.add_argument("--model", default="microsoft/codebert-base")
    ap.add_argument("--batch", type=int, default=50000, help="rows per insert")
    ap.add_argument("--db", default="duckdb/plag.db")
    args = ap.parse_args()

    langs = [x.strip() for x in args.langs.split(",") if x.strip()]
    splits = [x.strip() for x in args.splits.split(",") if x.strip()]

    con = duckdb.connect(args.db)
    con.execute("PRAGMA threads=8;")
    con.execute("PRAGMA enable_object_cache=true;")

    # Info: emb column type (FLOAT[] vs FLOAT[dim]) — just to warn user
    try:
        emb_type = column_type(con, "embeddings", "emb")
        print(f"[info] embeddings.emb type = {emb_type}")
    except Exception:
        emb_type = "unknown"

    total = 0
    for lang in langs:
        for sp in splits:
            base = Path(f"embeddings/{lang}/{sp}")
            mem = base / "embeddings.memmap"
            ids = base / "ids.tsv"
            if not mem.exists() or not ids.exists():
                print(f"⏭ skip {lang}/{sp} (missing files)")
                continue

            X, N = open_memmap(mem, args.dim)
            meta_it = ids_iter(ids)
            cursor = 0
            inserted = 0
            batch_id = str(uuid.uuid4())

            print(f"➡️  ingest {lang}/{sp}: {N:,} vectors")
            while cursor < N:
                take = min(args.batch, N - cursor)
                meta = take_n(meta_it, take)
                if not meta:
                    raise RuntimeError(f"ids.tsv shorter than memmap for {lang}/{sp}")
                if len(meta) != take:
                    take = len(meta)

                uids = [m[0] for m in meta]
                labels = [m[1] for m in meta]
                splits_b = [m[2] for m in meta]
                langs_b = [m[3] for m in meta]
                vecs = np.asarray(X[cursor : cursor + take])  # materialize

                tbl = make_arrow_table(
                    uids, labels, splits_b, langs_b, vecs, args.model, args.dim
                )
                con.register("emb_batch", tbl)
                con.execute(
                    """
                    INSERT INTO embeddings BY NAME
                    SELECT * FROM emb_batch
                    ON CONFLICT (uid, lang) DO NOTHING
                """
                )
                con.unregister("emb_batch")

                cursor += take
                inserted += take
                total += take
                print(f"   +{take:,} (cursor={cursor:,}/{N:,})")

            con.execute(
                """
                INSERT INTO embed_batches (batch_id, model, lang, split, rows_ingested, source_path)
                VALUES (?, ?, ?, ?, ?, ?)
            """,
                [batch_id, args.model, lang, sp, inserted, str(base)],
            )
            print(f"✅ {lang}/{sp}: inserted {inserted:,}")

    # Flush WAL (safer after large ingest)
    con.execute("PRAGMA force_checkpoint;")
    con.close()
    print(f"TOTAL inserted: {total:,}")


if __name__ == "__main__":
    main()
