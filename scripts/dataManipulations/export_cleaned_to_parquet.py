#!/usr/bin/env python
"""
Export cleaned HF datasets to zstd-compressed Parquet.

Input  (per language/split):
  data/cleaned/<lang>/<split>   # HF Dataset saved to disk

Output (per language/split):
  parquet/cleaned/<lang>/<split>/part-*.parquet  (zstd)

Unified schema (one row == one function):
  uid:   string     # "<id>_<side>", side in {1,2}; for single datasets, side=1
  id:    int64
  side:  int8       # 1 or 2 (single datasets -> 1)
  lang:  string
  split: string
  label: int32?     # nullable; -1 or NULL for single datasets
  code:  string     # cleaned source code


# Example: export all rows of train+validation, 100k rows per file, label=NULL for single datasets
.venv/bin/python scripts/export_cleaned_to_parquet.py \
  --langs java,python,javascript,go \
  --splits train,validation \
  --chunk_rows 100000 \
  --label_single null \
  --zstd_level 7

"""

import argparse
from pathlib import Path
from typing import List, Tuple

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_from_disk
from tqdm import tqdm

PAIR_COLS = {"id", "func1", "func2", "label"}
SINGLE_COLS = {"id", "code"}


def infer_mode(cols: set) -> str:
    if PAIR_COLS.issubset(cols):
        return "pair"
    if SINGLE_COLS.issubset(cols):
        return "single"
    raise ValueError(f"Unsupported schema. Columns found: {sorted(cols)}")


def iter_chunks(n: int, chunk: int) -> List[Tuple[int, int]]:
    return [(i, min(i + chunk, n)) for i in range(0, n, chunk)]


def build_table_pair(batch, lang: str, split: str) -> pa.Table:
    ids = [int(x) for x in batch["id"]]
    f1 = batch["func1"]
    f2 = batch["func2"]
    labels = [int(x) for x in batch["label"]]

    n = len(ids)
    uid = [f"{ids[i]}_1" for i in range(n)] + [f"{ids[i]}_2" for i in range(n)]
    id_col = ids + ids
    side = [1] * n + [2] * n
    lang_col = [lang] * (2 * n)
    split_col = [split] * (2 * n)
    label_col = labels + labels
    code = f1 + f2

    return pa.table(
        {
            "uid": pa.array(uid, type=pa.string()),
            "id": pa.array(id_col, type=pa.int64()),
            "side": pa.array(side, type=pa.int8()),
            "lang": pa.array(lang_col, type=pa.string()),
            "split": pa.array(split_col, type=pa.string()),
            "label": pa.array(label_col, type=pa.int32()),
            "code": pa.array(code, type=pa.string()),
        }
    )


def build_table_single(batch, lang: str, split: str, label_as: str) -> pa.Table:
    # label_as: "null" -> nullable nulls, "minus1" -> fill -1 (int)
    ids = [int(x) for x in batch["id"]]
    code = batch["code"]
    n = len(ids)
    uid = [f"{pid}_1" for pid in ids]
    side = [1] * n
    lang_col = [lang] * n
    split_col = [split] * n

    if label_as == "minus1":
        label = [-1] * n
        label_arr = pa.array(label, type=pa.int32())
    else:
        # nullable labels (all nulls)
        label_arr = pa.array([None] * n, type=pa.int32())

    return pa.table(
        {
            "uid": pa.array(uid, type=pa.string()),
            "id": pa.array(ids, type=pa.int64()),
            "side": pa.array(side, type=pa.int8()),
            "lang": pa.array(lang_col, type=pa.string()),
            "split": pa.array(split_col, type=pa.string()),
            "label": label_arr,
            "code": pa.array(code, type=pa.string()),
        }
    )


def export_split(
    lang: str,
    split: str,
    src_dir: Path,
    out_dir: Path,
    limit: int,
    chunk_rows: int,
    label_single: str,
    compression_level: int,
):
    if not src_dir.exists():
        print(f"⏭ skip {lang}/{split}: {src_dir} not found")
        return

    ds = load_from_disk(str(src_dir))
    n_total = len(ds)
    if n_total == 0:
        print(f"⏭ skip {lang}/{split}: empty dataset")
        return

    cols = set(ds.column_names)
    mode = infer_mode(cols)

    if limit > 0 and limit < n_total:
        n_total = limit
        ds = ds.select(range(n_total))

    out_dir.mkdir(parents=True, exist_ok=True)
    parts = iter_chunks(n_total, chunk_rows)
    print(
        f"➡️  {lang}/{split}: {mode} mode, rows={n_total:,}, chunks={len(parts)} → {out_dir}"
    )

    for pi, (a, b) in enumerate(tqdm(parts, desc=f"{lang}/{split}")):
        batch = ds[a:b]
        if mode == "pair":
            table = build_table_pair(batch, lang, split)
        else:
            table = build_table_single(batch, lang, split, label_single)

        out_path = out_dir / f"part-{pi:05d}.parquet"
        pq.write_table(
            table,
            out_path,
            compression="zstd",
            compression_level=compression_level,
            use_dictionary=True,
            data_page_size=1 << 20,  # 1MB pages
            write_statistics=True,
        )

    print(f"✅ done {lang}/{split} → {out_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--langs", required=True, help="comma-separated, e.g. java,python,javascript,go"
    )
    ap.add_argument(
        "--splits",
        default="train,validation",
        help="comma-separated, e.g. train,validation,test",
    )
    ap.add_argument("--source_base", default="data/cleaned", help="HF datasets root")
    ap.add_argument("--out_base", default="parquet/cleaned", help="Parquet root (zstd)")
    ap.add_argument(
        "--limit", type=int, default=0, help="take first N rows per split (0=all)"
    )
    ap.add_argument(
        "--chunk_rows", type=int, default=100_000, help="rows per Parquet file"
    )
    ap.add_argument(
        "--label_single",
        choices=["null", "minus1"],
        default="null",
        help="label value for single datasets (nullable nulls or -1)",
    )
    ap.add_argument(
        "--zstd_level", type=int, default=7, help="zstd compression level (1..22)"
    )
    args = ap.parse_args()

    langs = [x.strip() for x in args.langs.split(",") if x.strip()]
    splits = [x.strip() for x in args.splits.split(",") if x.strip()]

    for lang in langs:
        for sp in splits:
            src = Path(args.source_base) / lang / sp
            out = Path(args.out_base) / lang / sp
            export_split(
                lang=lang,
                split=sp,
                src_dir=src,
                out_dir=out,
                limit=args.limit,
                chunk_rows=args.chunk_rows,
                label_single=args.label_single,
                compression_level=args.zstd_level,
            )


if __name__ == "__main__":
    main()
