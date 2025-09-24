#!/usr/bin/env python3
"""
populate_eval_pairs_ch.py — Build evaluation pairs directly from Parquet and store them in ClickHouse.

Key features
- Reads Parquet shards (e.g., parquet/cleaned/<lang>/{train,validation}/*.parquet) via pyarrow
- Groups examples for positives by a **group key** from:
    • a Parquet **column** (e.g., id/task_id)  [--group-mode column --group-col <col>]
    • a **regex** over a path column          [--group-mode regex  --path-col <col> --group-regex ...]
- Samples **positives** per group (no huge combinations) and **negatives** globally with caps
- Materializes code snippets to temp files; inserts (path_a/path_b) into ClickHouse `eval_pairs`
- Rich logging + progress; safety caps: --max-rows-per-file, --max-groups, --max-pairs-per-lang
- Convenience: set --langs auto to detect all languages under --root

Requirements
  pip install pyarrow clickhouse-connect
"""
from __future__ import annotations

import argparse
import logging
import os
import random
import re
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import pyarrow.parquet as pq
except Exception as e:
    raise SystemExit("pyarrow is required. `pip install pyarrow`")

try:
    from clickhouse_connect import get_client
except Exception as e:
    raise SystemExit("clickhouse-connect is required. `pip install clickhouse-connect`")

# -------------------------- Logging --------------------------
LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)s | %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, datefmt="%H:%M:%S")
log = logging.getLogger("populate_pairs_parquet")


# -------------------------- Config ---------------------------
@dataclass
class CHCfg:
    host: str
    user: str
    pwd: str
    db: str


COLS = ["id_a", "id_b", "label", "path_a", "path_b", "lang", "split"]

DDL = """
CREATE TABLE IF NOT EXISTS eval_pairs
(
  id_a   String,
  id_b   String,
  label  Int8,          -- 1 = positive (similar), 0 = negative
  path_a String,
  path_b String,
  lang   LowCardinality(String) DEFAULT '',
  split  LowCardinality(String) DEFAULT 'test'
)
ENGINE = MergeTree
ORDER BY (split, lang, id_a, id_b)
"""

TMP_DIR: Path  # set in main()

# --------------------- ClickHouse helpers --------------------


def ch_client(ch: CHCfg):
    return get_client(host=ch.host, username=ch.user, password=ch.pwd, database=ch.db)


def ensure_table(client):
    client.command(DDL)


def clear_split(client, split: str, table: str = "eval_pairs"):
    client.command(
        f"ALTER TABLE {table} DELETE WHERE split = %(s)s", parameters={"s": split}
    )


def insert_rows(
    client,
    table: str,
    items: List[Tuple[Path, Path, int, str]],
    ch_split: str,
    batch_size: int = 10000,
    log_every: int = 10000,
) -> int:
    total = 0
    batch = []
    t0 = time.perf_counter()
    for idx, (pa, pb, lbl, lang) in enumerate(items, 1):
        batch.append((pa.name, pb.name, int(lbl), str(pa), str(pb), lang, ch_split))
        if len(batch) >= batch_size:
            client.insert(table, batch, column_names=COLS)
            total += len(batch)
            if total % max(log_every, 1) == 0:
                log.info(
                    f"  inserted {total} rows so far (last batch={len(batch)}), elapsed {time.perf_counter()-t0:.1f}s"
                )
            batch = []
    if batch:
        client.insert(table, batch, column_names=COLS)
        total += len(batch)
    log.info(f"Inserted total {total} rows in {time.perf_counter()-t0:.1f}s")
    return total


# -------------------- Parquet readers/utils ------------------


def list_parquet(lang_root: Path, splits: List[str]) -> List[Path]:
    files: List[Path] = []
    for sp in splits:
        d = lang_root / sp
        if d.exists():
            files.extend(sorted(p for p in d.glob("*.parquet") if p.is_file()))
    files.extend(
        sorted(p for p in lang_root.glob("*.parquet") if p.is_file())
    )  # also allow directly under lang root
    return files


def read_rows_from_parquet(
    pq_path: Path,
    code_col: str,
    path_col: str,  # used only for regex grouping
    lang: Optional[str],
    lang_col: Optional[str],
    group_mode: str,  # 'column' or 'regex'
    group_regex: Optional[str],
    group_col: Optional[str],
    max_rows: Optional[int] = None,
    log_every: int = 20000,
) -> List[dict]:
    t0 = time.perf_counter()
    table = pq.read_table(pq_path)
    cols = set(table.column_names)
    log.debug(f"[{pq_path.name}] columns={table.column_names}")

    # Required columns by mode
    if code_col not in cols:
        raise SystemExit(f"Parquet {pq_path} missing required column: {code_col}")
    if group_mode == "regex":
        if not path_col or path_col not in cols:
            raise SystemExit(
                f"Parquet {pq_path} missing required column for regex grouping: {path_col or '<unset path_col>'}"
            )
        if not group_regex:
            raise SystemExit("--group-mode regex requires --group-regex")
    if group_mode == "column":
        if not group_col or group_col not in cols:
            raise SystemExit(
                f"--group-mode column requires --group-col present in Parquet (missing {group_col or '<unset group_col>'} in {pq_path})"
            )

    pa_code = table[code_col]
    pa_lang = (
        table[lang_col] if (lang is None and lang_col and lang_col in cols) else None
    )
    pa_path = table[path_col] if (group_mode == "regex") else None
    pa_group = table[group_col] if (group_mode == "column") else None

    rows: List[dict] = []
    n = len(table)
    limit = min(n, max_rows) if max_rows else n
    log.info(f"Reading {pq_path.name}: rows={n} limit={limit} mode={group_mode}")

    for i in range(limit):
        if i and (i % log_every == 0):
            log.info(f"  {pq_path.name}: processed {i}/{limit} rows…")
        try:
            code = pa_code[i].as_py()
            if code is None:
                continue
            if group_mode == "regex":
                pth = pa_path[i].as_py()
                m = re.match(group_regex, pth) if pth is not None else None
                g = m.group("g") if m else None
            else:  # column
                g = pa_group[i].as_py()
                pth = None
            L = (
                lang
                if lang is not None
                else (pa_lang[i].as_py() if pa_lang is not None else "")
            )
            rows.append({"code": code, "path": pth, "group": g, "lang": L})
        except Exception:
            log.exception(f"  {pq_path.name}: failed to read row {i}")

    log.info(
        f"Read {len(rows)} rows from {pq_path.name} in {time.perf_counter()-t0:.1f}s"
    )
    return rows


def materialize_code(
    tmp_dir: Path, lang: str, split: str, code: str, idx: int, log_every: int = 20000
) -> Path:
    if idx and idx % max(log_every, 1) == 0:
        log.info(f"  materialized {idx} files so far…")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    fn = tmp_dir / f"{lang}_{split}_{uuid.uuid4().hex}.code"
    fn.write_text(code, encoding="utf-8")
    return fn.resolve()


# ------------------ Pair planning (optimized) ----------------


def make_pairs_from_rows(
    rows: List[dict],
    pos_per_group: int,
    neg_ratio: float,
    max_neg_per_lang: Optional[int],
    neg_per_group_legacy: int,  # kept for back-compat; not used in new fast sampler
    split: str,
    rng: random.Random,
    require_group: bool = True,
    log_every_groups: int = 10_000,
    max_groups: Optional[int] = None,
    max_pairs_per_lang: Optional[int] = None,
) -> List[Tuple[Path, Path, int, str]]:
    t0 = time.perf_counter()
    # Build groups: group_key -> list of row indices
    groups: Dict[str, List[int]] = {}
    for i, r in enumerate(rows):
        g = r.get("group")
        if require_group and (g is None or g == ""):
            continue
        groups.setdefault(str(g), []).append(i)

    total_groups = len(groups)
    log.info(f"Grouping: groups={total_groups} (only groups with >=1 row)")

    # Optional cap on groups for very large corpora
    if max_groups is not None and total_groups > max_groups:
        gkeys = list(groups.keys())
        rng.shuffle(gkeys)
        keep = set(gkeys[:max_groups])
        groups = {k: v for k, v in groups.items() if k in keep}
        total_groups = len(groups)
        log.warning(f"Limiting to max-groups={max_groups} → kept {total_groups}")

    # --- Positives: sample without enumerating all combinations ---
    pairs_idx: List[Tuple[int, int, int]] = []
    pos_planned = 0
    for gi, (g, idxs) in enumerate(groups.items(), 1):
        if gi % max(log_every_groups, 1) == 0:
            log.info(
                f"  positives: processed groups {gi}/{total_groups} (pairs so far: +{pos_planned} / {len(pairs_idx)})"
            )
        k = len(idxs)
        if k < 2:
            continue
        want = min(pos_per_group, (k * (k - 1)) // 2)
        if want <= 0:
            continue
        if k <= 64:
            # enumerate small set, shuffle, take first N
            enum_pairs: List[Tuple[int, int]] = []
            for a in range(k):
                for b in range(a + 1, k):
                    enum_pairs.append((idxs[a], idxs[b]))
            rng.shuffle(enum_pairs)
            take = enum_pairs[:want]
        else:
            # large group: random sample unique pairs
            seen = set()
            take: List[Tuple[int, int]] = []
            attempts = 0
            max_attempts = want * 10
            while len(take) < want and attempts < max_attempts:
                a, b = rng.sample(idxs, 2)
                key = (a, b) if a < b else (b, a)
                if key in seen:
                    attempts += 1
                    continue
                seen.add(key)
                take.append(key)
                attempts += 1
        for a, b in take:
            pairs_idx.append((a, b, 1))
        pos_planned += len(take)

    log.info(f"Planned positives: +pos={pos_planned}")

    # --- Negatives: random sampling with caps (no per-group sweep) ---
    neg_planned = 0
    gkeys = [k for k, v in groups.items() if len(v) >= 1]
    G = len(gkeys)

    # target negatives based on ratio/caps
    target_neg = int(pos_planned * max(0.0, neg_ratio))
    if max_neg_per_lang is not None:
        target_neg = min(target_neg, max_neg_per_lang)
    if max_pairs_per_lang is not None:
        target_neg = min(target_neg, max(0, max_pairs_per_lang - len(pairs_idx)))

    if G >= 2 and target_neg > 0:
        log.info(f"Sampling negatives: target_neg={target_neg} from {G} groups")
        while neg_planned < target_neg:
            g = rng.choice(gkeys)
            og = g
            # ensure different group
            attempts = 0
            while og == g and attempts < 10:
                og = rng.choice(gkeys)
                attempts += 1
            if og == g:
                continue
            ia = rng.choice(groups[g])
            ib = rng.choice(groups[og])
            pairs_idx.append((ia, ib, 0))
            neg_planned += 1
            if neg_planned % max(log_every_groups, 1) == 0:
                log.info(
                    f"  negatives: {neg_planned}/{target_neg} sampled (total pairs so far={len(pairs_idx)})"
                )
    else:
        log.info("Skipping negatives (not enough groups or target_neg=0)")

    log.info(f"Planned negatives: -neg={neg_planned} (total planned={len(pairs_idx)})")

    # Optional cap on total pairs (final safeguard)
    if max_pairs_per_lang is not None and len(pairs_idx) > max_pairs_per_lang:
        random.shuffle(pairs_idx)
        pairs_idx = pairs_idx[:max_pairs_per_lang]
        log.warning(f"Limiting total pairs to max-pairs-per-lang={max_pairs_per_lang}")

    # --- Materialize to temp files with progress ---
    out: List[Tuple[Path, Path, int, str]] = []
    for k, (ia, ib, lbl) in enumerate(pairs_idx, 1):
        if k % (max(log_every_groups, 2) // 2) == 0:
            log.info(f"  materializing pairs: {k}/{len(pairs_idx)}…")
        ra = rows[ia]
        rb = rows[ib]
        lang = ra.get("lang") or rb.get("lang") or ""
        pa = materialize_code(
            TMP_DIR / lang / split, lang, split, ra["code"], k, log_every=10**9
        )
        pb = materialize_code(
            TMP_DIR / lang / split, lang, split, rb["code"], k, log_every=10**9
        )
        out.append((pa, pb, lbl, lang))

    log.info(
        f"Materialized files for {len(out)} pairs in {time.perf_counter()-t0:.1f}s"
    )
    return out


# ----------------------------- CLI ---------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Root like parquet/cleaned with {lang}/{split}/*.parquet",
    )
    ap.add_argument(
        "--langs",
        nargs="*",
        type=str,
        required=True,
        help="Languages to include (e.g., java python) or 'auto' to scan subdirs",
    )
    ap.add_argument(
        "--splits",
        nargs="*",
        type=str,
        default=["train", "validation"],
        help="Subfolders with parquet shards",
    )
    ap.add_argument(
        "--split",
        type=str,
        default="test",
        help="Target eval split name to store in ClickHouse",
    )

    # Parquet schema
    ap.add_argument("--code-col", type=str, default="code")
    ap.add_argument(
        "--path-col", type=str, default="path", help="Used only for --group-mode regex"
    )
    ap.add_argument("--lang-col", type=str, default=None)

    # Grouping
    ap.add_argument(
        "--group-mode",
        type=str,
        default="column",
        choices=["column", "regex"],
        help="How to form positives",
    )
    ap.add_argument(
        "--group-regex",
        type=str,
        default=None,
        help="Regex with named group 'g' to extract grouping key from path (regex mode)",
    )
    ap.add_argument(
        "--group-col",
        type=str,
        default=None,
        help="Parquet column to use as grouping key (column mode)",
    )

    # Sampling & perf
    ap.add_argument("--pos-per-group", type=int, default=5)
    ap.add_argument(
        "--neg-ratio",
        type=float,
        default=1.0,
        help="Max negatives as a ratio of positives (e.g., 0.25 = 1 neg per 4 pos)",
    )
    ap.add_argument(
        "--max-neg-per-lang",
        type=int,
        default=None,
        help="Hard cap on number of negatives per language",
    )
    ap.add_argument(
        "--max-rows-per-file", type=int, default=50000, help="Safety cap per shard"
    )
    ap.add_argument(
        "--max-groups",
        type=int,
        default=None,
        help="Process at most this many groups (after grouping)",
    )
    ap.add_argument(
        "--max-pairs-per-lang",
        type=int,
        default=None,
        help="Hard cap on total pairs per language",
    )
    ap.add_argument("--seed", type=int, default=13)

    # Materialization
    ap.add_argument("--tmp-dir", type=Path, default=Path(".cache/eval_pairs"))

    # ClickHouse
    ap.add_argument("--ch-host", type=str, default=os.getenv("CH_HOST", "localhost"))
    ap.add_argument("--ch-user", type=str, default=os.getenv("CH_USER", "default"))
    ap.add_argument("--ch-pass", type=str, default=os.getenv("CH_PASS", ""))
    ap.add_argument("--ch-db", type=str, default=os.getenv("CH_DB", "codebase"))
    ap.add_argument("--table", type=str, default="eval_pairs")
    ap.add_argument("--create-table", action="store_true")
    ap.add_argument("--clear-split", action="store_true")
    ap.add_argument("--batch-size", type=int, default=10000)

    # Logging
    ap.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )
    ap.add_argument(
        "--log-every", type=int, default=10000, help="Log progress every N rows/items"
    )

    args = ap.parse_args()
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    # Seed RNG for repeatability
    rng = random.Random(args.seed)

    global TMP_DIR
    TMP_DIR = args.tmp_dir

    # Resolve languages (auto discovery if requested)
    langs: List[str]
    if len(args.langs) == 1 and args.langs[0].lower() == "auto":
        if not args.root.exists():
            raise SystemExit(f"Root does not exist: {args.root}")
        langs = sorted([p.name for p in args.root.iterdir() if p.is_dir()])
        log.info(f"Auto-discovered languages: {langs}")
    else:
        langs = args.langs

    ch = CHCfg(args.ch_host, args.ch_user, args.ch_pass, args.ch_db)
    client = ch_client(ch)

    if args.create_table:
        log.info("Ensuring eval_pairs table exists…")
        ensure_table(client)
    if args.clear_split:
        log.info(f"Clearing existing rows for split={args.split}…")
        clear_split(client, args.split, args.table)

    t_all = time.perf_counter()
    total_pairs = 0

    for lang in langs:
        t_lang = time.perf_counter()
        lang_root = args.root / lang
        pq_files = list_parquet(lang_root, args.splits)
        if not pq_files:
            log.warning(
                f"[{lang}] No parquet files found under {lang_root} (splits={args.splits})"
            )
            continue
        log.info(
            f"[{lang}] Found {len(pq_files)} parquet shards: {[p.name for p in pq_files[:5]]}{'...' if len(pq_files)>5 else ''}"
        )

        rows_accum: List[dict] = []
        for pq_path in pq_files:
            try:
                rows = read_rows_from_parquet(
                    pq_path=pq_path,
                    code_col=args.code_col,
                    path_col=args.path_col,
                    lang=lang if args.lang_col is None else None,
                    lang_col=args.lang_col,
                    group_mode=args.group_mode,
                    group_regex=args.group_regex,
                    group_col=args.group_col,
                    max_rows=args.max_rows_per_file,
                    log_every=args.log_every,
                )
                rows_accum.extend(rows)
            except SystemExit:
                raise
            except Exception:
                log.exception(f"[{lang}] Failed to read {pq_path}")
        log.info(
            f"[{lang}] Total loaded rows={len(rows_accum)} in {time.perf_counter()-t_lang:.1f}s"
        )

        if not rows_accum:
            log.warning(f"[{lang}] No rows accumulated → skipping language")
            continue

        pairs = make_pairs_from_rows(
            rows=rows_accum,
            pos_per_group=args.pos_per_group,
            neg_ratio=args.neg_ratio,
            max_neg_per_lang=args.max_neg_per_lang,
            neg_per_group_legacy=0,
            split=args.split,
            rng=rng,
            require_group=(args.group_mode in ("regex", "column")),
            log_every_groups=args.log_every,
            max_groups=args.max_groups,
            max_pairs_per_lang=args.max_pairs_per_lang,
        )
        log.info(f"[{lang}] Built {len(pairs)} pairs; starting insertion…")

        inserted = (
            insert_rows(
                client, args.table, pairs, args.split, args.batch_size, args.log_every
            )
            if pairs
            else 0
        )
        total_pairs += inserted
        log.info(
            f"[{lang}] DONE: inserted={inserted} pairs in {time.perf_counter()-t_lang:.1f}s"
        )

    log.info(
        f"ALL DONE in {time.perf_counter()-t_all:.1f}s. Total inserted pairs: {total_pairs}"
    )


if __name__ == "__main__":
    main()
