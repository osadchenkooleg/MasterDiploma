#!/usr/bin/env python3
"""
Build per-language boilerplate stop-lists across *all splits* and store them directly in ClickHouse.
Verbose logging + safe row-oriented inserts + guards against tiny-doc explosions.

Examples:
  python tools/build_stoplists_all.py \
    --root parquet/cleaned \
    --n 5 --df 0.005 --min-df-abs 2 \
    --ch-host localhost --ch-db codebase --ch-user default --ch-pass 1234 \
    --corpus-id cleaned_all_splits --version v1.0.0 \
    --create-table --replace-existing --log-level INFO --batch-size 20000

# Dry run (no CH writes), more logs:
#   ... --dry-run --log-level DEBUG --log-every 2000
"""
from __future__ import annotations

import argparse
import concurrent.futures as fut
import datetime as dt
import hashlib
import logging
import os
import re
import signal
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Tuple

ALLOWED_EXTS = {".code", ".java", ".py", ".go", ".js", ".ts", ".jsx", ".tsx"}

try:
    from clickhouse_connect import get_client
except Exception:
    raise SystemExit("clickhouse-connect is required. pip install clickhouse-connect")

# -------------------- logging --------------------
LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)s | %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, datefmt="%H:%M:%S")
log = logging.getLogger("stoplists")

# -------------------- tokenization --------------------
TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z_0-9]*|==|!=|<=|>=|&&|\|\||[{}();,\[\].:+\-*/]")


def tokenize(code: str) -> List[str]:
    return TOKEN_RE.findall(code)


def gen_shingles(tokens: List[str], n: int) -> Iterator[Tuple[str, ...]]:
    L = len(tokens)
    for i in range(0, max(0, L - n + 1)):
        yield tuple(tokens[i : i + n])


def shingle_hash(s: Tuple[str, ...]) -> str:
    return hashlib.blake2b("\u0001".join(s).encode("utf-8"), digest_size=8).hexdigest()


# -------------------- configs --------------------
@dataclass
class CHCfg:
    host: str
    user: str
    pwd: str
    db: str
    table: str


@dataclass
class LangJob:
    lang: str
    files: List[Path]
    n: int
    df: float | None
    topk: int | None
    min_df_abs: int
    ch: CHCfg
    corpus_id: str
    version: str
    create_table: bool
    replace_existing: bool
    dry_run: bool
    log_every: int
    batch_size: int


# -------------------- fs utils --------------------


def discover_languages(root: Path) -> List[str]:
    langs = [p.name for p in root.iterdir() if p.is_dir()]
    langs.sort()
    return langs


ALLOWED_EXTS = {
    "java": {".java"},
    "python": {".py"},
    "go": {".go"},
    "javascript": {".js", ".jsx", ".mjs", ".cjs", ".ts", ".tsx"},
}
IGNORE_DIRS = {
    ".git",
    ".svn",
    ".hg",
    "node_modules",
    "__pycache__",
    "venv",
    "build",
    "dist",
    "target",
}


def is_hidden(path: Path) -> bool:
    return any(p.startswith(".") for p in path.parts)


def collect_files_for_lang(root: Path, lang: str) -> list[Path]:
    lang_root = root / lang
    if not lang_root.exists():
        return []
    files: list[Path] = []
    for p in lang_root.rglob("*"):  # <-- recurse!
        if not p.is_file():
            continue
        if p.suffix.lower() not in ALLOWED_EXTS:
            continue
        try:
            if p.stat().st_size == 0:
                continue
            with p.open("rb") as fh:
                fh.read(2048).decode("utf-8")
        except Exception:
            continue
        files.append(p)
    return files


# -------------------- ClickHouse helpers --------------------


def ch_client(ch: CHCfg):
    return get_client(host=ch.host, username=ch.user, password=ch.pwd, database=ch.db)


def ch_create_table_if_needed(client, table: str):
    ddl = f"""
    CREATE TABLE IF NOT EXISTS {table} (
      lang LowCardinality(String),
      n UInt8,
      hash FixedString(16),
      shingle Array(String),
      df UInt32,
      docs UInt32,
      built_at DateTime,
      corpus_id LowCardinality(String),
      version LowCardinality(String)
    ) ENGINE = ReplacingMergeTree(built_at)
    PARTITION BY (lang, n)
    ORDER BY (lang, n, hash)
    """
    client.command(ddl)


def ch_delete_existing(client, ch: CHCfg, lang: str, n: int, corpus_id: str | None):
    if corpus_id:
        q = f"ALTER TABLE {ch.table} DELETE WHERE lang = %(lang)s AND n = %(n)s AND corpus_id = %(corpus)s"
        client.command(q, parameters={"lang": lang, "n": n, "corpus": corpus_id})
    else:
        q = f"ALTER TABLE {ch.table} DELETE WHERE lang = %(lang)s AND n = %(n)s"
        client.command(q, parameters={"lang": lang, "n": n})


# -------------------- row-oriented batch insert --------------------
COLS = [
    "lang",
    "n",
    "hash",
    "shingle",
    "df",
    "docs",
    "built_at",
    "corpus_id",
    "version",
]


def ch_insert_rows(client, ch: CHCfg, rows_iter, batch_size: int = 50_000) -> int:
    """Insert as a list-of-rows to satisfy clickhouse-connect's default expectations."""
    total = 0
    batch: List[tuple] = []

    def flush():
        nonlocal total, batch
        if not batch:
            return
        client.insert(ch.table, batch, column_names=COLS)
        total += len(batch)
        batch = []

    for row in rows_iter:
        batch.append(
            (
                row["lang"],
                int(row["n"]),
                row["hash"],
                row["shingle"],  # list[str] → Array(String)
                int(row["df"]),
                int(row["docs"]),
                row["built_at"],  # datetime
                row["corpus_id"],
                row["version"],
            )
        )
        if len(batch) >= batch_size:
            flush()
    flush()
    return total


# -------------------- signal handler --------------------
_should_stop = False


def _sigint(sig, frame):
    global _should_stop
    _should_stop = True
    log.warning("SIGINT received: finishing current file then stopping...")


signal.signal(signal.SIGINT, _sigint)

# -------------------- per-language worker --------------------


def build_and_store_for_lang(job: LangJob) -> tuple[str, int, int, int]:
    log.info(
        f"[{job.lang}] start | files={len(job.files)} n={job.n} df={job.df} topk={job.topk} dry_run={job.dry_run}"
    )
    client = ch_client(job.ch)
    if job.create_table and not job.dry_run:
        ch_create_table_if_needed(client, job.ch.table)

    df_counter = Counter()
    total_docs = 0

    for idx, p in enumerate(job.files, 1):
        if _should_stop:
            break
        try:
            code = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            if idx % job.log_every == 0:
                log.exception(f"[{job.lang}] read error at {p}")
            continue
        toks = tokenize(code)
        uniq = set(gen_shingles(toks, job.n))
        if uniq:
            df_counter.update(uniq)
            total_docs += 1
        if idx % job.log_every == 0:
            log.info(
                f"[{job.lang}] scanned {idx}/{len(job.files)} files | docs_with_shingles={total_docs} uniques={len(df_counter)}"
            )

    if total_docs == 0:
        log.warning(f"[{job.lang}] no documents with >= {job.n} tokens — skipping")
        return job.lang, 0, 0, 0

    # Choose by DF or Top-K (with absolute DF floor to avoid cutoff=1 explosions)
    if job.df is not None:
        raw_cutoff = int(total_docs * job.df)
        cutoff = max(job.min_df_abs, raw_cutoff)
        chosen = [(s, c) for s, c in df_counter.items() if c >= cutoff]
        log.info(
            f"[{job.lang}] DF cutoff={cutoff} (raw={raw_cutoff}, docs={total_docs}) → chosen={len(chosen)} from uniques={len(df_counter)}"
        )
    else:
        chosen = df_counter.most_common(job.topk or 0)
        log.info(
            f"[{job.lang}] TopK={job.topk} → chosen={len(chosen)} from uniques={len(df_counter)}"
        )

    if job.dry_run:
        log.info(
            f"[{job.lang}] dry-run: skipping DELETE/INSERT; would insert {len(chosen)} rows"
        )
        return job.lang, total_docs, len(df_counter), len(chosen)

    if job.replace_existing:
        log.info(
            f"[{job.lang}] deleting existing rows for (lang={job.lang}, n={job.n}, corpus_id={job.corpus_id})…"
        )
        ch_delete_existing(client, job.ch, job.lang, job.n, job.corpus_id)

    built_at = dt.datetime.now(
        dt.timezone.utc
    )  # TZ-aware; clickhouse-connect will convert

    def row_iter():
        for s, c in chosen:
            yield {
                "lang": job.lang,
                "n": job.n,
                "hash": shingle_hash(s),
                "shingle": list(s),
                "df": int(c),
                "docs": int(total_docs),
                "built_at": built_at,
                "corpus_id": job.corpus_id,
                "version": job.version,
            }

    inserted = ch_insert_rows(client, job.ch, row_iter(), job.batch_size)
    log.info(f"[{job.lang}] inserted rows={inserted}")
    return job.lang, total_docs, len(df_counter), inserted


# -------------------- main --------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Root with language subfolders (e.g., parquet/cleaned)",
    )
    ap.add_argument("--n", type=int, default=5, help="Shingle length")
    grp = ap.add_mutually_exclusive_group(required=True)
    grp.add_argument("--df", type=float, help="DF threshold fraction (e.g., 0.005)")
    grp.add_argument("--topk", type=int, help="Top-K shingles by DF")
    ap.add_argument(
        "--min-df-abs",
        type=int,
        default=2,
        help="Absolute DF cutoff floor (avoid singleton explosions)",
    )
    ap.add_argument(
        "--langs",
        type=str,
        nargs="*",
        help="Optional explicit languages; defaults to all subdirs",
    )
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 4)

    # ClickHouse
    ap.add_argument("--ch-host", type=str, default=os.getenv("CH_HOST", "localhost"))
    ap.add_argument("--ch-user", type=str, default=os.getenv("CH_USER", "default"))
    ap.add_argument("--ch-pass", type=str, default=os.getenv("CH_PASS", ""))
    ap.add_argument("--ch-db", type=str, default=os.getenv("CH_DB", "codebase"))
    ap.add_argument("--table", type=str, default="stoplist_shingles")
    ap.add_argument("--create-table", action="store_true")
    ap.add_argument("--replace-existing", action="store_true")

    # Metadata
    ap.add_argument("--corpus-id", type=str, default="cleaned_all_splits")
    ap.add_argument("--version", type=str, default="v1.0.0")

    # Logging & control
    ap.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )
    ap.add_argument(
        "--log-every",
        type=int,
        default=5000,
        help="Log progress every N files per language",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute DF and show stats; skip CH DELETE/INSERT",
    )
    ap.add_argument(
        "--batch-size",
        type=int,
        default=20000,
        help="ClickHouse insert batch size (rows)",
    )

    args = ap.parse_args()
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    root = args.root
    langs = args.langs or discover_languages(root)
    if not langs:
        log.error(f"No languages found under {root}")
        sys.exit(2)

    log.info(f"Languages discovered: {', '.join(langs)}")
    ch = CHCfg(
        host=args.ch_host,
        user=args.ch_user,
        pwd=args.ch_pass,
        db=args.ch_db,
        table=args.table,
    )

    # Prime connection and optionally create table
    try:
        cli = ch_client(ch)
        if args.create_table and not args.dry_run:
            ch_create_table_if_needed(cli, ch.table)
            log.info(f"Ensured table exists: {ch.table}")
    except Exception:
        log.exception("ClickHouse connection failed")
        sys.exit(2)

    jobs: List[LangJob] = []
    for lang in langs:
        files = collect_files_for_lang(root, lang)
        if not files:
            log.warning(f"[{lang}] no files under {root/lang}")
        jobs.append(
            LangJob(
                lang=lang,
                files=files,
                n=args.n,
                df=args.df,
                topk=args.topk,
                min_df_abs=args.min_df_abs,
                ch=ch,
                corpus_id=args.corpus_id,
                version=args.version,
                create_table=args.create_table,
                replace_existing=args.replace_existing,
                dry_run=args.dry_run,
                log_every=args.log_every,
                batch_size=args.batch_size,
            )
        )

    results = []
    with fut.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(build_and_store_for_lang, j) for j in jobs]
        for fu in fut.as_completed(futs):
            try:
                results.append(fu.result())
            except Exception:
                log.exception("worker crashed")

    # Summary table
    log.info("\n===== SUMMARY =====")
    for lang, docs, uniques, inserted in sorted(results):
        log.info(
            f"{lang:>8} | docs={docs:7d} | uniques={uniques:8d} | inserted={inserted:8d}"
        )
    log.info("Done.")


if __name__ == "__main__":
    main()
