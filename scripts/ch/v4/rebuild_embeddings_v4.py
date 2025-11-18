#!/usr/bin/env python3
import os
import sys
import time

# Додаємо src у PYTHONPATH вручну (щоб скрипт працював незалежно від запуску)
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from clickhouse_connect import get_client

ROOT = Path(__file__).resolve().parents[3]  # repo root
sys.path.insert(0, str(ROOT / "src"))

from app.api.deps import get_embed_model
from app.domain.boilerplate_filter import BoilerplateFilter

# ─────────────────────────────
# ClickHouse конфіг
# ─────────────────────────────
CH_HOST = os.getenv("CH_HOST", "localhost")
CH_USER = os.getenv("CH_USER", "default")
CH_PASS = os.getenv("CH_PASS", "1234")
CH_DB = os.getenv("CH_DB", "codebase")

BATCH = int(os.getenv("MIGRATE_BATCH", "50000"))

CODES_TABLE = "codebase.codes_v4"
EMB_TABLE = "codebase.embeddings_v4"

EMB_MODEL = os.getenv("CH_EMB_MODEL", "microsoft/codebert-base")
EMB_POOL = os.getenv("CH_EMB_POOL", "mean")
EMB_TVER = int(os.getenv("CH_EMB_TRANSFORM_VER", "3"))


# ─────────────────────────────
# Helpers
# ─────────────────────────────
def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def get_ch_client():
    log(f"Connecting to ClickHouse at {CH_HOST}...")
    client = get_client(
        host=CH_HOST, username=CH_USER, password=CH_PASS, database=CH_DB
    )
    log("Connected to ClickHouse.")
    return client


def fetch_codes_batch(client, last_code_id: Optional[str], limit: int):
    """
    SELECT code_id, lang, split, code
    """
    params = {"limit": int(limit)}
    where_parts = ["code IS NOT NULL", "length(code) > 0"]

    if last_code_id is not None:
        where_parts.append("code_id > {last:String}")
        params["last"] = last_code_id

    where_sql = " AND ".join(where_parts)

    q = f"""
        SELECT code_id, lang, split, code
        FROM {CODES_TABLE}
        WHERE {where_sql}
        ORDER BY code_id
        LIMIT {{limit:UInt32}}
    """

    t0 = time.time()
    rows = client.query(q, parameters=params).result_rows
    log(f"Fetched {len(rows)} rows in {time.time() - t0:.2f}s.")
    return rows


def vec_to_list(v):
    arr = np.asarray(v, dtype=np.float32)
    arr = arr.reshape(-1)
    return arr.tolist(), int(arr.shape[0])


def insert_embeddings_batch(client, rows):
    if not rows:
        return

    t0 = time.time()
    client.insert(
        EMB_TABLE,
        rows,
        column_names=[
            "code_id",
            "lang",
            "split",
            "model",
            "pooling",
            "transform_ver",
            "dim",
            "vector",
        ],
    )
    log(f"Inserted {len(rows)} embeddings in {time.time() - t0:.2f}s.")


# ─────────────────────────────
# Основний процес
# ─────────────────────────────
def main():
    log("Starting embeddings rebuild (v4)...")
    client = get_ch_client()

    log("Initializing boilerplate filter...")
    boiler = BoilerplateFilter()

    log("Loading embedding model...")
    model = get_embed_model()
    log("Embedding model loaded.")

    total_processed = 0
    last_code_id: Optional[str] = None

    t_start = time.time()

    while True:
        log(f"Fetching batch after last_code_id={last_code_id} (limit={BATCH})...")
        codes = fetch_codes_batch(client, last_code_id, BATCH)
        if not codes:
            log("No more rows. FINISHED.")
            break

        # Optional debug: top 5 longest codes
        top_long = sorted(codes, key=lambda x: len(x[3] or ""), reverse=True)[:5]
        log("Top 5 longest code snippets in batch:")
        for i, row in enumerate(top_long):
            log(f"  {i+1}) {row[0]} — {len(row[3])} chars")

        batch_rows = []
        t_batch_embed = 0.0
        t_batch_filter = 0.0

        for code_id, lang, split, code in codes:

            if not code:
                log(f"WARNING: code_id {code_id} has empty code. Skipping.")
                continue

            # ── 1) Boilerplate filter ─────────────────────
            t0 = time.time()
            filtered = boiler.filter_for_embedding(code, lang)
            t_batch_filter += time.time() - t0

            if not filtered.strip():
                log(
                    f"WARNING: code_id {code_id} becomes EMPTY after boilerplate filtering."
                )
                continue

            # ── 2) Embedding ─────────────────────────────
            t0 = time.time()
            vec = model.encode(filtered)
            t_batch_embed += time.time() - t0

            vec_list, dim = vec_to_list(vec)

            batch_rows.append(
                (
                    code_id,
                    lang or "",
                    split or "",
                    EMB_MODEL,
                    EMB_POOL,
                    EMB_TVER,
                    dim,
                    vec_list,
                )
            )

        log(f"Filtering time for batch: {t_batch_filter:.2f}s.")
        log(f"Embedding time for batch: {t_batch_embed:.2f}s.")

        # ── 3) Insert batch ─────────────────────────────
        insert_embeddings_batch(client, batch_rows)

        total_processed += len(batch_rows)
        last_code_id = codes[-1][0]

        log(f"Batch DONE: inserted={len(batch_rows)}, total={total_processed}.")

    # ─────────────────────────────
    t_total = time.time() - t_start
    log(f"ALL DONE. Total processed embeddings: {total_processed}.")
    log(f"Total time: {t_total/60:.1f} min ({t_total:.1f}s).")


if __name__ == "__main__":
    main()
