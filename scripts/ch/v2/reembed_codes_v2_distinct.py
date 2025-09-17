#!/usr/bin/env python
import os
import sys
from pathlib import Path
from typing import Iterable, Optional

import ulid
from clickhouse_connect import get_client

# Add project root to sys.path (2 levels up from scripts/ch/v2/)
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# from app.infra.embeddings.model_codebert import CodeEmbeddingModel
from app.infrastructure.embeddings.model_codebert import (  # <- if your folder is "infrastructure"
    CodeEmbeddingModel,
)

CH_HOST = os.getenv("CH_HOST", "localhost")
CH_USER = os.getenv("CH_USER", "default")
CH_PASS = os.getenv("CH_PASS", "1234")
CH_DB = os.getenv("CH_DB", "codebase")

MODEL = os.getenv("EMB_MODEL", "microsoft/codebert-base")
POOL = os.getenv("EMB_POOL", "mean")
TRANSFORM_VER = int(os.getenv("EMB_TRANSFORM_VER", "2"))
DIM = int(os.getenv("EMB_DIM", "768"))

BATCH_INSERT = int(os.getenv("BATCH_INSERT", "1000"))
FETCH_LIMIT = int(os.getenv("FETCH_LIMIT", "50000"))  # per language pull size


def fetch_rows(client, lang: Optional[str]) -> Iterable[tuple]:
    source = "codebase.codes_v2_distinct"
    check = client.query(f"EXISTS TABLE {source}").first_item
    if not check:
        source = "codebase.codes_v2"

    lang_filter = "" if not lang else "AND c.lang = %(lang)s"
    sql = f"""
    SELECT c.code_id, c.lang, c.split, c.code
    FROM {source} AS c
    LEFT JOIN codebase.embeddings_v2 AS e
      ON e.code_id = c.code_id
     AND e.lang = c.lang
     AND e.model = %(model)s
     AND e.pooling = %(pool)s
     AND e.transform_ver = %(tver)s
    WHERE c.code IS NOT NULL
      AND e.code_id = ''         -- <- default for String when no match
      {lang_filter}
    ORDER BY c.lang, c.code_id
    LIMIT %(limit)s
    """
    params = {"model": MODEL, "pool": POOL, "tver": TRANSFORM_VER, "limit": FETCH_LIMIT}
    if lang:
        params["lang"] = lang

    print(
        f"[debug] params: model={MODEL}, pool={POOL}, tver={TRANSFORM_VER}, lang={lang}"
    )
    print("[debug] Fetching candidates…")
    rows = client.query(sql, parameters=params).result_rows
    print(f"[debug] got {len(rows)} candidates")
    for row in rows:
        yield row


def main():
    lang = sys.argv[1] if len(sys.argv) > 1 else None
    client = get_client(
        host=CH_HOST, username=CH_USER, password=CH_PASS, database=CH_DB
    )
    model = CodeEmbeddingModel(MODEL)
    # ---------
    print(f"[init] Using device: {model.device}")
    from clickhouse_connect import common

    client.ping()
    src_exists = client.query("EXISTS TABLE codebase.codes_v2_distinct").first_item
    print(f"[init] source={'codes_v2_distinct' if src_exists else 'codes_v2'}")
    row = (
        client.query(
            """
      SELECT
        (SELECT count() FROM codebase.codes_v2 WHERE lang='java' AND code IS NOT NULL) AS v2_java,
        (SELECT count() FROM codebase.codes_v2_distinct WHERE lang='java' AND code IS NOT NULL) AS v2d_java
    """
        ).first_row
        if src_exists
        else client.query(
            """
      SELECT
        (SELECT count() FROM codebase.codes_v2 WHERE lang='java' AND code IS NOT NULL) AS v2_java
    """
        ).first_row
    )
    print(f"[init] counts: {row}")

    # Show existing embedding versions
    rows = client.query(
        """
      SELECT model, pooling, transform_ver, count() AS n
      FROM codebase.embeddings_v2
      WHERE lang='java'
      GROUP BY model, pooling, transform_ver
      ORDER BY n DESC
    """
    ).result_rows
    print(f"[init] existing versions: {rows}")
    # ---------
    # Print device info
    print(f"[init] Using device: {model.device}")

    total_inserted = 0
    while True:
        buffer = []
        fetched = 0
        for code_id, lang_val, split, code in fetch_rows(client, lang):
            fetched += 1
            vec = model.encode(code).tolist()
            emb_id = str(ulid.ULID())
            buffer.append(
                (emb_id, code_id, lang_val, split, MODEL, POOL, TRANSFORM_VER, DIM, vec)
            )

            if fetched % 100 == 0:
                print(f"[debug] processed {fetched} rows (lang={lang_val})")

            if len(buffer) >= BATCH_INSERT:
                client.insert(
                    "codebase.embeddings_v2",
                    buffer,
                    column_names=[
                        "embedding_id",
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
                total_inserted += len(buffer)
                print(
                    f"[debug] inserted batch of {len(buffer)} rows, total_inserted={total_inserted}"
                )
                buffer.clear()

        if buffer:
            client.insert(
                "codebase.embeddings_v2",
                buffer,
                column_names=[
                    "embedding_id",
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
            total_inserted += len(buffer)
            buffer.clear()

        print(f"[pass] fetched={fetched}, inserted_total={total_inserted}")
        if fetched == 0:
            break

    print(f"[done] total_inserted={total_inserted} (lang={lang or 'ALL'})")


if __name__ == "__main__":
    main()
