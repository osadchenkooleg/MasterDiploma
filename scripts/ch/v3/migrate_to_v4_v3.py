#!/usr/bin/env python3
import os

import ulid
from clickhouse_connect import get_client

CH_HOST = os.getenv("CH_HOST", "localhost")
CH_USER = os.getenv("CH_USER", "default")
CH_PASS = os.getenv("CH_PASS", "1234")
CH_DB = os.getenv("CH_DB", "codebase")

BATCH = int(os.getenv("MIGRATE_BATCH", "50000"))


def scalar(client, sql: str) -> int:
    rows = client.query(sql).result_rows
    return int(rows[0][0]) if rows and rows[0] else 0


def main():
    client = get_client(
        host=CH_HOST, username=CH_USER, password=CH_PASS, database=CH_DB
    )

    # Ensure (re)created tables exist
    client.command(
        """
    CREATE TABLE IF NOT EXISTS codebase.codes_v4
    (
      code_id    String,
      old_id     String,
      lang       LowCardinality(String),
      split      String,
      label      String,
      code       Nullable(String),
      code_hash  Nullable(String),
      source     String
    )
    ENGINE = MergeTree
    ORDER BY (code_id)
    """
    )
    client.command(
        """
    CREATE TABLE IF NOT EXISTS codebase.embeddings_v3
    (
      embedding_id String DEFAULT generateUUIDv4(),
      code_id      String,
      lang         LowCardinality(String),
      split        LowCardinality(String),
      model        LowCardinality(String) DEFAULT 'microsoft/codebert-base',
      pooling      LowCardinality(String) DEFAULT 'mean',
      transform_ver UInt16 DEFAULT 1,
      dim          UInt16 DEFAULT 768,
      vector       Array(Float32),
      created_at   DateTime DEFAULT now()
    )
    ENGINE = MergeTree
    ORDER BY (code_id, created_at)
    """
    )
    client.command(
        """
    CREATE TABLE IF NOT EXISTS codebase.code_id_map
    (
      lang        LowCardinality(String),
      old_code_id String,
      code_id     String
    )
    ENGINE = MergeTree
    ORDER BY (lang, old_code_id)
    """
    )

    total = scalar(client, "SELECT count() FROM codebase.codes_v3")
    print(f"codes_v3 rows: {total}")

    offset = 0
    cols_v4 = [
        "code_id",
        "old_id",
        "lang",
        "split",
        "label",
        "code",
        "code_hash",
        "source",
    ]
    cols_map = ["lang", "old_code_id", "code_id"]

    # Keep one ULID per (lang, old_code_id)
    seen: dict[tuple[str, str], str] = {}

    while offset < total:
        rows = client.query(
            f"""
            SELECT
              id,            -- old_id (usually 'lang/96401_1')
              code_id,       -- old_code_id ('96401_1')
              lang,
              split,
              label,
              code,
              code_hash,
              source
            FROM codebase.codes_v3
            ORDER BY id
            LIMIT {BATCH} OFFSET {offset}
        """
        ).result_rows
        if not rows:
            break

        v4_rows = []
        map_rows = []

        for old_id, old_code_id, lang, split, label, code, code_hash, source in rows:
            key = (lang or "", old_code_id or "")
            if key not in seen:
                new_ulid = str(ulid.ULID())
                seen[key] = new_ulid

                # Insert one codes_v4 row per unique (lang, old_code_id)
                v4_rows.append(
                    (
                        new_ulid,
                        old_id or "",  # keep legacy composite id for compatibility
                        lang or "",
                        split or "",
                        label or "",
                        code,  # Nullable
                        code_hash,  # Nullable
                        (source or ""),
                    )
                )
                # Map lang+old_code_id -> new ULID
                map_rows.append((lang or "", old_code_id or "", new_ulid))
            # else: duplicate (lang,old_code_id) row in v3 → skip creating a second v4 row

        if v4_rows:
            client.insert("codebase.codes_v4", v4_rows, column_names=cols_v4)
        if map_rows:
            client.insert("codebase.code_id_map", map_rows, column_names=cols_map)

        offset += len(rows)
        print(
            f"processed codes_v3: {offset}/{total}  (distinct pairs so far: {len(seen)})"
        )

    # Now migrate embeddings_v2 → embeddings_v3 using BOTH lang and old_code_id
    client.command(
        """
        INSERT INTO codebase.embeddings_v3
        SELECT
          generateUUIDv4() AS embedding_id,
          m.code_id        AS code_id,
          e.lang           AS lang,
          e.split          AS split,
          e.model          AS model,
          e.pooling        AS pooling,
          e.transform_ver  AS transform_ver,
          e.dim            AS dim,
          e.vector         AS vector,
          e.created_at     AS created_at
        FROM codebase.embeddings_v2 e
        INNER JOIN codebase.code_id_map m
          ON e.lang = m.lang AND e.code_id = m.old_code_id
        WHERE NOT EXISTS (
          SELECT 1
          FROM codebase.embeddings_v3 v3
          WHERE v3.code_id = m.code_id
            AND v3.created_at = e.created_at
        )
    """
    )

    new_codes = scalar(client, "SELECT count() FROM codebase.codes_v4")
    new_maps = scalar(client, "SELECT count() FROM codebase.code_id_map")
    new_embs = scalar(client, "SELECT count() FROM codebase.embeddings_v3")
    print(
        f"Done. ✅ codes_v4: {new_codes}, code_id_map: {new_maps}, embeddings_v3: {new_embs}"
    )


if __name__ == "__main__":
    main()
