# scripts/duckdb/init_duckdb_schema.py
#!/usr/bin/env python
import os
from pathlib import Path

import duckdb

DB_PATH = "duckdb/plag.db"
Path("duckdb").mkdir(exist_ok=True)

# cache extensions in-repo (optional but nice)
os.environ.setdefault("DUCKDB_EXTENSION_DIRECTORY", ".duckdb_extensions")

con = duckdb.connect(DB_PATH)
con.execute("PRAGMA threads=8;")
con.execute("PRAGMA enable_object_cache=true;")


def try_install_load(ext: str):
    try:
        con.execute(f"INSTALL '{ext}';")
    except Exception as e:
        print(f"[warn] INSTALL '{ext}': {e}")
    try:
        con.execute(f"LOAD '{ext}';")
    except Exception as e:
        print(f"[warn] LOAD '{ext}': {e}")


# We only need VSS. (No uuid needed — we’ll use TEXT ids from Python when inserting.)
try_install_load("vss")

# Core tables
con.execute(
    """
CREATE TABLE IF NOT EXISTS embeddings (
  uid TEXT PRIMARY KEY,              -- "<id>_<side>"
  lang TEXT NOT NULL,                -- 'java'|'python'|'go'|'javascript'|...
  split TEXT NOT NULL,               -- 'train'|'validation'|'test'
  label INTEGER,                     -- -1 or NULL for single-datasets
  model TEXT NOT NULL,               -- 'microsoft/codebert-base'
  dim SMALLINT NOT NULL DEFAULT 768, -- embedding dim
  emb FLOAT[] NOT NULL,              -- L2-normalized vector
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS embed_batches (
  batch_id TEXT,                     -- fill from Python (uuid4 string), no extension needed
  model TEXT, lang TEXT, split TEXT,
  rows_ingested INTEGER, source_path TEXT,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""
)


# Is VSS really available? (don’t assume duckdb_extensions() schema)
def vss_available() -> bool:
    try:
        row = con.execute(
            "SELECT 1 FROM duckdb_functions() WHERE function_name='vss_search' LIMIT 1"
        ).fetchone()
        return row is not None
    except Exception:
        return False


if vss_available():
    con.execute(
        """
        CREATE INDEX IF NOT EXISTS embeddings_vss
        ON embeddings USING vss(emb)
        WITH (metric='cosine');
    """
    )
    print("VSS index created/exists ✓")
else:
    print(
        "[info] VSS not available; skipped index creation. "
        "Once online, run: INSTALL 'vss'; LOAD 'vss'; "
        "then: CREATE INDEX embeddings_vss ON embeddings USING vss(emb) WITH (metric='cosine');"
    )

# Parquet view for code (zstd files)
con.execute(
    """
CREATE OR REPLACE VIEW code_all AS
SELECT * FROM read_parquet('parquet/cleaned/*/*/*.parquet');
"""
)

print("DuckDB schema ready →", DB_PATH)
