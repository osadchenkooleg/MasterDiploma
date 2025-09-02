from pathlib import Path

import duckdb

DB_PATH = "duckdb/plag.db"
Path("duckdb").mkdir(exist_ok=True)

con = duckdb.connect(DB_PATH)  # file-backed, or ":memory:"
con.execute("INSTALL vss; LOAD vss;")
con.execute("SET hnsw_enable_experimental_persistence = true;")  # optional, see caveats

# Ensure correct type (only if needed)
# con.execute("""
#   ALTER TABLE embeddings
#   ALTER COLUMN embedding SET DATA TYPE FLOAT[768]
#   USING embedding::FLOAT[768];
# """)

con.execute(
    """
  CREATE INDEX IF NOT EXISTS ix_embeddings_vec_cos
  ON embeddings USING HNSW (embedding)
  WITH (metric='cosine', ef_construction=200, M=16);
"""
)

# Query
query_vec = [-0.017709557] * 768  # your real vector here
con.execute(
    """
  SELECT uid, lang, split,
         array_cosine_distance(embedding, ?::FLOAT[768]) AS dist
  FROM embeddings
  ORDER BY dist
  LIMIT 20;
""",
    [query_vec],
).fetchdf()
