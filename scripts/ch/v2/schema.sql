CREATE DATABASE IF NOT EXISTS codebase;

/* ---------- codes_v2 ---------- */
/* Fresh code table reloaded from Parquet. */
CREATE TABLE IF NOT EXISTS codebase.codes_v2
(
  /* Keep your existing logical key */
  code_id String,                                   -- your uid
  lang LowCardinality(String),
  split LowCardinality(String) DEFAULT 'unknown',
  label LowCardinality(String) DEFAULT '',
  code Nullable(String) CODEC(ZSTD),
  code_hash Nullable(String),                       -- hex(SHA256(code)) if code present
  source LowCardinality(String) DEFAULT 'parquet',  -- 'parquet' | 'api' | ...
  ingested_at DateTime DEFAULT now()
)
ENGINE = ReplacingMergeTree(ingested_at)
PARTITION BY lang
ORDER BY (lang, code_id);

/* Helpful projections (optional, can add later) */
/*
ALTER TABLE codebase.codes_v2
  ADD PROJECTION p_lang_split
  (
    SELECT code_id, split, label, code, code_hash
    ORDER BY code_id
  );
*/

/* ---------- embeddings_v2 ---------- */
/* Embeddings table with its own ID and a FK-style link to code_id. */
CREATE TABLE IF NOT EXISTS codebase.embeddings_v2
(
  embedding_id String,                               -- ULID/UUID per embedding row
  code_id String,                                    -- links to codes_v2.code_id
  lang LowCardinality(String),
  split LowCardinality(String),
  model LowCardinality(String) DEFAULT 'microsoft/codebert-base',
  pooling LowCardinality(String) DEFAULT 'mean',     -- 'mean' | 'cls' | ...
  transform_ver UInt16 DEFAULT 1,                    -- 1=raw mean; 2=mean-centered; etc.
  dim UInt16 DEFAULT 768,
  vector Array(Float32),
  created_at DateTime DEFAULT now()
)
ENGINE = MergeTree
PARTITION BY lang
ORDER BY (lang, code_id, embedding_id);

/* (Optional, later) ANN index when you’re ready
ALTER TABLE codebase.embeddings_v2
  ADD INDEX idx_vss vector TYPE vector_similarity('hnsw', cosineDistance, 768);
-- After loading embeddings:
-- ALTER TABLE codebase.embeddings_v2 MATERIALIZE INDEX idx_vss SETTINGS mutations_sync = 2;
*/
