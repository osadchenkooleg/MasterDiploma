CREATE DATABASE IF NOT EXISTS codebase;

-- Supported languages (admin-editable)
CREATE TABLE IF NOT EXISTS codebase.languages
(
  lang LowCardinality(String),
  enabled UInt8
)
ENGINE = MergeTree
ORDER BY lang;

-- Codes: legacy + new (code may be NULL for legacy rows that lack source text)
CREATE TABLE IF NOT EXISTS codebase.codes
(
  id String,                                   -- your existing uid
  lang LowCardinality(String),
  split LowCardinality(String) DEFAULT 'unknown',
  label LowCardinality(String) DEFAULT '',
  code Nullable(String) CODEC(ZSTD),           -- compress large text
  created_at DateTime DEFAULT now()
)
ENGINE = MergeTree
PARTITION BY lang
ORDER BY (lang, id);

-- Embeddings: L2-normalized vectors (dim usually 768)
CREATE TABLE IF NOT EXISTS codebase.embeddings
(
  id String,                                   -- same uid as in codes
  lang LowCardinality(String),
  split LowCardinality(String),
  dim UInt16 DEFAULT 768,
  vector Array(Float32),
  CONSTRAINT vec_dim CHECK length(vector) = dim
)
ENGINE = MergeTree
PARTITION BY lang
ORDER BY (lang, id);
