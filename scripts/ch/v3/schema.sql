-- codes_v4: ULID-based primary id (stored as String), still stores old id and lang (+ all prior columns)
CREATE TABLE IF NOT EXISTS codebase.codes_v4
(
  code_id    String,                        -- NEW: ULID (String)
  old_id     String,                        -- legacy codes_v3.id
  lang       LowCardinality(String),
  split      String,
  label      String,
  code       Nullable(String),
  code_hash  Nullable(String),
  source     String
)
ENGINE = MergeTree
ORDER BY (code_id)
SETTINGS index_granularity = 8192;

-- embeddings_v3: references codes_v4 by code_id
CREATE TABLE IF NOT EXISTS codebase.embeddings_v3
(
  embedding_id String DEFAULT generateUUIDv4(),   -- keep per-embedding id; use UUID (or fill with ULID from app)
  code_id      String,                            -- FK-ish to codes_v4.code_id (no FK enforced in CH)
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
SETTINGS index_granularity = 8192;
