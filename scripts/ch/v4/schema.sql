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
SETTINGS index_granularity = 8192;

CREATE TABLE IF NOT EXISTS codebase.embeddings_v4
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
SETTINGS index_granularity = 8192;

CREATE TABLE IF NOT EXISTS codebase.stoplist_shingles
(
  lang    LowCardinality(String),
  shingle String,
  df      UInt64
)
ENGINE = MergeTree
ORDER BY (lang, df, shingle)
SETTINGS index_granularity = 8192;

CREATE TABLE IF NOT EXISTS codebase.threshold_policies
(
    policy_id UUID DEFAULT generateUUIDv4(),
    model LowCardinality(String) DEFAULT 'microsoft/codebert-base',
    transform_ver UInt16 DEFAULT 3,
    lang LowCardinality(String),
    metric LowCardinality(String) DEFAULT 'cosine_similarity',
    t_low  Float32,
    t_high Float32,
    target_precision_high_zone Nullable(Float32),
    target_recall_high_zone   Nullable(Float32),
    f1_at_t_high              Nullable(Float32),
    policy_comment Nullable(String),
    created_at DateTime DEFAULT now(),
    valid_from DateTime DEFAULT now(),
    valid_to   Nullable(DateTime),
    is_active  UInt8 DEFAULT 1
)
ENGINE = MergeTree
ORDER BY (model, transform_ver, lang, valid_from)
SETTINGS index_granularity = 8192;


