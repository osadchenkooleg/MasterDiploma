-- База
CREATE DATABASE IF NOT EXISTS codebase;
USE codebase;

-- Легка нормалізація (SQL UDF)
CREATE FUNCTION IF NOT EXISTS normalize_code_light AS (s) ->
    replaceRegexpAll(
      replaceRegexpAll(
        replaceRegexpAll(
          replaceRegexpAll(s, '(?s)/\\*.*?\\*/', ''),  -- /* ... */
          '(?m)//.*$', ''),                           -- // ...
        '(?m)#.*$', ''),                              -- # ...
      '[\\t ]+', ' '                                 -- множинні пробіли/таби
    );

-- Головна таблиця з кодом
CREATE TABLE IF NOT EXISTS practice_codes
(
  lang            LowCardinality(String),
  split           LowCardinality(String),     -- 'train' | 'validation' | 'test' | ...
  uid             String,
  code            String,

  -- Похідні
  code_len        UInt32          MATERIALIZED lengthUTF8(code),
  code_norm       String          MATERIALIZED normalize_code_light(code),
  code_norm_md5   FixedString(16) MATERIALIZED MD5(code_norm)
)
ENGINE = MergeTree
ORDER BY (lang, split, uid);

-- Пари для валідації/калібрування
CREATE TABLE IF NOT EXISTS practice_pairs
(
  pair_id   UUID DEFAULT generateUUIDv4(),
  split     LowCardinality(String) DEFAULT 'valid',
  a_lang    LowCardinality(String),
  a_uid     String,
  b_lang    LowCardinality(String),
  b_uid     String,
  label     UInt8,       -- 1 = схожі, 0 = різні
  notes     String DEFAULT ''
)
ENGINE = MergeTree
ORDER BY (split, a_lang, b_lang, a_uid, b_uid);

-- Результати порівняння (протокол експериментів)
CREATE TABLE IF NOT EXISTS practice_scores
(
  run_id        UUID DEFAULT generateUUIDv4(),
  ts            DateTime DEFAULT now(),
  model         LowCardinality(String),
  normalization LowCardinality(String) DEFAULT 'light',  -- 'none' | 'light'
  boilerplate   LowCardinality(String) DEFAULT 'off',    -- 'off' | 'stoplist@v1'
  metric        LowCardinality(String) DEFAULT 'cosine',
  threshold     Float32 DEFAULT 0.80,

  pair_id       UUID,
  a_lang        LowCardinality(String),
  a_uid         String,
  b_lang        LowCardinality(String),
  b_uid         String,
  label         UInt8,

  score         Float32,
  decision      LowCardinality(String)                   -- 'OK' | 'Review' | 'Plagiarism'
)
ENGINE = MergeTree
ORDER BY (run_id, pair_id);

-- Патерни «бойлерплейту» (опційно)
CREATE TABLE IF NOT EXISTS practice_boilerplate_patterns
(
  pattern_id  UUID DEFAULT generateUUIDv4(),
  lang        LowCardinality(String),
  kind        LowCardinality(String),   -- 'line' | 'shingle5' | ...
  text        String,
  weight      Float32 DEFAULT 1.0
)
ENGINE = MergeTree
ORDER BY (lang, kind, pattern_id);

-- Зберігаємо вектори ембеддингів
CREATE TABLE IF NOT EXISTS codebase.practice_embeddings
(
  lang          LowCardinality(String),
  uid           String,
  model         LowCardinality(String),   -- напр., 'microsoft/codebert-base'
  pool          LowCardinality(String),   -- 'mean'
  transform_ver UInt16 DEFAULT 1,
  dim           UInt16,
  vec           Array(Float32),
  created_at    DateTime DEFAULT now()
)
ENGINE = MergeTree
ORDER BY (lang, uid, model, pool, transform_ver);

-- Для обраних порогів і агрегованих метрик пробігу
CREATE TABLE IF NOT EXISTS codebase.practice_thresholds
(
  run_id        UUID,
  ts            DateTime DEFAULT now(),
  model         LowCardinality(String),
  pool          LowCardinality(String),
  transform_ver UInt16,
  split         LowCardinality(String) DEFAULT 'valid',
  metric        LowCardinality(String) DEFAULT 'cosine',
  threshold     Float32,
  roc_auc       Float32,
  pr_auc        Float32,
  precision_at_t Float32,
  recall_at_t    Float32,
  f1_at_t        Float32
)
ENGINE = MergeTree
ORDER BY (run_id, ts);

CREATE TABLE codebase.eval_pairs
(
  id_a   String,
  id_b   String,
  label  Int8,          -- 1 = позитив, 0 = негатив
  path_a String,
  path_b String,
  lang   LowCardinality(String) DEFAULT '',   -- НЕ nullable
  split  LowCardinality(String) DEFAULT 'test'
)
ENGINE = MergeTree
ORDER BY (split, lang, id_a, id_b);
