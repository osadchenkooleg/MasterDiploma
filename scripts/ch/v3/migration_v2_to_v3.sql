-- Mapping old -> new ids to do a fast JOIN when migrating embeddings
CREATE TABLE IF NOT EXISTS codebase.code_id_map
(
  old_code_id String,    -- codes_v3.code_id
  code_id     String     -- new ULID for codes_v4
)
ENGINE = MergeTree
ORDER BY (old_code_id);
