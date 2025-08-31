DESCRIBE SELECT emb FROM embeddings LIMIT 1;
-- if you see LIST or FLOAT[], we need to migrate to FLOAT[768]
-- add a new fixed-size column
ALTER TABLE embeddings ADD COLUMN emb_fixed FLOAT[768];

-- cast existing vectors into the fixed-size array
UPDATE embeddings SET emb_fixed = emb::FLOAT[768];

-- (optional sanity) all rows should have non-NULL emb_fixed
SELECT COUNT(*) AS bad_rows
FROM embeddings
WHERE emb_fixed IS NULL;

-- swap columns
ALTER TABLE embeddings DROP COLUMN emb;
ALTER TABLE embeddings RENAME COLUMN emb_fixed TO emb;
INSTALL vss;
LOAD vss;

-- optional: persist HNSW to disk (still experimental)
SET hnsw_enable_experimental_persistence = true;

-- build the index
CREATE INDEX IF NOT EXISTS embeddings_hnsw
ON embeddings USING HNSW (emb)
WITH (metric='cosine', ef_construction=128, M=16, ef_search=64);


EXPLAIN
SELECT uid
FROM embeddings
ORDER BY array_cosine_distance(emb, ?::FLOAT[768])
LIMIT 10;
-- look for HNSW_INDEX_SCAN in the plan
