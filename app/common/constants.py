PLAG_DB_PATH = "duckdb/plag.db"
APP_DB_PATH = "duckdb/app.db"

# .env (or your runtime env)
BACKEND_STORAGE = "clickhouse"
CH_USE_V3 = 1  # 1 = use codes_v2 / embeddings_v2
CH_EMB_MODEL = "microsoft/codebert-base"
CH_EMB_POOL = "mean"
CH_EMB_TRANSFORM_VER = 2
CH_EMB_DIM = 768
