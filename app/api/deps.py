# app/api/deps.py
import os
from functools import lru_cache

from app.domain.boilerplate_filter import BoilerplateFilter
from app.infrastructure.embeddings.model_codebert import CodeEmbeddingModel

EMB_MODEL = os.getenv("EMB_MODEL", "microsoft/codebert-base")
EMB_POOL = os.getenv("EMB_POOL", "mean")
EMB_TRANSFORM_VER = int(os.getenv("EMB_TRANSFORM_VER", "2"))  # <-- default 2


@lru_cache(maxsize=1)
def get_embed_model() -> CodeEmbeddingModel:
    m = CodeEmbeddingModel(model_name=EMB_MODEL)
    # attach metadata so routers/repos can read it
    m.model_name = EMB_MODEL
    m.pooling = EMB_POOL
    m.transform_ver = EMB_TRANSFORM_VER
    return m


@lru_cache(maxsize=1)
def get_boilerplate_filter() -> BoilerplateFilter:
    return BoilerplateFilter()
