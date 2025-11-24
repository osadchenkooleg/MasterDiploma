from __future__ import annotations

from app.infrastructure.db.clickhouse.codes_repo import CodesRepoCH
from app.infrastructure.db.clickhouse.embeddings_repo import EmbeddingsRepoCH

# ClickHouse DI providers (use these in your FastAPI routers when targeting CH)
from app.infrastructure.db.clickhouse.languages_repo import LanguagesRepoCH
from app.infrastructure.db.clickhouse.thresholds_repo import ThresholdPolicyRepository

_lang_repo_ch = LanguagesRepoCH()
_codes_repo_ch = CodesRepoCH()
_emb_repo_ch = EmbeddingsRepoCH()
_trs_policy_repo_ch = ThresholdPolicyRepository()


def get_languages_repo_ch():
    return _lang_repo_ch


def get_codes_repo_ch():
    return _codes_repo_ch


def get_embeddings_repo_ch():
    return _emb_repo_ch


def get_threshold_repo_ch():
    return _trs_policy_repo_ch
