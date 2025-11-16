# app/infra/clickhouse/repos/embeddings_repo.py
from __future__ import annotations

import os
from typing import Iterable, List, Optional, Tuple

from app.infrastructure.db.clickhouse.client import get_ch_client

EMB_DIM = int(os.getenv("CH_EMB_DIM", "768"))
EMB_MODEL = os.getenv("CH_EMB_MODEL", "microsoft/codebert-base")
EMB_POOL = os.getenv("CH_EMB_POOL", "mean")
EMB_TVER = int(os.getenv("CH_EMB_TRANSFORM_VER", "3"))  # <-- default 3

EMB_TABLE = "codebase.embeddings_v3"


class EmbeddingsRepoCH:
    def insert(
        self,
        code_id: str,
        vec: List[float],
        *,
        lang: str = "",
        split: str = "",
        model: str = EMB_MODEL,
        pooling: str = EMB_POOL,
        transform_ver: int = EMB_TVER,
        dim: int = EMB_DIM,
    ) -> None:
        client = get_ch_client()
        client.insert(
            EMB_TABLE,
            [
                (
                    code_id,
                    lang,
                    split,
                    model,
                    pooling,
                    int(transform_ver),
                    int(dim),
                    list(map(float, vec)),
                )
            ],
            column_names=[
                "code_id",
                "lang",
                "split",
                "model",
                "pooling",
                "transform_ver",
                "dim",
                "vector",
            ],
        )

    def k_neighbors(
        self,
        qvec: List[float],
        k: int = 10,
        languages: Optional[Iterable[str]] = None,
        *,
        model: Optional[str] = EMB_MODEL,
        pooling: Optional[str] = EMB_POOL,
        transform_ver: Optional[int] = EMB_TVER,
    ) -> List[Tuple[str, float]]:
        """
        Return top-k neighbors as (code_id, similarity).
        Filters are OPTIONAL. Any None value will be omitted from WHERE.
        """
        client = get_ch_client()
        params = {
            "qvec": list(map(float, qvec)),
            "k": int(k),
        }
        where = []

        # Optional model meta filters
        if model is not None:
            where.append("model = {model:String}")
            params["model"] = model
        if pooling is not None:
            where.append("pooling = {pool:String}")
            params["pool"] = pooling
        if transform_ver is not None:
            where.append("transform_ver = {tver:UInt16}")
            params["tver"] = int(transform_ver)

        # Optional language filter
        if languages:
            where.append("lang IN {langs:Array(String)}")
            params["langs"] = list(languages)

        where_sql = " AND ".join(where) if where else "1"

        rows = client.query(
            f"""
            SELECT code_id,
                   1 - cosineDistance(vector, {{qvec:Array(Float32)}}) AS similarity
            FROM {EMB_TABLE}
            WHERE {where_sql}
            ORDER BY cosineDistance(vector, {{qvec:Array(Float32)}}) ASC, code_id ASC
            LIMIT {{k:UInt32}}
            """,
            parameters=params,
        ).result_rows

        return [(r[0], float(r[1])) for r in rows]

    def fetch_candidates_with_code(
        self,
        qvec: list[float],
        k: int = 20,
        languages: Optional[Iterable[str]] = None,
        *,
        model: Optional[str] = EMB_MODEL,
        pooling: Optional[str] = EMB_POOL,
        transform_ver: Optional[int] = EMB_TVER,
    ):
        """
        Returns rows: (code_id, lang, split, approx_sim, code_text, old_id).
        We join embeddings_v3 -> codes_v4 and shortlist by ANN in CH.
        """
        client = get_ch_client()
        params = {"qvec": list(map(float, qvec)), "k": int(k)}
        where = []

        if model is not None:
            where.append("e.model = {model:String}")
            params["model"] = model
        if pooling is not None:
            where.append("e.pooling = {pool:String}")
            params["pool"] = pooling
        if transform_ver is not None:
            where.append("e.transform_ver = {tver:UInt16}")
            params["tver"] = int(transform_ver)
        if languages:
            where.append("e.lang IN {langs:Array(String)}")
            params["langs"] = list(languages)

        where_sql = " AND ".join(where) if where else "1"

        rows = client.query(
            f"""
            SELECT
              e.code_id,
              e.lang,
              e.split,
              1 - cosineDistance(e.vector, {{qvec:Array(Float32)}}) AS approx_sim,
              c.code AS code_text,
              c.old_id
            FROM codebase.embeddings_v3 e
            INNER JOIN codebase.codes_v4 c ON c.code_id = e.code_id
            WHERE {where_sql}
            ORDER BY cosineDistance(e.vector, {{qvec:Array(Float32)}}) ASC, e.code_id ASC
            LIMIT {{k:UInt32}}
            """,
            parameters=params,
        ).result_rows

        # [(code_id, lang, split, approx_sim, code_text, old_id), ...]
        return rows
