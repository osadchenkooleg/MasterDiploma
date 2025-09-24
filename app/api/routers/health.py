# app/api/routers/health.py
from fastapi import APIRouter

from app.infrastructure.db.clickhouse.client import get_ch_client

router = APIRouter(prefix="/health", tags=["health"])


@router.get("/storage")
def storage_health():
    c = get_ch_client()
    counts = c.query(
        """
      SELECT
        (SELECT count() FROM codebase.codes_v4)  AS codes_v2,
        (SELECT count() FROM codebase.embeddings_v3) AS emb_v2
    """
    ).first_row
    return {"codes_v4": int(counts[0]), "embeddings_v3": int(counts[1])}
