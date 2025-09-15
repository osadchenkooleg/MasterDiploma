from typing import List, Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from app.api.deps import get_embed_model, get_embeddings_repo


class UniquenessRequest(BaseModel):
    code: str
    languages: Optional[List[str]] = None


class UniquenessResponse(BaseModel):
    uniqueness_percent: float
    closest_id: str | None
    similarity: float | None


router = APIRouter(prefix="/uniqueness", tags=["uniqueness"])


@router.post("", response_model=UniquenessResponse)
def compute_uniqueness(
    req: UniquenessRequest,
    embs=Depends(get_embeddings_repo),
    model=Depends(get_embed_model),
):
    qvec = model.encode(req.code)
    top1 = embs.nearest_top1(qvec, req.languages)
    if not top1:
        return UniquenessResponse(
            uniqueness_percent=100.0, closest_id=None, similarity=None
        )

    closest_id, sim = top1[0], float(top1[1] or 0.0)
    uniqueness = max(0.0, min(100.0, (1.0 - sim) * 100.0))
    return UniquenessResponse(
        uniqueness_percent=uniqueness, closest_id=closest_id, similarity=sim
    )
