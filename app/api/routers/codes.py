import ulid
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from app.api.deps import get_codes_repo, get_embed_model, get_embeddings_repo


class AddCodeRequest(BaseModel):
    lang: str
    code: str


class AddCodeResponse(BaseModel):
    id: str


router = APIRouter(prefix="/codes", tags=["codes"])


@router.post("", response_model=AddCodeResponse)
def add_code(
    req: AddCodeRequest,
    codes=Depends(get_codes_repo),
    embs=Depends(get_embeddings_repo),
    model=Depends(get_embed_model),
):
    code_id = str(ulid.ULID())
    codes.insert(code_id, req.lang, req.code)
    vec = model.encode(req.code)
    embs.insert(code_id, vec)
    return AddCodeResponse(id=code_id)


@router.get("/{id}")
def get_code(id: str, repo=Depends(get_codes_repo)):
    row = repo.get(id)
    if not row:
        raise HTTPException(status_code=404, detail="Not found")
    return row
