import ulid
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from app.api.deps import get_embed_model
from app.api.deps_clickhouse import get_codes_repo_ch, get_embeddings_repo_ch


class AddCodeRequest(BaseModel):
    lang: str
    code: str


class AddCodeResponse(BaseModel):
    id: str


from fastapi import APIRouter, Depends, HTTPException, Query

router = APIRouter(prefix="/codes", tags=["codes"])


@router.post("", response_model=AddCodeResponse)
def add_code(
    req: AddCodeRequest,
    codes=Depends(get_codes_repo_ch),
    embs=Depends(get_embeddings_repo_ch),
    model=Depends(get_embed_model),
):
    code_id = str(ulid.ULID())
    codes.insert(code_id, req.lang, req.code)
    vec = model.encode(req.code)
    embs.insert(code_id, vec)
    return AddCodeResponse(id=code_id)


@router.get("/")
def get_code(
    id: str = Query(...),
    lang: str | None = Query(None),
    repo=Depends(get_codes_repo_ch),
):
    if lang:
        id = f"{lang}/{id}"
    row = repo.get(id)
    if row is None:
        raise HTTPException(status_code=404, detail="Code not found")
    return row
