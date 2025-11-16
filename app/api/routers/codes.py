import ulid
from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from pydantic import BaseModel

from app.api.deps import get_boilerplate_filter, get_embed_model
from app.api.deps_clickhouse import get_codes_repo_ch, get_embeddings_repo_ch


class AddCodeResponse(BaseModel):
    id: str  # we’ll return the new ULID code_id


router = APIRouter(prefix="/codes", tags=["codes"])


@router.post("", response_model=AddCodeResponse, status_code=201)
async def add_code(
    lang: str = Form(..., description="Programming language of the uploaded code"),
    file: UploadFile = File(..., description="Code file (text)"),
    codes=Depends(get_codes_repo_ch),  # CodesRepoCH (v4)
    embs=Depends(get_embeddings_repo_ch),  # EmbeddingsRepoCH (v3)
    model=Depends(get_embed_model),
    boiler=Depends(get_boilerplate_filter),
):
    # Read & decode file (UTF-8 preferred; tolerate errors)
    raw = await file.read()
    try:
        code = raw.decode("utf-8")
    except UnicodeDecodeError:
        code = raw.decode("utf-8", errors="replace")

    if not code.strip():
        raise HTTPException(
            status_code=400, detail="Uploaded file is empty or not decodable as text."
        )

    # Create new code_id (ULID string)
    code_id = str(ulid.ULID())

    # Store code in codes_v4
    # (repo signature: insert(code_id, lang, code, *, split="", label="", source="api", old_id="", code_hash=None))
    codes.insert(code_id, lang, code, source="api")

    # Compute and store embedding in embeddings_v3
    filtered = boiler.filter_for_embedding(code, lang)
    vec = model.encode(filtered)

    embs.insert(code_id, vec, lang=lang, split="")

    return AddCodeResponse(id=code_id)


@router.get("/")
def get_code(
    id: str = Query(..., description="Either legacy 'lang/oldid' or new ULID code_id"),
    repo=Depends(get_codes_repo_ch),
):
    row = repo.get(id)
    if row is None:
        raise HTTPException(status_code=404, detail="Code not found")
    return row
