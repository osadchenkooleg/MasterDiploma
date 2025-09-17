from __future__ import annotations

from typing import List, Optional

import numpy as np
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from app.api.deps_clickhouse import get_embeddings_repo_ch
from app.infrastructure.embeddings.model_codebert import CodeEmbeddingModel

router = APIRouter(prefix="/uniqueness", tags=["uniqueness"])

# Model singleton for performance
_model_singleton: CodeEmbeddingModel | None = None


def get_model() -> CodeEmbeddingModel:
    global _model_singleton
    if _model_singleton is None:
        _model_singleton = CodeEmbeddingModel()
    return _model_singleton


def _parse_languages(
    langs_multi: Optional[List[str]], langs_csv: Optional[str]
) -> Optional[List[str]]:
    if langs_multi and len(langs_multi) > 0:
        return [s.strip() for s in langs_multi if s and s.strip()]
    if langs_csv:
        return [s.strip() for s in langs_csv.split(",") if s.strip()]
    return None


def _decode_bytes(b: bytes) -> str:
    # Be permissive with encodings; prioritise UTF-8
    try:
        return b.decode("utf-8")
    except UnicodeDecodeError:
        try:
            return b.decode("utf-8-sig")
        except UnicodeDecodeError:
            # Latin-1 fallback (no errors), preserves bytes -> unicode 1:1
            return b.decode("latin-1")


@router.post("/file")
async def uniqueness_from_file(
    file: UploadFile = File(..., description="Text file containing source code"),
    languages: Optional[List[str]] = Form(
        default=None,
        description="Repeatable form field: languages=java&languages=python",
    ),
    languages_csv: Optional[str] = Form(
        default=None,
        description="Alternative: comma-separated list, e.g. 'java,python'",
    ),
    embs=Depends(get_embeddings_repo_ch),
    model: CodeEmbeddingModel = Depends(get_model),
):
    # light size safety (e.g., 2 MB)
    MAX_BYTES = 2 * 1024 * 1024
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")
    if len(data) > MAX_BYTES:
        raise HTTPException(
            status_code=413, detail=f"File too large (> {MAX_BYTES} bytes)"
        )

    code_text = _decode_bytes(data)
    langs = _parse_languages(languages, languages_csv)

    # embed and query
    # qvec = model.encode(code_text).tolist()
    vec = model.encode(code_text)  # np.ndarray (float32, 768)
    print("DEBUG EMBED first10:", vec[:10])  # or use logging
    qvec = vec.tolist()
    top1 = embs.nearest_top1(qvec, langs)

    if not top1:
        return {"uniqueness_percent": 100.0, "closest_id": None, "similarity": None}

    closest_id, sim = top1  # sim in [0,1]
    uniqueness = max(0.0, min(100.0, (1.0 - float(sim)) * 100.0))
    return {
        "uniqueness_percent": uniqueness,
        "closest_id": closest_id,
        "similarity": float(sim),
    }


@router.post("/uniqueness/debug")
async def uniqueness_debug(
    file: UploadFile = File(..., description="Text file containing source code"),
    languages: Optional[List[str]] = Form(
        default=None,
        description="Repeatable form field: languages=java&languages=python",
    ),
    embs=Depends(get_embeddings_repo_ch),
    model=Depends(get_model),
):
    MAX_BYTES = 2 * 1024 * 1024
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")
    if len(data) > MAX_BYTES:
        raise HTTPException(
            status_code=413, detail=f"File too large (> {MAX_BYTES} bytes)"
        )

    code_text = _decode_bytes(data)
    vec = model.encode(code_text)  # np.ndarray (float32, 768)
    print("DEBUG EMBED first10:", vec[:10])  # or use logging
    qvec = vec.tolist()
    # vec = model.encode(req.code)
    return {
        "vec_first10": [float(x) for x in vec[:10]],
        "vec_norm": float(np.linalg.norm(vec)),
        "lang_filter": languages,
    }
