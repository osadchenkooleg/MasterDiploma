# src/routers/uniqueness.py
from __future__ import annotations

import logging
import math
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from app.api.deps import get_embed_model
from app.api.deps_clickhouse import get_codes_repo_ch, get_embeddings_repo_ch

logger = logging.getLogger("uniqueness")
router = APIRouter(prefix="/uniqueness", tags=["uniqueness"])


# ============================ Models ============================


class NeighborDebug(BaseModel):
    code_id: str
    approx_sim: float
    code_len: int
    reembed_sim: float
    token_jaccard: float
    used_for_ranking: bool = False


class UniquenessResponse(BaseModel):
    uniqueness_percent: float
    closest_id: Optional[str]
    similarity: float
    debug_info: Optional[Dict[str, Any]] = Field(
        default=None, description="Returned only if debug=1"
    )


# ============================ Helpers ============================


@dataclass
class ModelMeta:
    name: str
    pool: str
    ver: int


def _model_meta(model) -> ModelMeta:
    return ModelMeta(
        name=getattr(model, "model_name", "microsoft/codebert-base"),
        pool=getattr(model, "pooling", "mean"),
        ver=int(getattr(model, "transform_ver", 2)),
    )


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity with defensive normalization of b."""
    b = b.astype(np.float32, copy=False)
    n = float(np.linalg.norm(b))
    if n > 0:
        b = b / n
    return float(np.dot(a, b))


_tok = re.compile(r"[A-Za-z_][A-Za-z_0-9]*")

GO_STOP = {
    # common Go boilerplate & keywords to down-weight trivial overlap
    "package",
    "import",
    "func",
    "main",
    "fmt",
    "println",
    "printf",
    "var",
    "const",
    "type",
    "if",
    "for",
    "return",
    "go",
    "defer",
    "range",
    "make",
    "new",
    "len",
    "cap",
    "true",
    "false",
    "nil",
}


def _norm_tokens(text: str, lang: Optional[str]) -> set[str]:
    toks = set(_tok.findall(text.lower()))
    if lang == "go":
        toks = {t for t in toks if t not in GO_STOP}
    return toks


def _token_jaccard_filtered(a_text: str, b_text: str, lang: Optional[str]) -> float:
    A = _norm_tokens(a_text, lang)
    B = _norm_tokens(b_text, lang)
    if not A or not B:
        return 0.0
    inter = len(A & B)
    union = len(A | B)
    return inter / union if union else 0.0


def _length_penalty(len_a: int, len_b: int) -> float:
    """
    Smoothly penalize large length gaps.
    1.0 when equal; ~0.86 at 2x diff; ~0.61 at 4x diff (with k=2.0).
    """
    m = max(len_a, len_b)
    if m == 0:
        return 0.0
    gap = abs(len_a - len_b) / m
    return math.exp(-2.0 * gap)


def _hybrid_score(sim_c: float, jacc: float, len_pen: float) -> float:
    """
    Multiplicative hybrid score:
      final = cosine * (0.5 + 0.5*jaccard) * length_penalty
    Ensures we only reduce optimistic cosine when lexical overlap is weak and/or lengths differ a lot.
    """
    mod = 0.5 + 0.5 * jacc  # in [0.5, 1.0]
    return sim_c * mod * len_pen


def _fallback_fetch_with_code(
    embs_repo,
    codes_repo,
    qvec: List[float],
    k: int,
    languages: Optional[Iterable[str]],
    model: Optional[str],
    pooling: Optional[str],
    transform_ver: Optional[int],
) -> List[Tuple[str, str, str, float, Optional[str], Optional[str]]]:
    """
    Fallback path if repo doesn't expose fetch_candidates_with_code.
    Uses k_neighbors() → fetch code via codes_repo.get(id).
    Returns (code_id, lang, split, approx_sim, code_text, old_id).
    """
    neighbors = embs_repo.k_neighbors(
        qvec=qvec,
        k=k,
        languages=languages,
        model=model,
        pooling=pooling,
        transform_ver=transform_ver,
    )
    rows: List[Tuple[str, str, str, float, Optional[str], Optional[str]]] = []
    for code_id, approx_sim in neighbors:
        row = codes_repo.get(code_id)  # supports ULID direct
        lang = row.get("lang") if row else ""
        split = row.get("split") if row else ""
        code = row.get("code") if row else None
        old_id = row.get("old_id") if row else None
        rows.append((code_id, lang, split, float(approx_sim), code, old_id))
    return rows


# ============================ Endpoint ============================


@router.post("/check", response_model=UniquenessResponse)
async def check_uniqueness_file(
    file: UploadFile = File(..., description="Code file (text, UTF-8 preferred)"),
    lang: Optional[str] = Form(
        None, description="Programming language filter (e.g., 'go')"
    ),
    top_k: int = Form(
        5, ge=1, le=50, description="How many nearest neighbors to retrieve"
    ),
    debug: int = Form(0, description="Return debug info if 1"),
    embs=Depends(get_embeddings_repo_ch),  # EmbeddingsRepoCH (v3)
    codes=Depends(get_codes_repo_ch),  # CodesRepoCH (v4)
    model=Depends(get_embed_model),
):
    t0 = time.perf_counter()

    # 1) Read file -> text
    raw = await file.read()
    try:
        code = raw.decode("utf-8")
    except UnicodeDecodeError:
        code = raw.decode("utf-8", errors="replace")
    code = code.strip()
    if not code:
        raise HTTPException(
            status_code=400, detail="Uploaded file is empty or not decodable as text."
        )

    # 2) Encode query (your model already L2-normalizes output)
    q_vec: np.ndarray = model.encode(code)
    qn = float(np.linalg.norm(q_vec))
    if not np.isfinite(qn) or qn == 0.0:
        resp = UniquenessResponse(
            uniqueness_percent=100.0, closest_id=None, similarity=0.0
        )
        if debug:
            resp.debug_info = {"stage": "encode", "q_norm": qn}
        return resp

    meta = _model_meta(model)
    langs = [lang] if lang else None
    shortlist_attempts: List[Dict[str, Any]] = []
    rows: List[Tuple[str, str, str, float, Optional[str], Optional[str]]] = []

    def try_shortlist(
        model_f, pool_f, ver_f, langs_f
    ) -> List[Tuple[str, str, str, float, Optional[str], Optional[str]]]:
        """
        Returns rows: (code_id, lang, split, approx_sim, code_text, old_id)
        """
        if hasattr(embs, "fetch_candidates_with_code"):
            return embs.fetch_candidates_with_code(
                qvec=q_vec.tolist(),
                k=max(20, top_k * 3),
                languages=langs_f,
                model=model_f,
                pooling=pool_f,
                transform_ver=ver_f,
            )
        # Fallback path
        return _fallback_fetch_with_code(
            embs,
            codes,
            q_vec.tolist(),
            max(20, top_k * 3),
            langs_f,
            model_f,
            pool_f,
            ver_f,
        )

    # 3) Shortlist candidates (strict → relaxed)
    for attempt, (m, p, v, L) in enumerate(
        [
            (meta.name, meta.pool, meta.ver, langs),
            (meta.name, meta.pool, meta.ver, None),
            (None, None, None, None),
        ],
        start=1,
    ):
        t = time.perf_counter()
        try:
            rows = try_shortlist(m, p, v, L)
        except Exception as e:
            logger.exception("Shortlist failed on attempt %d", attempt)
            rows = []
            err = repr(e)
        else:
            err = None
        took_ms = round((time.perf_counter() - t) * 1000, 1)
        shortlist_attempts.append(
            {
                "attempt": attempt,
                "filters": {"model": m, "pool": p, "ver": v, "langs": L},
                "rows": len(rows),
                "ms": took_ms,
                "error": err,
            }
        )
        if rows:
            break

    if not rows:
        resp = UniquenessResponse(
            uniqueness_percent=100.0, closest_id=None, similarity=0.0
        )
        if debug:
            resp.debug_info = {
                "stage": "shortlist_empty",
                "attempts": shortlist_attempts,
            }
        return resp

    # 4) Re-embed candidates’ actual text & compute hybrid score
    best_id: Optional[str] = None
    best_sim_final: float = -1.0
    validate_n = min(len(rows), max(10, top_k * 2))
    q_len = len(code)

    neighbors_dbg: List[NeighborDebug] = []

    for code_id, c_lang, c_split, approx_sim, cand_code, old_id in rows[:validate_n]:
        code_len = len(cand_code or "")
        if not cand_code:
            neighbors_dbg.append(
                NeighborDebug(
                    code_id=code_id,
                    approx_sim=float(approx_sim),
                    code_len=0,
                    reembed_sim=0.0,
                    token_jaccard=0.0,
                    used_for_ranking=False,
                )
            )
            continue

        # cosine on fresh vectors (defensive)
        cand_vec = model.encode(cand_code)
        sim_c = _cosine(q_vec, cand_vec)

        # lexical & length components
        jacc = _token_jaccard_filtered(code, cand_code, lang)
        lp = _length_penalty(q_len, code_len)

        sim_final = _hybrid_score(sim_c, jacc, lp)

        # optional absolute sanity cap for boilerplate
        if sim_c > 0.985 and jacc < 0.08:
            sim_final *= 0.85

        used = False
        if sim_final > best_sim_final:
            best_sim_final = sim_final
            best_id = code_id
            used = True

        neighbors_dbg.append(
            NeighborDebug(
                code_id=code_id,
                approx_sim=float(approx_sim),
                code_len=code_len,
                reembed_sim=float(sim_c),
                token_jaccard=float(jacc),
                used_for_ranking=used,
            )
        )

    if best_id is None or best_sim_final <= 0.0:
        resp = UniquenessResponse(
            uniqueness_percent=100.0, closest_id=None, similarity=0.0
        )
        if debug:
            resp.debug_info = {
                "stage": "validated_but_no_match",
                "attempts": shortlist_attempts,
                "validated": [n.model_dump() for n in neighbors_dbg],
                "total_ms": round((time.perf_counter() - t0) * 1000, 1),
            }
        return resp

    # 5) Return result
    resp = UniquenessResponse(
        uniqueness_percent=(1.0 - float(best_sim_final)) * 100.0,
        closest_id=best_id,
        similarity=float(best_sim_final),
    )
    if debug:
        # include winner components
        chosen = next((n for n in neighbors_dbg if n.code_id == best_id), None)
        resp.debug_info = {
            "attempts": shortlist_attempts,
            "winner": best_id,
            "cosine": float(chosen.reembed_sim) if chosen else None,
            "token_jaccard": float(chosen.token_jaccard) if chosen else None,
            "final_sim": float(best_sim_final),
            "validated": [n.model_dump() for n in neighbors_dbg],
            "total_ms": round((time.perf_counter() - t0) * 1000, 1),
            "model": {"name": meta.name, "pool": meta.pool, "ver": meta.ver},
            "lang_filter": langs,
        }
    return resp
