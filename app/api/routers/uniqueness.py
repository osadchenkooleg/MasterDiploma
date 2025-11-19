from __future__ import annotations

import hashlib
import logging
import math
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from app.api.deps import get_boilerplate_filter, get_embed_model
from app.api.deps_clickhouse import get_codes_repo_ch, get_embeddings_repo_ch
from app.infrastructure.db.clickhouse.client import get_ch_client

logger = logging.getLogger("uniqueness")
router = APIRouter(prefix="/uniqueness", tags=["uniqueness"])

WORD_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


# ============================ Models ============================


class NeighborDebug(BaseModel):
    code_id: str
    approx_sim: float
    code_len: int
    reembed_sim: float
    token_recall: float
    used_for_ranking: bool = False


class UniquenessResponse(BaseModel):
    uniqueness_percent: float
    closest_id: Optional[str]
    similarity: float
    stage: Optional[str] = Field(
        default=None,
        description="Decision stage: 'exact', 'lexical' or 'embedding'",
    )
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
        ver=int(getattr(model, "transform_ver", 3)),
    )


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity with defensive normalization of b."""
    b = b.astype(np.float32, copy=False)
    n = float(np.linalg.norm(b))
    if n > 0:
        b = b / n
    return float(np.dot(a, b))


def _length_penalty(len_a: int, len_b: int) -> float:
    """
    Smoothly penalize large length gaps.
    1.0 when equal; ~0.86 at 2x diff; ~0.61 at 4x diff (with k=2.0).
    Для коротких запитів штраф послаблюємо.
    """
    m = max(len_a, len_b)
    if m == 0:
        return 0.0

    if len_a < 200:
        k = 0.8
    else:
        k = 2.0

    gap = abs(len_a - len_b) / m
    return math.exp(-k * gap)


def _hybrid_score(sim_c: float, rec: float, len_pen: float, q_len: int) -> float:
    """
    Комбінований скор:
    - для коротких запитів більше ваги на token-recall,
    - для довгих – більше на cosine.
    """
    if q_len < 400:  # ~короткий snippet (в символах, не в рядках)
        alpha = 0.3  # 30% cosine, 70% recall
    else:
        alpha = 0.7  # 70% cosine, 30% recall

    base = alpha * sim_c + (1.0 - alpha) * rec
    return base * len_pen


def _soft_tokens(text: str) -> Set[str]:
    """
    Дуже проста токенізація:
    - беремо тільки "слова" (ідентифікатори / ключові слова),
    - лоуеркасимо,
    - не робимо ніякої анонімізації типу v1/v2.
    """
    return {t.lower() for t in WORD_RE.findall(text or "") if t}


def _token_recall_raw(query_code: str, cand_code: str) -> float:
    """
    Asymmetric recall: наскільки добре кандидат покриває токени запиту.
    |A ∩ B| / |A|, де A = токени запиту, B = токени кандидата.
    """
    A = _soft_tokens(query_code)
    B = _soft_tokens(cand_code)
    if not A:
        return 0.0
    inter = len(A & B)
    return inter / len(A)


def _token_jaccard_raw(a: str, b: str) -> float:
    """
    Симетричний Jaccard по простим токенам.
    """
    A = _soft_tokens(a)
    B = _soft_tokens(b)
    if not A and not B:
        return 0.0
    union = len(A | B)
    if union == 0:
        return 0.0
    inter = len(A & B)
    return inter / union


def _sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


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


# ============================ Stages ============================


def try_exact_hash_stage(
    code_raw: str,
    lang: Optional[str],
    debug: bool,
) -> Optional[UniquenessResponse]:
    client = get_ch_client()
    h = _sha256_hex(code_raw)
    params: Dict[str, Any] = {"h": h}

    if lang:
        sql = """
        SELECT code_id
        FROM codebase.codes_v4
        WHERE code_hash = %(h)s AND lang = %(lang)s
        LIMIT 1
        """
        params["lang"] = lang
    else:
        sql = """
        SELECT code_id
        FROM codebase.codes_v4
        WHERE code_hash = %(h)s
        LIMIT 1
        """

    res = client.query(sql, parameters=params)
    rows = res.result_rows  # список кортежів
    if not rows:
        return None

    code_id = rows[0][0]

    resp = UniquenessResponse(
        uniqueness_percent=0.0,
        closest_id=code_id,
        similarity=1.0,
        stage="exact",
    )
    if debug:
        resp.debug_info = {
            "stage": "exact",
            "code_hash": h,
        }
    return resp


def try_lexical_stage(
    code: str,
    lang: Optional[str],
    codes_repo,
    debug: bool,
) -> Optional[UniquenessResponse]:
    """
    Stage 'lexical': шукає сильні текстові збіги без ембеддингів.
    Використовує codes_repo.search_text + token_recall/jaccard.
    """
    # Щоб не ламати performance, можна обмежити довжину підрядка
    q_snippet = code if len(code) <= 512 else code[:512]
    languages = [lang] if lang else None

    total, hits = codes_repo.search_text(
        q=q_snippet,
        languages=languages,
        offset=0,
        limit=100,
    )
    if not hits:
        return None

    best_public_id: Optional[str] = None  # те, що повертає search_text (go/..., old_id)
    best_rec: float = 0.0
    best_jacc: float = 0.0

    for h in hits:
        cand_code = h.get("code") or ""
        cid = h.get("id")  # тут "go/ULID" або old_id

        rec = _token_recall_raw(code, cand_code)
        jacc = _token_jaccard_raw(code, cand_code)

        if rec > best_rec:
            best_rec = rec
            best_jacc = jacc
            best_public_id = cid

    # Пороги можна підкрутити; зараз агресивні для "майже тотожного" тексту
    if best_public_id and best_rec >= 0.95 and best_jacc >= 0.85:
        # Отут конвертуємо public id -> raw ULID через codes_repo.get
        row = codes_repo.get(best_public_id)
        canonical_code_id = (
            row["code_id"] if row and row.get("code_id") else best_public_id
        )

        resp = UniquenessResponse(
            uniqueness_percent=(1.0 - float(best_rec)) * 100.0,
            closest_id=canonical_code_id,  # <- ТУТ уже чистий ULID
            similarity=float(best_rec),
            stage="lexical",
        )
        if debug:
            resp.debug_info = {
                "stage": "lexical",
                "best_id": best_public_id,  # public id (go/ULID) для дебагу
                "canonical_code_id": canonical_code_id,
                "best_recall": best_rec,
                "best_jaccard": best_jacc,
                "hits": len(hits),
                "total": total,
            }
        return resp

    return None


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
    embs=Depends(get_embeddings_repo_ch),  # EmbeddingsRepoCH (v3/v4)
    codes=Depends(get_codes_repo_ch),  # CodesRepoCH (v4)
    model=Depends(get_embed_model),
    boiler=Depends(get_boilerplate_filter),
):
    t0 = time.perf_counter()

    # 1) Read file -> text
    raw = await file.read()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = raw.decode("utf-8", errors="replace")

    code_raw = text  # для hash-етапу
    code = text.strip()

    if not code:
        raise HTTPException(
            status_code=400, detail="Uploaded file is empty or not decodable as text."
        )

    # 1.1) Exact-hash stage
    try:
        exact_resp = try_exact_hash_stage(
            code_raw=code_raw, lang=lang, debug=bool(debug)
        )
    except Exception:
        logger.exception("Exact-hash stage failed")
        exact_resp = None

    if exact_resp is not None:
        return exact_resp

    # 1.2) Lexical stage
    try:
        lexical_resp = try_lexical_stage(
            code=code,
            lang=lang,
            codes_repo=codes,
            debug=bool(debug),
        )
    except Exception:
        logger.exception("Lexical stage failed")
        lexical_resp = None

    if lexical_resp is not None:
        return lexical_resp

    # 2) Embedding stage (поточний пайплайн)
    filtered = boiler.filter_for_embedding(code, lang)

    # 3) Encode query (модель сама L2-нормалізує)
    q_vec: np.ndarray = model.encode(filtered)
    qn = float(np.linalg.norm(q_vec))
    if not np.isfinite(qn) or qn == 0.0:
        resp = UniquenessResponse(
            uniqueness_percent=100.0,
            closest_id=None,
            similarity=0.0,
            stage="embedding",
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
            uniqueness_percent=100.0,
            closest_id=None,
            similarity=0.0,
            stage="embedding",
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
                    code_len=code_len,
                    reembed_sim=0.0,
                    token_recall=0.0,
                    used_for_ranking=False,
                )
            )
            continue

        # cosine on fresh vectors (defensive)
        cand_filtered = boiler.filter_for_embedding(cand_code, c_lang or lang)
        cand_vec = model.encode(cand_filtered)
        sim_c = _cosine(q_vec, cand_vec)

        rec = _token_recall_raw(code, cand_code)
        lp = _length_penalty(q_len, len(cand_code))

        # якщо кандидат покриває майже всі токени запиту – трактуємо як дуже сильний збіг
        if rec > 0.95 and sim_c > 0.80:
            sim_final = 1.0
        else:
            sim_final = _hybrid_score(sim_c, rec, lp, q_len)

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
                token_recall=float(rec),
                used_for_ranking=used,
            )
        )

    if best_id is None or best_sim_final <= 0.0:
        resp = UniquenessResponse(
            uniqueness_percent=100.0,
            closest_id=None,
            similarity=0.0,
            stage="embedding",
        )
        if debug:
            resp.debug_info = {
                "stage": "validated_but_no_match",
                "attempts": shortlist_attempts,
                "validated": [n.model_dump() for n in neighbors_dbg],
                "total_ms": round((time.perf_counter() - t0) * 1000, 1),
            }
        return resp

    # 5) Return result (embedding stage)
    resp = UniquenessResponse(
        uniqueness_percent=(1.0 - float(best_sim_final)) * 100.0,
        closest_id=best_id,
        similarity=float(best_sim_final),
        stage="embedding",
    )
    if debug:
        # include winner components
        chosen = next((n for n in neighbors_dbg if n.code_id == best_id), None)
        resp.debug_info = {
            "attempts": shortlist_attempts,
            "winner": best_id,
            "cosine": float(chosen.reembed_sim) if chosen else None,
            "token_recall": float(chosen.token_recall) if chosen else None,
            "final_sim": float(best_sim_final),
            "validated": [n.model_dump() for n in neighbors_dbg],
            "total_ms": round((time.perf_counter() - t0) * 1000, 1),
            "model": {"name": meta.name, "pool": meta.pool, "ver": meta.ver},
            "lang_filter": langs,
        }
    return resp
