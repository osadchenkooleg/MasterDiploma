# app/infra/clickhouse/repos/codes_repo.py
from __future__ import annotations

import os
from typing import Any, Dict, Iterable, List, Optional, Tuple

from app.infrastructure.db.clickhouse.client import get_ch_client

CODES_TABLE = "codebase.codes_v3"  # or env-flagged


def _split_composite_id(s: str):
    return s.split("/", 1) if "/" in s else (None, s)


class CodesRepoCH:
    def get(self, composite_or_plain_id: str):
        client = get_ch_client()
        lang_hint, code_id = _split_composite_id(composite_or_plain_id)

        if lang_hint:
            q = client.query(
                f"""
                SELECT id, lang, split, label, code
                FROM {CODES_TABLE}
                WHERE id = %(id)s
                LIMIT 1
            """,
                parameters={"id": composite_or_plain_id},
            )
        else:
            # No lang given -> pick the most recent across langs (or return 404 if you prefer)
            q = client.query(
                f"""
                SELECT id, lang, split, label, code
                FROM {CODES_TABLE}
                WHERE code_id = %(cid)s
                ORDER BY id ASC  -- or ingested_at DESC if present
                LIMIT 1
            """,
                parameters={"cid": code_id},
            )

        rows = q.result_rows
        if not rows:
            return None  # let router turn into 404
        r = rows[0]
        return {"id": r[0], "lang": r[1], "split": r[2], "label": r[3], "code": r[4]}

    def search_text(self, q: str, languages=None, offset=0, limit=10):
        client = get_ch_client()
        params = {"q": q, "langs": languages or [], "limit": limit, "offset": offset}
        total = client.query(
            f"""
            SELECT count()
            FROM {CODES_TABLE}
            WHERE positionCaseInsensitive(code, %(q)s) > 0
              AND (empty(%(langs)s) OR lang IN %(langs)s)
        """,
            parameters=params,
        ).first_item
        rows = client.query(
            f"""
            SELECT id, lang, code
            FROM {CODES_TABLE}
            WHERE positionCaseInsensitive(code, %(q)s) > 0
              AND (empty(%(langs)s) OR lang IN %(langs)s)
            ORDER BY id
            LIMIT %(limit)s OFFSET %(offset)s
        """,
            parameters=params,
        ).result_rows
        return int(total), [{"id": r[0], "lang": r[1], "code": r[2]} for r in rows]
