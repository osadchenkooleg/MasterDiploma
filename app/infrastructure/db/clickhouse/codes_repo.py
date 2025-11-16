# app/infra/clickhouse/repos/codes_repo.py
from __future__ import annotations

import hashlib
from typing import Optional, Tuple

from app.infrastructure.db.clickhouse.client import get_ch_client

CODES_TABLE = "codebase.codes_v4"


def _split_composite_id(s: str) -> Tuple[Optional[str], str]:
    # "go/96401_1" -> ("go", "96401_1"); "01J9Y2..." -> (None, "01J9Y2...")
    return s.split("/", 1) if "/" in s else (None, s)


def _sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class CodesRepoCH:
    """
    Repo for codes_v4:
      code_id (ULID)  | old_id (legacy) | lang | split | label | code | code_hash | source
    """

    # ---------- writes ----------

    def insert(
        self,
        code_id: str,
        lang: str,
        code: Optional[str],
        *,
        split: str = "",
        label: str = "",
        source: str = "api",
        old_id: str = "",
        code_hash: Optional[str] = None,
    ) -> None:
        """
        Insert a code row.
        - code_id: ULID you generate in the API
        - old_id: keep empty for brand-new rows; set for migrated legacy ids
        """
        client = get_ch_client()
        chash = (
            code_hash
            if code_hash is not None
            else (_sha256_hex(code) if code is not None else None)
        )
        rows = [
            (
                code_id,
                old_id or "",
                lang or "",
                split or "",
                label or "",
                code,  # Nullable(String) -> pass None when absent
                chash,  # Nullable(String)
                source or "api",
            )
        ]
        client.insert(
            CODES_TABLE,
            rows,
            column_names=[
                "code_id",
                "old_id",
                "lang",
                "split",
                "label",
                "code",
                "code_hash",
                "source",
            ],
        )

    # ---------- reads ----------

    def get(self, composite_or_plain_id: str):
        """
        Accepts either:
          - legacy composite id like 'go/96401_1' (matches old_id)
          - ULID (matches code_id)
        Returns a dict with a stable 'id' for backward-compat:
          - prefer old_id when present; otherwise 'lang/code_id'
        """
        client = get_ch_client()
        lang_hint, id_part = _split_composite_id(composite_or_plain_id)

        if lang_hint:
            q = client.query(
                f"""
                SELECT old_id, lang, split, label, code, code_id
                FROM {CODES_TABLE}
                WHERE old_id = %(oid)s
                LIMIT 1
                """,
                parameters={"oid": composite_or_plain_id},
            )
        else:
            q = client.query(
                f"""
                SELECT old_id, lang, split, label, code, code_id
                FROM {CODES_TABLE}
                WHERE code_id = %(cid)s
                LIMIT 1
                """,
                parameters={"cid": id_part},
            )

        rows = q.result_rows
        if not rows:
            return None

        old_id, lang, split, label, code, code_id = rows[0]
        public_id = old_id if old_id else (f"{lang}/{code_id}" if lang else code_id)
        return {
            "id": public_id,
            "lang": lang,
            "split": split,
            "label": label,
            "code": code,
            # expose raw ULID in case callers want it
            "code_id": code_id,
            "old_id": old_id,
        }

    def search_text(self, q: str, languages=None, offset: int = 0, limit: int = 10):
        """
        Simple LIKE-ish search over code.
        Returns legacy-like 'id' to keep routers/views unchanged.
        """
        client = get_ch_client()
        params = {
            "q": q,
            "langs": languages or [],
            "limit": int(limit),
            "offset": int(offset),
        }

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
            SELECT
              if(old_id != '', old_id, concat(lang, '/', code_id)) AS id,
              lang,
              code
            FROM {CODES_TABLE}
            WHERE positionCaseInsensitive(code, %(q)s) > 0
              AND (empty(%(langs)s) OR lang IN %(langs)s)
            ORDER BY id
            LIMIT %(limit)s OFFSET %(offset)s
            """,
            parameters=params,
        ).result_rows

        return int(total), [{"id": r[0], "lang": r[1], "code": r[2]} for r in rows]
