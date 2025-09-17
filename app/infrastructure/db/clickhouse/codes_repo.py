from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple

from app.infrastructure.db.clickhouse.client import get_ch_client


class CodesRepoCH:
    """
    ClickHouse implementation for code rows (add/get/search-test).
    Table: codebase.codes(id String, lang LC(String), split LC(String), label LC(String), code Nullable(String), created_at DateTime)
    """

    def insert(
        self,
        id: str,
        lang: str,
        code: Optional[str],
        split: str = "inbox",
        label: str = "",
    ) -> None:
        client = get_ch_client()
        # parameterized single row insert (command)
        client.command(
            """
            INSERT INTO codebase.codes (id, lang, split, label, code)
            VALUES (%(id)s, %(lang)s, %(split)s, %(label)s, %(code)s)
        """,
            parameters={
                "id": id,
                "lang": lang,
                "split": split,
                "label": label,
                "code": code,
            },
        )

    def get(self, id: str) -> Optional[Dict[str, Any]]:
        client = get_ch_client()
        row = client.query(
            """
            SELECT id, lang, split, label, code, created_at
            FROM codebase.codes
            WHERE id = %(id)s
            LIMIT 1
        """,
            parameters={"id": id},
        ).first_row
        if not row:
            return None
        return {
            "id": row[0],
            "lang": row[1],
            "split": row[2],
            "label": row[3],
            "code": row[4],
            "created_at": str(row[5]),
        }

    def search_text(
        self, q: str, languages: Optional[Iterable[str]], offset: int, limit: int
    ) -> Tuple[int, List[Dict[str, Any]]]:
        client = get_ch_client()
        params = {
            "q": q,
            "langs": list(languages) if languages else [],
            "limit": int(limit),
            "offset": int(offset),
        }

        total = client.query(
            """
            SELECT count()
            FROM codebase.codes
            WHERE positionCaseInsensitive(code, %(q)s) > 0
              AND (empty(%(langs)s) OR lang IN %(langs)s)
        """,
            parameters=params,
        ).first_item

        rows = client.query(
            """
            SELECT id, lang, code
            FROM codebase.codes
            WHERE positionCaseInsensitive(code, %(q)s) > 0
              AND (empty(%(langs)s) OR lang IN %(langs)s)
            ORDER BY id
            LIMIT %(limit)s OFFSET %(offset)s
        """,
            parameters=params,
        ).result_rows

        items = [{"id": r[0], "lang": r[1], "code": r[2]} for r in rows]
        return int(total), items
