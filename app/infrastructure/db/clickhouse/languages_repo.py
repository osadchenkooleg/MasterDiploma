from __future__ import annotations

from typing import List

from app.infrastructure.db.clickhouse.client import get_ch_client


class LanguagesRepoCH:
    """
    ClickHouse implementation for /languages
    """

    def list_enabled(self) -> List[str]:
        client = get_ch_client()
        rows = client.query(
            """
            SELECT lang FROM codebase.languages
            WHERE enabled = 1
            ORDER BY lang
        """
        ).result_rows
        return [r[0] for r in rows]
