from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel

from app.infrastructure.db.clickhouse.client import get_ch_client

THRESHOLDS_TABLE = "codebase.threshold_policies"


class ThresholdPolicy(BaseModel):
    t_low: float
    t_high: float
    created_at: datetime


class ThresholdPolicyRepository:
    """
    Repo for reading current threshold policy from ClickHouse.

    Uses get_ch_client() per call (same pattern as CodesRepoCH),
    without storing a shared client on the instance.
    """

    def get_latest_policy(self) -> Optional[ThresholdPolicy]:
        """
        Повертає найсвіжішу активну політику з threshold_policies
        (за created_at), або None, якщо немає жодного запису.
        """
        client = get_ch_client()

        q = client.query(
            f"""
            SELECT t_low, t_high, created_at
            FROM {THRESHOLDS_TABLE}
            WHERE is_active = 1
            ORDER BY created_at DESC
            LIMIT 1
            """
        )

        rows = q.result_rows
        if not rows:
            return None

        t_low, t_high, created_at = rows[0]
        return ThresholdPolicy(
            t_low=float(t_low),
            t_high=float(t_high),
            created_at=created_at,
        )
