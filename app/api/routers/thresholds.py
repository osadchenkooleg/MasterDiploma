# app/thresholds/schemas.py
from datetime import datetime

# app/thresholds/router.py
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from app.api.deps_clickhouse import get_threshold_repo_ch
from app.infrastructure.db.clickhouse.thresholds_repo import ThresholdPolicyRepository

router = APIRouter(prefix="/thresholds", tags=["thresholds"])


class CurrentThresholdsResponse(BaseModel):
    t_low: float
    t_high: float
    created_at: datetime


@router.get("/current", response_model=CurrentThresholdsResponse)
def get_current_thresholds(
    repo: ThresholdPolicyRepository = Depends(get_threshold_repo_ch),
):
    policy = repo.get_latest_policy()
    if policy is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No active threshold policy found",
        )

    return CurrentThresholdsResponse(
        t_low=policy.t_low,
        t_high=policy.t_high,
        created_at=policy.created_at,
    )
