from fastapi import APIRouter, Depends

from app.api.deps_clickhouse import get_languages_repo_ch

router = APIRouter(prefix="/languages", tags=["languages"])


@router.get("")
def list_languages(repo=Depends(get_languages_repo_ch)):
    return [{"lang": l} for l in repo.list_enabled()]
