from fastapi import APIRouter, Depends

from app.api.deps import get_languages_repo

router = APIRouter(prefix="/languages", tags=["languages"])


@router.get("")
def list_languages(repo=Depends(get_languages_repo)):
    return [{"lang": l} for l in repo.list_enabled()]
