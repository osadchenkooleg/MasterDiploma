from typing import List, Optional

from fastapi import APIRouter, Depends, Query, Response

from app.api.deps_clickhouse import get_codes_repo_ch

router = APIRouter(prefix="/search", tags=["search"])


@router.get("")
def search(
    q: str,
    page: int = 1,
    page_size: int = 20,
    languages: Optional[List[str]] = Query(default=None),
    repo=Depends(get_codes_repo_ch),
    response: Response = None,
):
    page = max(page, 1)
    page_size = min(max(page_size, 1), 100)
    offset = (page - 1) * page_size
    total, items = repo.search_text(q, languages, offset, page_size)
    response.headers["X-Total-Count"] = str(total)
    return {"items": items, "page": page, "page_size": page_size}
