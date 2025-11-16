import logging

from fastapi import Depends, FastAPI

from app.api import deps_clickhouse as chdeps
from app.api.routers import uniqueness  # <-- new
from app.api.routers import codes, health, languages, search
from app.infrastructure.db.clickhouse.client import get_ch_client

logging.basicConfig(level=logging.INFO)
logging.getLogger("uniqueness").setLevel(logging.INFO)


def create_app() -> FastAPI:
    _ = get_ch_client()  # ping CH at startup
    app = FastAPI(title="Code Backend (ClickHouse)", version="v1")

    app.include_router(languages.router)
    app.include_router(codes.router)
    app.include_router(search.router)
    app.include_router(health.router)
    app.include_router(uniqueness.router)  # /uniqueness/file

    return app


app = create_app()
