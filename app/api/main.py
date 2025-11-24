import logging

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routers import codes, health, languages, search, thresholds, uniqueness
from app.infrastructure.db.clickhouse.client import get_ch_client

logging.basicConfig(level=logging.INFO)
logging.getLogger("uniqueness").setLevel(logging.INFO)


def create_app() -> FastAPI:
    _ = get_ch_client()  # ping CH at startup
    app = FastAPI(title="Code Backend (ClickHouse)", version="v1")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:8080", "http://127.0.0.1:5173"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(languages.router)
    app.include_router(codes.router)
    app.include_router(search.router)
    app.include_router(health.router)
    app.include_router(uniqueness.router)
    app.include_router(thresholds.router)

    return app


app = create_app()
