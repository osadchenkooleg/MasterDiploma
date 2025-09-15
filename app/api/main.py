from fastapi import FastAPI

from .routers import codes, languages, search, uniqueness


def create_app() -> FastAPI:
    app = FastAPI(title="Code Backend", version="v1")
    app.include_router(languages.router)
    app.include_router(codes.router)
    app.include_router(search.router)
    app.include_router(uniqueness.router)
    return app


app = create_app()
