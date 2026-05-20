"""FastAPI app factory."""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .config import settings
from .database import init_db
from .routes import chat as chat_routes
from .routes import components as comp_routes
from .routes import pages as page_routes
from .routes import component_chat as comp_chat_routes
from .routes import sessions as sessions_routes


def create_app() -> FastAPI:
    init_db()

    app = FastAPI(title="Hardware Base", version="0.1.0")

    here = Path(__file__).resolve().parent
    static_dir = here.parent / "static"
    static_dir.mkdir(exist_ok=True)
    app.mount("/static", StaticFiles(directory=static_dir), name="static")
    # Serve datasheet PDFs directly so they can be opened in-browser.
    app.mount("/datasheets", StaticFiles(directory=settings.datasheet_dir_abs), name="datasheets")

    templates = Jinja2Templates(directory=str(here / "templates"))
    app.state.templates = templates

    app.include_router(page_routes.router)
    app.include_router(comp_routes.router)
    app.include_router(chat_routes.router)
    app.include_router(comp_chat_routes.router)
    app.include_router(sessions_routes.router)

    return app
