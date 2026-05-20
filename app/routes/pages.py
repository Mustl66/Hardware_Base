"""HTML pages."""
from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse
from sqlalchemy.orm import Session

from ..database import Component, get_db
from ..search import search_components

router = APIRouter()


def _tpl(request: Request):
    return request.app.state.templates


@router.get("/", response_class=HTMLResponse)
def home(request: Request, q: str = "", category: str = "", db: Session = Depends(get_db)):
    hits = search_components(db, q, category=category or None, limit=100)
    categories = [r[0] for r in db.execute(
        __import__("sqlalchemy").text(
            "SELECT DISTINCT category FROM components WHERE category != '' ORDER BY category"
        )
    ).all()]
    return _tpl(request).TemplateResponse("index.html", {
        "request": request,
        "hits": hits,
        "q": q,
        "category": category,
        "categories": categories,
    })


@router.get("/components/{cid}", response_class=HTMLResponse)
def component_detail(cid: int, request: Request, db: Session = Depends(get_db)):
    comp = db.get(Component, cid)
    if not comp:
        return HTMLResponse("Not found", status_code=404)
    return _tpl(request).TemplateResponse("component_detail.html", {
        "request": request,
        "c": comp,
    })


@router.get("/upload", response_class=HTMLResponse)
def upload_page(request: Request):
    return _tpl(request).TemplateResponse("upload.html", {"request": request})


@router.get("/chat", response_class=HTMLResponse)
def chat_page(request: Request):
    return _tpl(request).TemplateResponse("chat.html", {"request": request})
