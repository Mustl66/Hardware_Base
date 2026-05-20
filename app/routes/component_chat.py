"""Per-component AI chat + saved notes CRUD."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ..assistant import stream_component_answer
from ..database import Component, ComponentNote, get_db

router = APIRouter(prefix="/api/components/{cid}", tags=["component-chat"])


# ── Chat ─────────────────────────────────────────────────────────────────────

class CompChatRequest(BaseModel):
    message: str
    history: list[dict] = []


@router.post("/chat/stream")
async def component_chat_stream(cid: int, req: CompChatRequest, db: Session = Depends(get_db)):
    comp = db.get(Component, cid)
    if not comp:
        raise HTTPException(404, "Component not found")

    async def gen():
        try:
            async for delta in stream_component_answer(db, comp, req.message, req.history):
                yield delta
        except Exception as e:
            yield f"\n\n[error: {e}]"

    return StreamingResponse(gen(), media_type="text/plain; charset=utf-8")


# ── Notes CRUD ────────────────────────────────────────────────────────────────

class NoteCreate(BaseModel):
    question: str
    content: str


class NoteUpdate(BaseModel):
    content: str


@router.get("/notes")
def list_notes(cid: int, db: Session = Depends(get_db)):
    comp = db.get(Component, cid)
    if not comp:
        raise HTTPException(404, "Component not found")
    return JSONResponse([
        {
            "id": n.id,
            "question": n.question,
            "content": n.content,
            "created_at": n.created_at.isoformat() if n.created_at else None,
            "updated_at": n.updated_at.isoformat() if n.updated_at else None,
        }
        for n in comp.notes
    ])


@router.post("/notes")
def create_note(cid: int, body: NoteCreate, db: Session = Depends(get_db)):
    comp = db.get(Component, cid)
    if not comp:
        raise HTTPException(404, "Component not found")
    note = ComponentNote(component_id=cid, question=body.question, content=body.content)
    db.add(note)
    db.commit()
    db.refresh(note)
    return JSONResponse({
        "id": note.id,
        "question": note.question,
        "content": note.content,
        "created_at": note.created_at.isoformat() if note.created_at else None,
        "updated_at": note.updated_at.isoformat() if note.updated_at else None,
    })


@router.put("/notes/{nid}")
def update_note(cid: int, nid: int, body: NoteUpdate, db: Session = Depends(get_db)):
    note = db.get(ComponentNote, nid)
    if not note or note.component_id != cid:
        raise HTTPException(404, "Note not found")
    note.content = body.content
    db.commit()
    db.refresh(note)
    return JSONResponse({
        "id": note.id,
        "question": note.question,
        "content": note.content,
        "updated_at": note.updated_at.isoformat() if note.updated_at else None,
    })


@router.delete("/notes/{nid}")
def delete_note(cid: int, nid: int, db: Session = Depends(get_db)):
    note = db.get(ComponentNote, nid)
    if not note or note.component_id != cid:
        raise HTTPException(404, "Note not found")
    db.delete(note)
    db.commit()
    return JSONResponse({"ok": True})
