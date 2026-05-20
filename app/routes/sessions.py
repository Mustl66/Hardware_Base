"""Persistent chat sessions: global + component-scoped.

Endpoints
---------
GET    /api/sessions                  list all sessions (sidebar)
POST   /api/sessions                  create new session
DELETE /api/sessions/{sid}            delete session
GET    /api/sessions/{sid}/messages   full message history
POST   /api/sessions/{sid}/chat       stream one turn + persist both turns
PATCH  /api/sessions/{sid}/title      rename session
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ..assistant import stream_answer, stream_component_answer
from ..database import ChatSession, ChatSessionMessage, Component, get_db

router = APIRouter(prefix="/api/sessions", tags=["sessions"])


# ── helpers ──────────────────────────────────────────────────────────────────

def _session_summary(s: ChatSession) -> dict:
    last = s.messages[-1].content[:80] if s.messages else ""
    return {
        "id": s.id,
        "title": s.title,
        "component_id": s.component_id,
        "component_part": s.component.part_number if s.component else None,
        "updated_at": s.updated_at.isoformat() if s.updated_at else None,
        "preview": last,
        "message_count": len(s.messages),
    }


def _msg_dict(m: ChatSessionMessage) -> dict:
    return {"id": m.id, "role": m.role, "content": m.content,
            "created_at": m.created_at.isoformat() if m.created_at else None}


# ── list / create / delete sessions ──────────────────────────────────────────

@router.get("")
def list_sessions(db: Session = Depends(get_db)):
    sessions = (
        db.query(ChatSession)
        .order_by(ChatSession.updated_at.desc())
        .all()
    )
    return JSONResponse([_session_summary(s) for s in sessions])


class SessionCreate(BaseModel):
    component_id: int | None = None
    title: str = "New Chat"


@router.post("")
def create_session(body: SessionCreate, db: Session = Depends(get_db)):
    comp = None
    if body.component_id:
        comp = db.get(Component, body.component_id)
        if not comp:
            raise HTTPException(404, "Component not found")
    title = body.title
    if title == "New Chat" and comp:
        title = f"{comp.part_number} Chat"
    s = ChatSession(title=title, component_id=body.component_id)
    db.add(s)
    db.commit()
    db.refresh(s)
    return JSONResponse(_session_summary(s))


@router.delete("/{sid}")
def delete_session(sid: int, db: Session = Depends(get_db)):
    s = db.get(ChatSession, sid)
    if not s:
        raise HTTPException(404, "Session not found")
    db.delete(s)
    db.commit()
    return JSONResponse({"ok": True})


# ── rename ────────────────────────────────────────────────────────────────────

class TitleUpdate(BaseModel):
    title: str


@router.patch("/{sid}/title")
def rename_session(sid: int, body: TitleUpdate, db: Session = Depends(get_db)):
    s = db.get(ChatSession, sid)
    if not s:
        raise HTTPException(404, "Session not found")
    s.title = body.title[:200]
    db.commit()
    return JSONResponse({"id": s.id, "title": s.title})


# ── get messages ──────────────────────────────────────────────────────────────

@router.get("/{sid}/messages")
def get_messages(sid: int, db: Session = Depends(get_db)):
    s = db.get(ChatSession, sid)
    if not s:
        raise HTTPException(404, "Session not found")
    return JSONResponse([_msg_dict(m) for m in s.messages])


# ── chat (streaming, persists both turns) ─────────────────────────────────────

class ChatTurn(BaseModel):
    message: str


@router.post("/{sid}/chat")
async def session_chat(sid: int, body: ChatTurn, db: Session = Depends(get_db)):
    s = db.get(ChatSession, sid)
    if not s:
        raise HTTPException(404, "Session not found")

    # Persist user message immediately
    user_msg = ChatSessionMessage(session_id=sid, role="user", content=body.message)
    db.add(user_msg)
    db.commit()

    # Build history from DB (all prior turns)
    history = [{"role": m.role, "content": m.content} for m in s.messages[:-1]]

    # Auto-title from first user message
    if len(s.messages) == 1:
        s.title = body.message[:60] + ("…" if len(body.message) > 60 else "")
        db.commit()

    comp = db.get(Component, s.component_id) if s.component_id else None

    async def gen():
        accumulated = ""
        try:
            if comp:
                stream = stream_component_answer(db, comp, body.message, history)
            else:
                stream = stream_answer(db, body.message, history)
            async for delta in stream:
                accumulated += delta
                yield delta
        except Exception as e:
            accumulated += f"\n\n[error: {e}]"
            yield f"\n\n[error: {e}]"
        finally:
            # Persist assistant turn
            ai_msg = ChatSessionMessage(session_id=sid, role="assistant", content=accumulated)
            db.add(ai_msg)
            from datetime import datetime
            s.updated_at = datetime.utcnow()
            db.commit()

    return StreamingResponse(gen(), media_type="text/plain; charset=utf-8")
