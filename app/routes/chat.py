"""Streaming chat endpoint. History is sent from the client each turn (stateless server)."""
from __future__ import annotations

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ..assistant import stream_answer
from ..database import get_db

router = APIRouter(prefix="/api/chat", tags=["chat"])


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str
    history: list[ChatMessage] = []


@router.post("/stream")
async def chat_stream_endpoint(req: ChatRequest, db: Session = Depends(get_db)):
    async def gen():
        try:
            async for delta in stream_answer(
                db, req.message, [m.model_dump() for m in req.history]
            ):
                yield delta
        except Exception as e:
            yield f"\n\n[error: {e}]"

    return StreamingResponse(gen(), media_type="text/plain; charset=utf-8")
