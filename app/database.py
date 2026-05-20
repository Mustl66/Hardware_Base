"""SQLAlchemy models + DB init. SQLite with FTS5 virtual table for full-text search."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from sqlalchemy import (
    Column, Integer, String, Text, DateTime, ForeignKey, create_engine, event, text
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker, Session

from .config import settings

Base = declarative_base()


class Component(Base):
    __tablename__ = "components"
    id = Column(Integer, primary_key=True)
    part_number = Column(String(128), index=True, nullable=False)
    manufacturer = Column(String(128), index=True, default="")
    category = Column(String(64), index=True, default="")        # e.g. op-amp, buck, mcu
    package = Column(String(64), default="")
    description = Column(Text, default="")
    summary = Column(Text, default="")                            # LLM-written tech summary
    features = Column(Text, default="")                           # bullet list
    warnings = Column(Text, default="")
    applications = Column(Text, default="")
    tags = Column(Text, default="")                               # comma-separated
    specs_json = Column(Text, default="{}")                       # flexible specs
    datasheet_filename = Column(String(255), default="")
    created_at = Column(DateTime, default=datetime.utcnow)

    chunks = relationship("DatasheetChunk", back_populates="component", cascade="all, delete-orphan")
    notes = relationship("ComponentNote", back_populates="component", cascade="all, delete-orphan", order_by="ComponentNote.created_at")

    @property
    def specs(self) -> dict:
        try:
            return json.loads(self.specs_json or "{}")
        except Exception:
            return {}


class DatasheetChunk(Base):
    """RAG chunk: text + embedding (stored as JSON list of floats)."""
    __tablename__ = "datasheet_chunks"
    id = Column(Integer, primary_key=True)
    component_id = Column(Integer, ForeignKey("components.id", ondelete="CASCADE"), index=True)
    page = Column(Integer, default=0)
    text = Column(Text, nullable=False)
    embedding_json = Column(Text, nullable=False)  # JSON list[float]
    component = relationship("Component", back_populates="chunks")


class ComponentNote(Base):
    """AI-generated (or user-edited) note saved against a component."""
    __tablename__ = "component_notes"
    id = Column(Integer, primary_key=True)
    component_id = Column(Integer, ForeignKey("components.id", ondelete="CASCADE"), index=True)
    question = Column(Text, nullable=False)        # the user question that produced this note
    content = Column(Text, nullable=False)         # AI answer (editable)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    component = relationship("Component", back_populates="notes")


class ChatSession(Base):
    """A persisted chat conversation (global or component-scoped)."""
    __tablename__ = "chat_sessions"
    id = Column(Integer, primary_key=True)
    title = Column(String(200), default="New Chat")
    component_id = Column(Integer, ForeignKey("components.id", ondelete="SET NULL"), nullable=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)
    messages = relationship(
        "ChatSessionMessage", back_populates="session",
        cascade="all, delete-orphan", order_by="ChatSessionMessage.id"
    )
    component = relationship("Component")


class ChatSessionMessage(Base):
    """One message turn in a ChatSession."""
    __tablename__ = "chat_session_messages"
    id = Column(Integer, primary_key=True)
    session_id = Column(Integer, ForeignKey("chat_sessions.id", ondelete="CASCADE"), index=True)
    role = Column(String(20), nullable=False)   # "user" | "assistant"
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    session = relationship("ChatSession", back_populates="messages")


# --- Engine ---
engine = create_engine(
    f"sqlite:///{settings.db_path_abs}",
    future=True,
    connect_args={"check_same_thread": False},
)


@event.listens_for(engine, "connect")
def _sqlite_pragmas(dbapi_conn, _):
    cur = dbapi_conn.cursor()
    cur.execute("PRAGMA journal_mode=WAL")
    cur.execute("PRAGMA foreign_keys=ON")
    cur.close()


SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)


def get_db() -> Session:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    Base.metadata.create_all(engine)
    # FTS5 virtual table mirroring searchable fields.
    with engine.begin() as conn:
        conn.execute(text("""
            CREATE VIRTUAL TABLE IF NOT EXISTS components_fts USING fts5(
                part_number, manufacturer, category, description, summary,
                features, applications, tags,
                content='components', content_rowid='id'
            )
        """))
        # Triggers to keep FTS in sync
        conn.execute(text("""
            CREATE TRIGGER IF NOT EXISTS components_ai AFTER INSERT ON components BEGIN
              INSERT INTO components_fts(rowid, part_number, manufacturer, category, description, summary, features, applications, tags)
              VALUES (new.id, new.part_number, new.manufacturer, new.category, new.description, new.summary, new.features, new.applications, new.tags);
            END;
        """))
        conn.execute(text("""
            CREATE TRIGGER IF NOT EXISTS components_ad AFTER DELETE ON components BEGIN
              INSERT INTO components_fts(components_fts, rowid, part_number, manufacturer, category, description, summary, features, applications, tags)
              VALUES('delete', old.id, old.part_number, old.manufacturer, old.category, old.description, old.summary, old.features, old.applications, old.tags);
            END;
        """))
        conn.execute(text("""
            CREATE TRIGGER IF NOT EXISTS components_au AFTER UPDATE ON components BEGIN
              INSERT INTO components_fts(components_fts, rowid, part_number, manufacturer, category, description, summary, features, applications, tags)
              VALUES('delete', old.id, old.part_number, old.manufacturer, old.category, old.description, old.summary, old.features, old.applications, old.tags);
              INSERT INTO components_fts(rowid, part_number, manufacturer, category, description, summary, features, applications, tags)
              VALUES (new.id, new.part_number, new.manufacturer, new.category, new.description, new.summary, new.features, new.applications, new.tags);
            END;
        """))
