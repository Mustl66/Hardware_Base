"""Retrieval: FTS5 keyword search + vector similarity over datasheet chunks."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from sqlalchemy import text
from sqlalchemy.orm import Session

from .config import settings
from .database import Component, DatasheetChunk
from .embeddings import embed_one, from_json


@dataclass
class ComponentHit:
    component: Component
    score: float
    snippet: str = ""


@dataclass
class ChunkHit:
    component: Component
    chunk: DatasheetChunk
    score: float


def _fts_query(q: str) -> str:
    """Escape user input into a safe FTS5 query — prefix-match each token, OR them."""
    tokens = [t for t in (w.strip('"\'`*()[]{}:;,') for w in q.split()) if t]
    if not tokens:
        return '""'
    return " OR ".join(f'"{t}"*' for t in tokens)


def search_components(db: Session, q: str, category: Optional[str] = None,
                      manufacturer: Optional[str] = None, limit: int = 50) -> list[ComponentHit]:
    """Keyword search via FTS5; falls back to LIKE if FTS query is empty."""
    if q.strip():
        fts = _fts_query(q)
        rows = db.execute(text("""
            SELECT c.id, bm25(components_fts) AS score
            FROM components_fts
            JOIN components c ON c.id = components_fts.rowid
            WHERE components_fts MATCH :q
              AND (:cat = '' OR c.category = :cat)
              AND (:mfr = '' OR c.manufacturer = :mfr)
            ORDER BY score
            LIMIT :lim
        """), {"q": fts, "cat": category or "", "mfr": manufacturer or "", "lim": limit}).all()
    else:
        rows = db.execute(text("""
            SELECT id, 0 AS score FROM components
            WHERE (:cat = '' OR category = :cat)
              AND (:mfr = '' OR manufacturer = :mfr)
            ORDER BY created_at DESC
            LIMIT :lim
        """), {"cat": category or "", "mfr": manufacturer or "", "lim": limit}).all()

    if not rows:
        return []
    ids = [r[0] for r in rows]
    comps = {c.id: c for c in db.query(Component).filter(Component.id.in_(ids)).all()}
    return [ComponentHit(component=comps[i], score=float(s)) for (i, s) in rows if i in comps]


def rag_chunks(db: Session, query: str, k: Optional[int] = None,
               component_id: Optional[int] = None) -> list[ChunkHit]:
    """Vector search over datasheet chunks. Optionally scope to one component."""
    k = k or settings.RAG_TOP_K
    q_vec = embed_one(query)

    qry = db.query(DatasheetChunk)
    if component_id is not None:
        qry = qry.filter(DatasheetChunk.component_id == component_id)
    chunks = qry.all()
    if not chunks:
        return []

    matrix = np.vstack([from_json(c.embedding_json) for c in chunks])
    sims = matrix @ q_vec
    order = np.argsort(-sims)[:k]

    comp_ids = {chunks[int(i)].component_id for i in order}
    comps = {c.id: c for c in db.query(Component).filter(Component.id.in_(comp_ids)).all()}
    out: list[ChunkHit] = []
    for i in order:
        ch = chunks[int(i)]
        out.append(ChunkHit(component=comps[ch.component_id], chunk=ch, score=float(sims[int(i)])))
    return out
