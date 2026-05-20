"""Embeddings via sentence-transformers (loaded lazily, runs locally on CPU)."""
from __future__ import annotations

import json
from functools import lru_cache
from typing import Iterable

import numpy as np

from .config import settings


@lru_cache(maxsize=1)
def _model():
    # Lazy import — sentence-transformers pulls torch which is heavy.
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(settings.EMBEDDING_MODEL)


def embed(texts: list[str]) -> np.ndarray:
    if not texts:
        return np.zeros((0, 384), dtype=np.float32)
    vecs = _model().encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(vecs, dtype=np.float32)


def embed_one(text: str) -> np.ndarray:
    return embed([text])[0]


def to_json(vec: np.ndarray) -> str:
    return json.dumps(vec.astype(float).tolist())


def from_json(s: str) -> np.ndarray:
    return np.asarray(json.loads(s), dtype=np.float32)


def cosine_topk(query: np.ndarray, matrix: np.ndarray, k: int) -> list[tuple[int, float]]:
    """matrix rows are L2-normalized; query is L2-normalized → dot = cosine."""
    if matrix.size == 0:
        return []
    sims = matrix @ query
    idx = np.argsort(-sims)[:k]
    return [(int(i), float(sims[i])) for i in idx]
