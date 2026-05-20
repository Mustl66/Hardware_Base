"""PDF text extraction with pdfplumber + chunking utilities."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Iterator

import pdfplumber

from .config import settings


def extract_text_per_page(pdf_path: Path, max_pages: int | None = None) -> list[tuple[int, str]]:
    """Returns list of (page_number_1based, text). Tables are flattened inline."""
    max_pages = max_pages or settings.MAX_PDF_PAGES
    pages: list[tuple[int, str]] = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages[:max_pages], start=1):
            txt = page.extract_text() or ""
            # Append simple table dumps so specs survive
            try:
                for tbl in page.extract_tables() or []:
                    flat = "\n".join(
                        " | ".join((c or "").strip() for c in row) for row in tbl if row
                    )
                    if flat.strip():
                        txt += "\n[TABLE]\n" + flat + "\n[/TABLE]\n"
            except Exception:
                pass
            txt = _clean(txt)
            if txt.strip():
                pages.append((i, txt))
    return pages


def _clean(s: str) -> str:
    s = s.replace("\x00", " ")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def chunk_text(pages: list[tuple[int, str]], chunk_size: int | None = None,
               overlap: int | None = None) -> Iterator[tuple[int, str]]:
    """Yield (page, chunk_text). Simple sliding window over concatenated page text."""
    chunk_size = chunk_size or settings.RAG_CHUNK_SIZE
    overlap = overlap or settings.RAG_CHUNK_OVERLAP
    for page, text in pages:
        if len(text) <= chunk_size:
            yield page, text
            continue
        i = 0
        while i < len(text):
            piece = text[i:i + chunk_size]
            yield page, piece
            if i + chunk_size >= len(text):
                break
            i += chunk_size - overlap


def head_text(pages: list[tuple[int, str]], chars: int = 8000) -> str:
    """First N characters across pages — used for LLM metadata extraction."""
    buf = []
    total = 0
    for _, t in pages:
        if total + len(t) > chars:
            buf.append(t[: chars - total])
            break
        buf.append(t)
        total += len(t)
    return "\n\n".join(buf)
