"""Datasheet ingestion pipeline:
1. Save PDF
2. Extract text (pdfplumber) + chunk
3. Ask LLM to extract structured metadata from the head of the doc
4. Embed chunks and store
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Optional

from sqlalchemy.orm import Session

from .config import settings
from .database import Component, DatasheetChunk
from .embeddings import embed, to_json
from .llm import chat, extract_json
from .pdf_extract import chunk_text, extract_text_per_page, head_text


EXTRACTION_PROMPT = """You are a hardware engineering assistant analyzing an electronic component datasheet.

From the text below, extract a JSON object with EXACTLY these keys:
{
  "part_number": "primary part number (string)",
  "manufacturer": "company name (string)",
  "category": "one of: op-amp, comparator, adc, dac, mcu, mpu, ldo, buck, boost, buck-boost, charger, gate-driver, mosfet, bjt, diode, sensor, transceiver, memory, logic, clock, rf, passive, connector, other",
  "package": "package code if mentioned, else empty string",
  "description": "one-sentence plain-English description of what the chip does",
  "summary": "2-4 sentence technical summary covering function, key parameters, target applications",
  "features": "newline-separated bullet list of headline features (5-10 bullets, each starting with '- ')",
  "warnings": "newline-separated bullet list of critical design considerations, absolute-max ratings to respect, common pitfalls",
  "applications": "newline-separated bullet list of typical use cases",
  "tags": "comma-separated short tags (e.g. 'precision, low-noise, rail-to-rail, automotive')",
  "specs": { "free-form key/value pairs of important electrical specs you find, e.g. 'Vin_min':'4.5V', 'Iq':'25uA' }
}

Return ONLY the JSON object, no prose, no markdown fences.

DATASHEET TEXT:
---
{TEXT}
---
"""


async def ingest_pdf(db: Session, upload_filename: str, src_path: Path,
                     hint_part_number: Optional[str] = None) -> Component:
    # 1. Move file into datasheet dir with a clean name (we rename after part_number is known)
    settings.datasheet_dir_abs.mkdir(parents=True, exist_ok=True)
    staging = settings.datasheet_dir_abs / upload_filename
    if src_path != staging:
        shutil.move(str(src_path), str(staging))

    # 2. Extract text
    pages = extract_text_per_page(staging)
    if not pages:
        raise RuntimeError("Could not extract any text from the PDF.")

    head = head_text(pages, chars=8000)

    # 3. LLM metadata extraction
    prompt = EXTRACTION_PROMPT.replace("{TEXT}", head)
    raw = await chat(
        messages=[
            {"role": "system", "content": "You output strict JSON only."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.1,
        max_tokens=4000,
        json_mode=True,
    )
    meta = extract_json(raw) or {}

    part_number = (hint_part_number or meta.get("part_number") or staging.stem).strip()
    specs = meta.get("specs") if isinstance(meta.get("specs"), dict) else {}

    comp = Component(
        part_number=part_number,
        manufacturer=(meta.get("manufacturer") or "").strip(),
        category=(meta.get("category") or "other").strip().lower(),
        package=(meta.get("package") or "").strip(),
        description=(meta.get("description") or "").strip(),
        summary=(meta.get("summary") or "").strip(),
        features=(meta.get("features") or "").strip(),
        warnings=(meta.get("warnings") or "").strip(),
        applications=(meta.get("applications") or "").strip(),
        tags=(meta.get("tags") or "").strip(),
        specs_json=json.dumps(specs, ensure_ascii=False),
        datasheet_filename=staging.name,
    )
    db.add(comp)
    db.flush()  # get id

    # 4. Chunk + embed
    chunks = list(chunk_text(pages))
    if chunks:
        vecs = embed([c[1] for c in chunks])
        for (page, txt), vec in zip(chunks, vecs):
            db.add(DatasheetChunk(
                component_id=comp.id,
                page=page,
                text=txt,
                embedding_json=to_json(vec),
            ))
    db.commit()
    db.refresh(comp)
    return comp
