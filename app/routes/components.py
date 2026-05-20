"""Component CRUD: upload, list, delete."""
from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, RedirectResponse
from sqlalchemy.orm import Session

from ..database import Component, get_db
from ..ingest import ingest_pdf

router = APIRouter(prefix="/api/components", tags=["components"])


@router.post("/upload")
async def upload_datasheet(
    file: UploadFile = File(...),
    part_number_hint: str = Form(""),
    db: Session = Depends(get_db),
):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are supported.")

    # Stage to temp first; ingest will move into datasheet dir.
    tmp = Path(tempfile.gettempdir()) / file.filename
    with open(tmp, "wb") as f:
        f.write(await file.read())

    try:
        comp = await ingest_pdf(
            db,
            upload_filename=file.filename,
            src_path=tmp,
            hint_part_number=(part_number_hint or None),
        )
    except Exception as e:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        raise HTTPException(500, f"Ingestion failed: {e}")

    return RedirectResponse(url=f"/components/{comp.id}", status_code=303)


@router.delete("/{cid}")
def delete_component(cid: int, db: Session = Depends(get_db)):
    comp = db.get(Component, cid)
    if not comp:
        raise HTTPException(404, "Not found")
    db.delete(comp)
    db.commit()
    return JSONResponse({"ok": True})
