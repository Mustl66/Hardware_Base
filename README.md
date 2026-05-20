# Hardware Base

Personal AI hardware-engineering lab assistant. Local SQLite component library + LM Studio LLM.

## Setup (PyCharm)

1. Open this folder as a PyCharm project.
2. Create a venv (PyCharm: Settings → Project → Python Interpreter → Add → Virtualenv).
3. In the PyCharm terminal:
   ```
   pip install -r requirements.txt
   ```
4. Start LM Studio:
   - Load your model (default `google/gemma-4-e4b`)
   - Start the local server on `http://127.0.0.1:1234`
5. Right-click `main.py` → **Run 'main'**.
6. Open http://127.0.0.1:8000

## What you get

- **Library** (`/`) — searchable table of components. FTS5 keyword search + category filter.
- **Add Datasheet** (`/upload`) — drop a PDF; the app extracts part number, manufacturer,
  category, package, summary, features, warnings, applications, tags, and free-form specs,
  then embeds the full text for RAG.
- **Component detail** (`/components/{id}`) — all extracted fields side-by-side with the
  embedded datasheet PDF viewer.
- **Assistant** (`/chat`) — streaming chat grounded on your library. Suggestions from the
  database are cited with `[PART_NUMBER]`. Anything from general knowledge is prefixed
  with `⚠️ Not in your library — general suggestions:`.

## Configuration

Edit `.env`:
- `LMSTUDIO_BASE_URL` — defaults to `http://127.0.0.1:1234/v1`
- `LMSTUDIO_MODEL` — change to whatever model name LM Studio shows
- `EMBEDDING_MODEL` — sentence-transformers model (downloaded on first run, ~80MB)
- `RAG_TOP_K`, `RAG_CHUNK_SIZE` — retrieval tuning

## Project layout

```
Hardware_Base/
├── main.py                    ← PyCharm entry point
├── requirements.txt
├── .env
├── app/
│   ├── __init__.py            ← FastAPI factory
│   ├── config.py
│   ├── database.py            ← SQLAlchemy + FTS5
│   ├── llm.py                 ← LM Studio client
│   ├── embeddings.py          ← sentence-transformers
│   ├── pdf_extract.py         ← pdfplumber
│   ├── ingest.py              ← PDF → metadata + chunks
│   ├── search.py              ← FTS5 + vector RAG
│   ├── assistant.py           ← grounded chat
│   ├── routes/                ← pages, components, chat
│   └── templates/             ← Jinja2 HTML
├── static/app.css             ← dark engineering UI
└── data/
    ├── hardware.db            ← SQLite (created on first run)
    └── datasheets/            ← uploaded PDFs
```

## Notes

- First upload is slow: sentence-transformers downloads its model. Subsequent uploads:
  ~10-30s per datasheet depending on length and LLM speed.
- The assistant streams tokens directly from LM Studio.
- Single-user, localhost-only. No auth.
