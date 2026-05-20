"""Chat assistant: RAG over the local library, with explicit labeling when
the answer leans on general LLM knowledge rather than the user's database."""
from __future__ import annotations

from typing import AsyncIterator

from sqlalchemy.orm import Session

from .llm import chat_stream
from .search import rag_chunks, search_components

SYSTEM_PROMPT = """You are a personal hardware engineering assistant for an electronics/IC developer.

You have access to the user's local component library via retrieved context.

RESPONSE RULES — follow these exactly:
1. PREFER recommending components that appear in the retrieved LIBRARY CONTEXT below.
   When you do, cite them inline like [PART_NUMBER].
2. If the library does not contain a suitable part, you MAY suggest well-known parts
   from general knowledge — but you MUST prefix that section with the literal line:
   "⚠️ Not in your library — general suggestions:"
3. Be concrete: give part numbers, key parameters (Vin, Iq, BW, slew rate, etc.),
   and brief justification. Mention package when relevant.
4. For circuit design questions, outline the architecture in 3-6 short steps and
   list the building blocks (with parts where possible).
5. Call out critical pitfalls (absolute max ratings, layout, decoupling, thermal).
6. Be concise. Engineer-to-engineer. No filler.
"""

COMPONENT_SYSTEM_PROMPT = """You are a personal hardware engineering assistant with deep knowledge of electronic components, circuits, and datasheets.

You are currently viewing the component: {part_number} by {manufacturer} ({category}).

COMPONENT DATA:
{component_data}

DATASHEET EXCERPTS:
{datasheet_context}

RESPONSE RULES:
1. Answer specifically about this component. Reference its actual specs, pin names, and datasheet info.
2. For pinout/circuit requests, list every relevant pin with number, name, and function.
3. For design questions, give concrete values and formulas using this part's actual parameters.
4. Call out pitfalls specific to this part (supply voltage limits, input common-mode, layout notes).
5. If asked for a schematic or circuit, describe it step by step with actual component values.
6. Be concise and engineer-to-engineer. No filler.
"""


def _format_context(hits) -> str:
    if not hits:
        return "(library is empty or no relevant matches found)"
    blocks = []
    seen = set()
    for h in hits:
        c = h.component
        key = c.id
        if key in seen:
            continue
        seen.add(key)
        blocks.append(
            f"[{c.part_number}] mfr={c.manufacturer or '?'} category={c.category or '?'} "
            f"package={c.package or '?'}\n"
            f"  desc: {c.description}\n"
            f"  summary: {c.summary}\n"
            f"  tags: {c.tags}\n"
            f"  specs: {c.specs}\n"
            f"  excerpt (p{h.chunk.page if hasattr(h, 'chunk') else '-'}): "
            f"{(h.chunk.text[:500] + '…') if hasattr(h, 'chunk') and len(h.chunk.text) > 500 else (h.chunk.text if hasattr(h, 'chunk') else '')}"
        )
    return "\n\n".join(blocks)


def build_messages(db: Session, user_msg: str, history: list[dict]) -> list[dict]:
    # Hybrid retrieval: keyword hits (component-level) + RAG (chunk-level)
    kw_hits = search_components(db, user_msg, limit=5)
    rag = rag_chunks(db, user_msg)

    ctx_blocks = []
    if kw_hits:
        ctx_blocks.append("KEYWORD MATCHES:\n" + "\n".join(
            f"[{h.component.part_number}] {h.component.category} — {h.component.description}"
            for h in kw_hits
        ))
    if rag:
        ctx_blocks.append("DATASHEET EXCERPTS:\n" + _format_context(rag))

    ctx = "\n\n".join(ctx_blocks) if ctx_blocks else "(library is empty or no matches)"

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    # Trim history to last 10 turns
    for m in history[-10:]:
        if m.get("role") in ("user", "assistant") and m.get("content"):
            messages.append({"role": m["role"], "content": m["content"]})
    messages.append({
        "role": "user",
        "content": f"LIBRARY CONTEXT:\n{ctx}\n\n---\nUSER QUESTION:\n{user_msg}",
    })
    return messages


def build_component_messages(db: Session, component, user_msg: str, history: list[dict]) -> list[dict]:
    """Build messages scoped to a specific component — richer context, component-aware system prompt."""
    import json

    # Pull datasheet chunks for this specific component filtered by user query
    rag = rag_chunks(db, user_msg, component_id=component.id)
    ds_ctx = _format_context(rag) if rag else "(no matching datasheet excerpts)"

    component_data = (
        f"Part: {component.part_number}\n"
        f"Manufacturer: {component.manufacturer}\n"
        f"Category: {component.category}\n"
        f"Package: {component.package}\n"
        f"Description: {component.description}\n"
        f"Summary: {component.summary}\n"
        f"Features:\n{component.features}\n"
        f"Warnings:\n{component.warnings}\n"
        f"Applications:\n{component.applications}\n"
        f"Specs: {json.dumps(component.specs, indent=2)}\n"
        f"Tags: {component.tags}"
    )

    system = COMPONENT_SYSTEM_PROMPT.format(
        part_number=component.part_number,
        manufacturer=component.manufacturer or "unknown",
        category=component.category or "unknown",
        component_data=component_data,
        datasheet_context=ds_ctx,
    )

    messages = [{"role": "system", "content": system}]
    for m in history[-10:]:
        if m.get("role") in ("user", "assistant") and m.get("content"):
            messages.append({"role": m["role"], "content": m["content"]})
    messages.append({"role": "user", "content": user_msg})
    return messages


async def stream_answer(db: Session, user_msg: str, history: list[dict]) -> AsyncIterator[str]:
    msgs = build_messages(db, user_msg, history)
    async for delta in chat_stream(msgs, temperature=0.3, max_tokens=1500):
        yield delta


async def stream_component_answer(db: Session, component, user_msg: str, history: list[dict]) -> AsyncIterator[str]:
    """Stream answer scoped to a specific component."""
    msgs = build_component_messages(db, component, user_msg, history)
    async for delta in chat_stream(msgs, temperature=0.3, max_tokens=2000):
        yield delta

