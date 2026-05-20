"""Thin async client for LM Studio's OpenAI-compatible /v1 endpoint."""
from __future__ import annotations

import json
from typing import AsyncIterator, Optional

import httpx

from .config import settings


class LLMError(RuntimeError):
    pass


async def _post_chat(payload: dict) -> dict:
    async with httpx.AsyncClient(timeout=180.0) as client:
        r = await client.post(f"{settings.LMSTUDIO_BASE_URL}/chat/completions", json=payload)
        if r.status_code >= 400:
            # Surface the real LM Studio error body — without this you just see "400 Bad Request".
            body = r.text
            raise LLMError(f"LM Studio {r.status_code}: {body[:500]}")
        return r.json()


async def chat(messages: list[dict], temperature: float = 0.2, max_tokens: int = 1500,
               json_mode: bool = False, model: Optional[str] = None) -> str:
    """Non-streaming chat completion. Falls back gracefully if json_mode is unsupported."""
    payload = {
        "model": model or settings.LMSTUDIO_MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }
    if json_mode:
        # LM Studio rejects {"type":"json_object"} (it wants "json_schema" or "text").
        # We skip response_format entirely and reinforce JSON via the prompt — extract_json()
        # handles markdown fences and truncation.
        if messages and messages[0].get("role") == "system":
            messages = [{**messages[0], "content": messages[0]["content"]
                         + "\n\nIMPORTANT: Respond with ONLY a single valid JSON object. No prose, no code fences."}] + messages[1:]
        else:
            messages = [{"role": "system",
                         "content": "Respond with ONLY a single valid JSON object. No prose, no code fences."}] + messages
        payload["messages"] = messages

    try:
        data = await _post_chat(payload)
        return data["choices"][0]["message"]["content"]
    except LLMError as e:
        # If json_mode caused a 400, retry without it. Many local models (gemma in LM Studio)
        # don't accept response_format. extract_json() will still recover the JSON.
        msg = str(e)
        if json_mode and "400" in msg:
            payload.pop("response_format", None)
            # Reinforce JSON via prompt instead.
            if messages and messages[0].get("role") == "system":
                messages = [
                    {**messages[0],
                     "content": messages[0]["content"]
                     + "\n\nIMPORTANT: Respond with ONLY a single valid JSON object. No prose, no code fences."}
                ] + messages[1:]
            else:
                messages = [{"role": "system",
                             "content": "Respond with ONLY a single valid JSON object. No prose, no code fences."}] + messages
            payload["messages"] = messages
            data = await _post_chat(payload)
            return data["choices"][0]["message"]["content"]
        raise
    except httpx.HTTPError as e:
        raise LLMError(f"LM Studio request failed: {e}") from e


async def chat_stream(messages: list[dict], temperature: float = 0.3, max_tokens: int = 1500,
                      model: Optional[str] = None) -> AsyncIterator[str]:
    """Streams content deltas from /v1/chat/completions."""
    payload = {
        "model": model or settings.LMSTUDIO_MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": True,
    }
    async with httpx.AsyncClient(timeout=None) as client:
        async with client.stream("POST", f"{settings.LMSTUDIO_BASE_URL}/chat/completions",
                                 json=payload) as r:
            if r.status_code >= 400:
                body = (await r.aread()).decode("utf-8", errors="replace")
                raise LLMError(f"LM Studio {r.status_code}: {body[:500]}")
            async for line in r.aiter_lines():
                if not line or not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    obj = json.loads(data)
                    delta = obj["choices"][0].get("delta", {}).get("content")
                    if delta:
                        yield delta
                except Exception:
                    continue


def extract_json(text: str) -> dict:
    """Best-effort extraction of a JSON object from LLM output. Tolerates markdown
    fences and truncation (closes open strings + braces so partial output is still usable)."""
    text = text.strip()
    if text.startswith("```"):
        # strip first fence line and trailing fence
        text = text.split("\n", 1)[1] if "\n" in text else text.strip("`")
        if text.endswith("```"):
            text = text[:-3]
    start = text.find("{")
    if start == -1:
        return {}
    text = text[start:]
    end = text.rfind("}")
    candidate = text[:end + 1] if end != -1 else text
    try:
        return json.loads(candidate)
    except Exception:
        pass
    # Recovery: walk the string tracking braces/quotes, close anything left open.
    buf = []
    in_str = False
    escape = False
    depth = 0
    last_valid = 0
    for i, ch in enumerate(text):
        buf.append(ch)
        if escape:
            escape = False
            continue
        if ch == "\\" and in_str:
            escape = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                last_valid = i + 1
    repaired = text[:last_valid] if last_valid else "".join(buf)
    if not last_valid:
        if in_str:
            repaired += '"'
        repaired += "}" * max(depth, 0)
    try:
        return json.loads(repaired)
    except Exception:
        return {}
