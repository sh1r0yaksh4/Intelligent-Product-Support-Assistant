from __future__ import annotations

import json
import os
from typing import Any

from google import genai
from google.genai import types

from .config import EMBEDDING_DIMENSIONS, EMBEDDING_MODEL, GENERATION_MODEL, SEARCH_MODEL
from .models import Citation, ProductEntity


class GeminiUnavailable(RuntimeError):
    pass


def _client() -> genai.Client:
    if not os.getenv("GEMINI_API_KEY"):
        raise GeminiUnavailable("GEMINI_API_KEY is not configured. Copy .env.example to .env and add a key.")
    return genai.Client(api_key=os.environ["GEMINI_API_KEY"])


def embed_texts(texts: list[str], task_type: str) -> list[list[float]]:
    if not texts:
        return []
    response = _client().models.embed_content(
        model=EMBEDDING_MODEL,
        contents=texts,
        config=types.EmbedContentConfig(
            task_type=task_type,
            output_dimensionality=EMBEDDING_DIMENSIONS,
        ),
    )
    return [embedding.values for embedding in response.embeddings]


def generate_grounded_answer(prompt: str) -> str:
    response = _client().models.generate_content(
        model=GENERATION_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(temperature=0.1, max_output_tokens=700),
    )
    return (response.text or "NOT_FOUND: I could not produce a grounded answer.").strip()


def _citation_from_annotation(annotation: Any) -> Citation | None:
    url = getattr(annotation, "url", None)
    if not url:
        return None
    return Citation(title=getattr(annotation, "title", None) or url, url=url, source_type="web")


def search_grounded_answer(prompt: str) -> tuple[str, list[Citation]]:
    """Use Google Search grounding and return only API-provided citations."""
    interaction = _client().interactions.create(
        model=SEARCH_MODEL,
        input=prompt,
        tools=[{"type": "google_search"}],
    )
    citations: list[Citation] = []
    for step in getattr(interaction, "steps", []) or []:
        if getattr(step, "type", "") != "model_output":
            continue
        for block in getattr(step, "content", []) or []:
            for annotation in getattr(block, "annotations", []) or []:
                citation = _citation_from_annotation(annotation)
                if citation and citation.url not in {item.url for item in citations}:
                    citations.append(citation)
    return (getattr(interaction, "output_text", "") or "").strip(), citations


def identify_product_structured(
    message: str, previous_product: str | None = None, previous_version: str | None = None
) -> ProductEntity:
    """Extract structured product name, model, hardware version, and revision from message using Gemini JSON output."""
    prompt = f"""Extract consumer product entity details mentioned in this customer message.
Return JSON with EXACTLY these fields:
- manufacturer: string (e.g. "TP-Link", "Sony", "Netgear", or "" if unknown)
- product_name: string (e.g. "TP-Link Archer AX21", "Sony WH-1000XM5", or "" if unknown)
- model: string (e.g. "AX21", "WH-1000XM5", or "" if unknown)
- hardware_version: string (e.g. "V1", "V2", "v2.0", or "" if unknown)
- revision: string (e.g. "Rev A", or "")
- confidence: float between 0.0 and 1.0

Previous active product: {previous_product or 'none'}
Previous active hardware version: {previous_version or 'none'}
Message: {message}"""

    try:
        response = _client().models.generate_content(
            model=GENERATION_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(response_mime_type="application/json", temperature=0.0),
        )
        data = json.loads(response.text or "{}")
        p_name = data.get("product_name") or previous_product or ""
        h_ver = data.get("hardware_version") or previous_version or ""
        return ProductEntity(
            manufacturer=data.get("manufacturer") or "",
            product_name=p_name,
            model=data.get("model") or "",
            hardware_version=h_ver,
            revision=data.get("revision") or "",
            confidence=float(data.get("confidence", 0.8)),
        )
    except Exception:
        return ProductEntity(
            manufacturer="",
            product_name=previous_product or "",
            model="",
            hardware_version=previous_version or "",
            revision="",
            confidence=0.5,
        )
