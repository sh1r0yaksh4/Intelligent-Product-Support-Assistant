from __future__ import annotations

import json
import os
from typing import Any

from google import genai
from google.genai import types

from .config import EMBEDDING_DIMENSIONS, EMBEDDING_MODEL, GENERATION_MODEL, SEARCH_MODEL
from .models import Citation


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


def identify_product(message: str, previous_product: str | None = None) -> str | None:
    """Best-effort extraction; an uncertain name never blocks research."""
    prompt = f"""Extract the most specific consumer product name/model mentioned in this message.
Return JSON only with one field named product_name. Use null when no product is identifiable.
Previous active product: {previous_product or 'none'}
Message: {message}"""
    response = _client().models.generate_content(
        model=GENERATION_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(response_mime_type="application/json", temperature=0),
    )
    try:
        return json.loads(response.text or "{}").get("product_name") or previous_product
    except json.JSONDecodeError:
        return previous_product

