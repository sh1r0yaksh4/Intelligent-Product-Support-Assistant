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


_CLIENT_INSTANCE: tuple[str, genai.Client] | None = None


def _client() -> genai.Client:
    global _CLIENT_INSTANCE
    key = os.getenv("GEMINI_API_KEY", "").strip()
    if not key:
        raise GeminiUnavailable("GEMINI_API_KEY is not configured. Copy .env.example to .env and add a key.")
    if _CLIENT_INSTANCE is None or _CLIENT_INSTANCE[0] != key:
        _CLIENT_INSTANCE = (key, genai.Client(api_key=key))
    return _CLIENT_INSTANCE[1]


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
    """Use Google Search grounding and return only API-provided citations.

    Returns empty results on rate-limit or quota errors so the caller
    can fall through to escalation instead of crashing.
    """
    try:
        interaction = _client().interactions.create(
            model=SEARCH_MODEL,
            input=prompt,
            tools=[{"type": "google_search"}],
        )
    except Exception as exc:
        exc_str = str(exc).lower()
        if "429" in exc_str or "quota" in exc_str or "rate" in exc_str or "resource_exhausted" in exc_str:
            return "", []
        raise
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


def assess_conversation(
    message: str,
    history: list[dict[str, str]] | None = None,
    active_product: str | None = None,
    active_version: str | None = None,
) -> dict:
    """Assess whether there is enough context to search and answer or if a clarification question is needed."""
    prompt = f"""You are a product support conversation evaluator.
Analyze the user's message and prior conversation history. Decide whether you have enough specific product and issue context to search documentation and provide a helpful answer, or if you should ask a single short clarifying question.

Return JSON with EXACTLY these fields:
- "action": "answer" or "clarify"
- "product_name": string (extracted product name such as "TP-Link Archer AX21", "Sony WH-1000XM5", or "" if unknown)
- "manufacturer": string (e.g. "TP-Link", "Sony", or "")
- "model": string (e.g. "AX21", "WH-1000XM5", or "")
- "hardware_version": string (e.g. "V1", "V2", or "")
- "confidence": float between 0.0 and 1.0
- "clarification_question": string (if action is "clarify", ask exactly ONE concise, conversational follow-up question under 30 words; if action is "answer", set to "")
- "reasoning": string (brief explanation)

Guidelines:
1. If the user named a specific product (or it was established earlier in history/active product) and described a specific question/problem -> action = "answer".
2. If the product is unknown and cannot be inferred from history -> action = "clarify", ask which product (brand and model) they are using.
3. If the product is known but the problem description is completely ambiguous or empty (e.g. "it doesn't work", "help") -> action = "clarify", ask for details about the specific issue.
4. If the message is a greeting or general remark without a product or issue -> action = "clarify", greet politely and ask how you can help with their product.
5. Keep clarification questions friendly, direct, and under 30 words. Ask only ONE question at a time.

Active product workspace: {active_product or 'none'}
Active hardware version: {active_version or 'none'}
Recent conversation history: {history[-6:] if history else []}
Latest message: {message}"""

    try:
        response = _client().models.generate_content(
            model=GENERATION_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(response_mime_type="application/json", temperature=0.0),
        )
        data = json.loads(response.text or "{}")
        action = data.get("action", "answer").lower()
        if action not in {"answer", "clarify"}:
            action = "answer"

        return {
            "action": action,
            "product_name": data.get("product_name") or active_product or "",
            "manufacturer": data.get("manufacturer") or "",
            "model": data.get("model") or "",
            "hardware_version": data.get("hardware_version") or active_version or "",
            "confidence": float(data.get("confidence", 0.8)),
            "clarification_question": data.get("clarification_question", "").strip(),
            "reasoning": data.get("reasoning", ""),
        }
    except Exception:
        return {
            "action": "answer",
            "product_name": active_product or "",
            "manufacturer": "",
            "model": "",
            "hardware_version": active_version or "",
            "confidence": 0.5,
            "clarification_question": "",
            "reasoning": "Fallback to answer pipeline on assessment failure",
        }
