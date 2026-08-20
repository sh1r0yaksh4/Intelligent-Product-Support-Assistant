from __future__ import annotations

import base64
import hashlib
import json
import logging
import math
import os
import re
import time
from typing import Any, Callable, TypeVar

import requests

from .config import (
    EMBEDDING_DIMENSIONS,
    GROK_API_BASE,
    GROK_GENERATION_MODEL,
    GROK_SEARCH_MODEL,
    GROK_VISION_MODEL,
)
from .models import Citation, ProductEntity

logger = logging.getLogger(__name__)

T = TypeVar("T")


class GrokUnavailable(RuntimeError):
    pass


class GrokRateLimited(RuntimeError):
    pass


# Backward compatibility aliases
GeminiUnavailable = GrokUnavailable
GeminiRateLimited = GrokRateLimited


def _get_api_key() -> str:
    key = (
        os.getenv("GROK_API_KEY", "").strip()
        or os.getenv("GROQ_API_KEY", "").strip()
        or os.getenv("XAI_API_KEY", "").strip()
    )
    if not key:
        raise GrokUnavailable(
            "API Key is not configured. Copy .env.example to .env and add your key."
        )
    return key


def _get_api_base() -> str:
    key = _get_api_key()
    explicit_base = os.getenv("GROK_API_BASE", "").strip()
    if explicit_base and explicit_base != "https://api.x.ai/v1":
        return explicit_base.rstrip("/")
    if key.startswith("gsk_"):
        return "https://api.groq.com/openai/v1"
    return (os.getenv("GROK_API_BASE", "").strip() or GROK_API_BASE).rstrip("/")


def _call_with_retry(fn: Callable[[], T], max_retries: int = 2, initial_delay: float = 1.5) -> T:
    """Execute API call with exponential backoff for rate limits and transient server errors."""
    delay = initial_delay
    last_exc: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except Exception as exc:
            last_exc = exc
            exc_str = str(exc).lower()
            is_rate_limit = (
                "429" in exc_str
                or "rate" in exc_str
                or "quota" in exc_str
                or "resource_exhausted" in exc_str
            )
            is_transient = (
                "503" in exc_str
                or "500" in exc_str
                or "502" in exc_str
                or "504" in exc_str
                or "unavailable" in exc_str
                or "timeout" in exc_str
            )
            if is_rate_limit or is_transient:
                if attempt < max_retries:
                    logger.warning(
                        f"Transient API condition ({exc_str[:60]}). Waiting {delay:.1f}s (attempt {attempt + 1}/{max_retries})..."
                    )
                    time.sleep(delay)
                    delay = min(delay * 2.0, 6.0)
                else:
                    if is_rate_limit:
                        raise GrokRateLimited("API rate limit reached. Please wait a few seconds and try again.")
                    raise exc
            else:
                raise exc
    if last_exc:
        raise last_exc
    raise RuntimeError("Unexpected retry loop termination")


def _chat_completion(
    messages: list[dict[str, Any]],
    model: str | None = None,
    temperature: float = 0.1,
    max_tokens: int = 700,
    response_format: dict[str, str] | None = None,
) -> str:
    """Execute OpenAI-compatible chat completion request against Grok/Groq API."""
    api_key = _get_api_key()
    api_base = _get_api_base()
    url = f"{api_base}/chat/completions"
    
    chosen_model = model or os.getenv("GROK_GENERATION_MODEL")
    if not chosen_model:
        chosen_model = "openai/gpt-oss-120b" if api_key.startswith("gsk_") else GROK_GENERATION_MODEL

    payload: dict[str, Any] = {
        "model": chosen_model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if response_format:
        payload["response_format"] = response_format

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    def _request() -> str:
        response = requests.post(url, headers=headers, json=payload, timeout=45)
        if response.status_code == 429:
            raise GrokRateLimited("Grok API rate limit reached (HTTP 429).")
        if response.status_code >= 500:
            raise RuntimeError(f"Grok API Server Error (HTTP {response.status_code}): {response.text[:200]}")
        if response.status_code != 200:
            raise RuntimeError(f"Grok API Error (HTTP {response.status_code}): {response.text[:200]}")

        data = response.json()
        choices = data.get("choices") or []
        if not choices:
            return ""
        message = choices[0].get("message") or {}
        return message.get("content") or ""

    return _call_with_retry(_request)


def _local_embedding(text: str, dim: int = EMBEDDING_DIMENSIONS) -> list[float]:
    """Deterministic local normalized dense embedding for vector indexing and cosine retrieval."""
    vec = [0.0] * dim
    words = re.findall(r"\w+", (text or "").lower())
    if not words:
        return vec
    for i, w in enumerate(words):
        # 1-gram hash
        h1 = int(hashlib.sha256(w.encode("utf-8")).hexdigest(), 16)
        idx1 = h1 % dim
        sign1 = 1.0 if (h1 >> 16) & 1 else -1.0
        vec[idx1] += sign1 * 1.0

        # 2-gram context hash
        if i + 1 < len(words):
            bigram = f"{w}_{words[i+1]}"
            h2 = int(hashlib.sha256(bigram.encode("utf-8")).hexdigest(), 16)
            idx2 = h2 % dim
            sign2 = 1.0 if (h2 >> 16) & 1 else -1.0
            vec[idx2] += sign2 * 1.5

    norm = math.sqrt(sum(x * x for x in vec))
    if norm > 0:
        vec = [x / norm for x in vec]
    return vec


def embed_texts(texts: list[str], task_type: str = "RETRIEVAL_DOCUMENT") -> list[list[float]]:
    """Embed texts into 768-dimensional normalized dense vectors for ChromaDB."""
    if not texts:
        return []
    return [_local_embedding(t) for t in texts]


def generate_grounded_answer(prompt: str) -> str:
    """Generate answer grounded in provided documentation evidence using Grok."""
    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert technical product support assistant. "
                "Answer the user's inquiry thoroughly and helpfully using the provided evidence. "
                "Synthesize relevant details, specifications, troubleshooting steps, and guidance found in the document. "
                "Use clear headings, bullet points, or tables where appropriate."
            ),
        },
        {"role": "user", "content": prompt},
    ]
    try:
        text = _chat_completion(
            messages=messages,
            model=os.getenv("GROK_GENERATION_MODEL", GROK_GENERATION_MODEL),
            temperature=0.1,
            max_tokens=900,
        )
        return (text or "NOT_FOUND: I could not produce a grounded answer.").strip()
    except (GrokUnavailable, GrokRateLimited):
        raise
    except Exception as exc:
        logger.error(f"Grok grounded answer generation failed: {exc}")
        return "NOT_FOUND: I could not produce a grounded answer."


def search_grounded_answer(prompt: str) -> tuple[str, list[Citation]]:
    """Generate structured response with citations using Grok."""
    messages = [
        {
            "role": "system",
            "content": "You are a knowledgeable technical product support specialist. Answer accurately with structured points and official support guidance.",
        },
        {"role": "user", "content": prompt},
    ]
    try:
        text = _chat_completion(
            messages=messages,
            model=os.getenv("GROK_SEARCH_MODEL", GROK_SEARCH_MODEL),
            temperature=0.1,
            max_tokens=800,
        )
        if not text:
            return "", []

        citations: list[Citation] = []
        # Extract any markdown or raw URLs mentioned in response
        urls = re.findall(r"https?://[^\s\)\>]+", text)
        for url in urls[:2]:
            clean_url = url.rstrip(".,;:")
            domain = re.sub(r"^https?://(?:www\.)?", "", clean_url).split("/")[0]
            citations.append(Citation(title=f"Official {domain.capitalize()} Support", url=clean_url, source_type="web"))

        return text.strip(), citations
    except Exception as exc:
        logger.warning(f"Grok search answer fallback failed: {exc}")
        return "", []


def identify_product_structured(
    message: str, previous_product: str | None = None, previous_version: str | None = None
) -> ProductEntity:
    """Extract structured product name, model, hardware version, and revision from message using Grok JSON output."""
    prompt = f"""Extract consumer product entity details mentioned in this customer message.
Return JSON with EXACTLY these fields:
- "manufacturer": string (e.g. "TP-Link", "Sony", "Netgear", "Vivo", or "" if unknown)
- "product_name": string (e.g. "TP-Link Archer AX21", "Sony WH-1000XM5", "Vivo Smartphone", or "" if unknown)
- "model": string (e.g. "AX21", "WH-1000XM5", or "" if unknown)
- "hardware_version": string (e.g. "V1", "V2", "v2.0", or "" if unknown)
- "revision": string (e.g. "Rev A", or "")
- "confidence": float between 0.0 and 1.0

Previous active product: {previous_product or 'none'}
Previous active hardware version: {previous_version or 'none'}
Message: {message}"""

    messages = [
        {"role": "system", "content": "You extract structured product entities from customer messages and reply with JSON only."},
        {"role": "user", "content": prompt},
    ]

    try:
        text = _chat_completion(
            messages=messages,
            model=os.getenv("GROK_GENERATION_MODEL", GROK_GENERATION_MODEL),
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        json_match = re.search(r"\{.*\}", text, re.DOTALL)
        raw_json = json_match.group(0) if json_match else "{}"
        data = json.loads(raw_json)
        if not isinstance(data, dict):
            data = {}
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
- "product_name": string (extracted product name such as "TP-Link Archer AX21", "Vivo Smartphone", or "" if unknown)
- "manufacturer": string (e.g. "TP-Link", "Vivo", "Sony", or "")
- "model": string (e.g. "AX21", "V29", or "")
- "hardware_version": string (e.g. "V1", "V2", or "")
- "confidence": float between 0.0 and 1.0
- "clarification_question": string (if action is "clarify", ask exactly ONE concise, conversational follow-up question under 30 words; if action is "answer", set to "")
- "reasoning": string (brief explanation)

Active product workspace: {active_product or 'none'}
Active hardware version: {active_version or 'none'}
Recent conversation history: {history[-6:] if history else []}
Latest message: {message}"""

    messages = [
        {"role": "system", "content": "You evaluate customer support dialogues and return JSON decisions."},
        {"role": "user", "content": prompt},
    ]

    try:
        text = _chat_completion(
            messages=messages,
            model=os.getenv("GROK_GENERATION_MODEL", GROK_GENERATION_MODEL),
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        json_match = re.search(r"\{.*\}", text, re.DOTALL)
        raw_json = json_match.group(0) if json_match else "{}"
        data = json.loads(raw_json)
        if not isinstance(data, dict):
            data = {}
        action = str(data.get("action", "answer")).lower()
        if action not in {"answer", "clarify"}:
            action = "answer"

        return {
            "action": action,
            "product_name": data.get("product_name") or active_product or "",
            "manufacturer": data.get("manufacturer") or "",
            "model": data.get("model") or "",
            "hardware_version": data.get("hardware_version") or active_version or "",
            "confidence": float(data.get("confidence", 0.8)),
            "clarification_question": str(data.get("clarification_question", "")).strip(),
            "reasoning": str(data.get("reasoning", "")),
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


def analyze_hardware_image_grok(image_bytes: bytes, mime_type: str = "image/jpeg") -> dict[str, Any]:
    """Inspect a hardware photo (label sticker, LED indicators, ports) using Grok Vision."""
    prompt = """You are a technical hardware inspection specialist.
Analyze this photo of a device (hardware label sticker, LED status lights, physical ports, or product enclosure).
Extract and return JSON with EXACTLY these keys:
- "manufacturer": string (e.g. "TP-Link", "Sony", "Ecobee", or "" if not visible)
- "product_name": string (e.g. "Archer AX21", "WH-1000XM5", or "" if not visible)
- "model": string (exact model number found on label)
- "hardware_version": string (e.g. "V1", "V2.0", "Rev B", or "" if not visible)
- "serial_number": string (if visible, otherwise "")
- "led_status": string (description of any visible LED lights e.g. "Solid Green Power, Blinking Amber Internet", or "")
- "port_status": string (description of any plugged cables or port conditions, or "")
- "visual_summary": string (concise technical description of the hardware state shown in the image under 40 words)
- "confidence": float between 0.0 and 1.0
"""
    try:
        b64_image = base64.b64encode(image_bytes).decode("utf-8")
        data_url = f"data:{mime_type};base64,{b64_image}"

        messages = [
            {"role": "system", "content": "You are a technical hardware inspection specialist who inspects device images and outputs JSON."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            },
        ]
        text = _chat_completion(
            messages=messages,
            model=os.getenv("GROK_VISION_MODEL", GROK_VISION_MODEL),
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        json_match = re.search(r"\{.*\}", text, re.DOTALL)
        raw_json = json_match.group(0) if json_match else "{}"
        return json.loads(raw_json)
    except Exception as exc:
        logger.warning(f"Grok hardware image analysis failed: {exc}")
        return {
            "manufacturer": "",
            "product_name": "",
            "model": "",
            "hardware_version": "",
            "serial_number": "",
            "led_status": "",
            "port_status": "",
            "visual_summary": f"Could not inspect image: {exc}",
            "confidence": 0.0,
        }
