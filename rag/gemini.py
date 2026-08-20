"""Re-export module from rag.grok for backwards compatibility."""

from __future__ import annotations

from .grok import (
    GeminiRateLimited,
    GeminiUnavailable,
    GrokRateLimited,
    GrokUnavailable,
    analyze_hardware_image_grok,
    assess_conversation,
    embed_texts,
    generate_grounded_answer,
    identify_product_structured,
    search_grounded_answer,
)

__all__ = [
    "GrokUnavailable",
    "GrokRateLimited",
    "GeminiUnavailable",
    "GeminiRateLimited",
    "embed_texts",
    "generate_grounded_answer",
    "search_grounded_answer",
    "identify_product_structured",
    "assess_conversation",
    "analyze_hardware_image_grok",
]
