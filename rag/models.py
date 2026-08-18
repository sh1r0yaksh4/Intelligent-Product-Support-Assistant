from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class DocumentChunk:
    text: str
    metadata: dict[str, Any]


@dataclass
class RetrievedChunk:
    text: str
    metadata: dict[str, Any]
    score: float


@dataclass
class Citation:
    title: str
    url: str
    section: str | None = None
    source_type: str = "document"


@dataclass
class ChatResult:
    answer: str
    citations: list[Citation] = field(default_factory=list)
    escalated: bool = False
    product_name: str | None = None
    used_search: bool = False
    interaction_id: str | None = None

