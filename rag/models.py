from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ProductEntity:
    manufacturer: str
    product_name: str
    model: str
    hardware_version: str = ""
    revision: str = ""
    confidence: float = 1.0


@dataclass
class SourceMetadata:
    source_id: str
    product_id: str
    manufacturer: str
    model: str
    hardware_version: str
    source_name: str
    source_type: str
    source_url: str = ""
    status: str = "PENDING"


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
    hardware_version: str | None = None
    used_search: bool = False
    used_memory: bool = False
    interaction_id: str | None = None
    clarification_needed: bool = False
