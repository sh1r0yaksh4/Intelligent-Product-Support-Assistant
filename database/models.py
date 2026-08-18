from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class UserRecord:
    id: str
    username: str
    password_hash: str
    role: str
    created_at: str = ""


@dataclass
class ProductRecord:
    id: str
    manufacturer: str
    name: str
    model: str
    hardware_version: str = ""
    description: str = ""
    created_at: str = ""


@dataclass
class SourceRecord:
    id: str
    product_id: str
    source_name: str
    source_type: str
    source_url: str = ""
    file_path: str = ""
    status: str = "PENDING"
    chunk_count: int = 0
    created_at: str = ""


@dataclass
class InteractionRecord:
    id: str
    user_id: str
    product_id: str
    product_name: str
    hardware_version: str
    question: str
    answer: str
    citations_json: str
    escalated: int
    used_search: int
    review_status: str
    created_at: str = ""
