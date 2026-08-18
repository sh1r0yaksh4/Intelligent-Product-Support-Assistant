from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
PRODUCTS_DIR = DATA_DIR / "products"
CHROMA_DIR = ROOT_DIR / "chroma_store"
INTERACTIONS_FILE = DATA_DIR / "interactions.jsonl"

load_dotenv(ROOT_DIR / ".env")

DOCUMENT_COLLECTION = "product_documents"
MEMORY_COLLECTION = "approved_interactions"
EMBEDDING_DIMENSIONS = 768
TOP_K = 5
MIN_DOCUMENT_SIMILARITY = float(os.getenv("MIN_DOCUMENT_SIMILARITY", "0.42"))

GENERATION_MODEL = os.getenv("GEMINI_GENERATION_MODEL", "gemini-2.5-flash")
SEARCH_MODEL = os.getenv("GEMINI_SEARCH_MODEL", "gemini-2.5-flash")
EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")
SUPPORT_URL = "https://www.google.com/search?q=official+manufacturer+support"


def ensure_directories() -> None:
    for path in (DATA_DIR, PRODUCTS_DIR, CHROMA_DIR):
        path.mkdir(parents=True, exist_ok=True)


def slugify(value: str) -> str:
    clean = "".join(char.lower() if char.isalnum() else "-" for char in value)
    return "-".join(part for part in clean.split("-") if part)[:80] or "general"

