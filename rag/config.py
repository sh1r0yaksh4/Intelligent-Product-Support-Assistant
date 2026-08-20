from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
PRODUCTS_DIR = DATA_DIR / "products"
CHROMA_DIR = DATA_DIR / "chroma_store"
INTERACTIONS_FILE = DATA_DIR / "interactions.jsonl"
DB_PATH = DATA_DIR / "assistant.db"

load_dotenv(ROOT_DIR / ".env", override=True)

DOCUMENT_COLLECTION = "product_documents"
MEMORY_COLLECTION = "approved_interactions"
FAQ_COLLECTION = "general_support_faqs"
EMBEDDING_DIMENSIONS = 768
TOP_K = 5
FAQ_TOP_K = 3
MIN_DOCUMENT_SIMILARITY = float(os.getenv("MIN_DOCUMENT_SIMILARITY", "0.25"))
MIN_FAQ_SIMILARITY = float(os.getenv("MIN_FAQ_SIMILARITY", "0.25"))
FAQ_DIR = DATA_DIR / "faq"

# Grok / Groq Configuration
GROK_API_BASE = os.getenv("GROK_API_BASE", "https://api.groq.com/openai/v1")
GROK_API_KEY = (
    os.getenv("GROK_API_KEY")
    or os.getenv("GROQ_API_KEY")
    or os.getenv("XAI_API_KEY")
    or os.getenv("GEMINI_API_KEY", "")
)
GROK_GENERATION_MODEL = os.getenv("GROK_GENERATION_MODEL", "openai/gpt-oss-120b")
GROK_SEARCH_MODEL = os.getenv("GROK_SEARCH_MODEL", "openai/gpt-oss-120b")
GROK_VISION_MODEL = os.getenv("GROK_VISION_MODEL", "openai/gpt-oss-120b")

# Backward compatibility model variables
GENERATION_MODEL = GROK_GENERATION_MODEL
SEARCH_MODEL = GROK_SEARCH_MODEL
EMBEDDING_MODEL = "grok-local-embedding"
SUPPORT_URL = "https://www.google.com/search?q=official+manufacturer+support"


def ensure_directories() -> None:
    for path in (DATA_DIR, PRODUCTS_DIR, CHROMA_DIR, FAQ_DIR):
        path.mkdir(parents=True, exist_ok=True)


def slugify(value: str) -> str:
    clean = "".join(char.lower() if char.isalnum() else "-" for char in value)
    return "-".join(part for part in clean.split("-") if part)[:80] or "general"
