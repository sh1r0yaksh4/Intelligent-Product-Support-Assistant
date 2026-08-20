from __future__ import annotations

import json
import re
import uuid
from datetime import UTC, datetime

from database.db import get_db

from .config import INTERACTIONS_FILE, ensure_directories
from .indexer import ProductIndex
from .models import ChatResult

SUSPICIOUS_PATTERNS = [
    r"ignore (?:all )?previous instructions",
    r"system prompt",
    r"<script[\s>]",
    r"javascript:",
    r"eval\(",
    r"drop table",
]


def _is_safe_for_memory(question: str, answer: str, citations: list[dict]) -> bool:
    """Validate that Q&A pair is safe and grounded before promoting to ChromaDB memory."""
    if not answer or len(answer.strip()) < 20:
        return False
    if not citations:
        return False

    combined = f"{question} {answer}".lower()
    for pattern in SUSPICIOUS_PATTERNS:
        if re.search(pattern, combined, re.I):
            return False
    return True


def record_interaction(product_id: str, question: str, result: ChatResult) -> str:
    ensure_directories()
    interaction_id = str(uuid.uuid4())
    citations_data = [citation.__dict__ for citation in result.citations]
    row = {
        "id": interaction_id,
        "created_at": datetime.now(UTC).isoformat(),
        "product_id": product_id,
        "product_name": result.product_name,
        "question": question,
        "answer": result.answer,
        "citations": citations_data,
        "escalated": result.escalated,
        "used_search": result.used_search,
        "feedback": None,
        "review_status": "pending",
    }

    # Save to JSONL file audit log
    with INTERACTIONS_FILE.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")

    # Sync to SQLite Database
    try:
        get_db().record_interaction(
            interaction_id=interaction_id,
            product_id=product_id,
            product_name=result.product_name,
            question=question,
            answer=result.answer,
            citations=citations_data,
            escalated=result.escalated,
            used_search=result.used_search,
        )
    except Exception:
        pass

    return interaction_id


def submit_feedback(interaction_id: str, helpful: bool) -> bool:
    if not INTERACTIONS_FILE.exists():
        return False
    rows = [json.loads(line) for line in INTERACTIONS_FILE.read_text(encoding="utf-8").splitlines() if line.strip()]
    selected = next((row for row in rows if row["id"] == interaction_id), None)
    if not selected:
        return False

    feedback_str = "helpful" if helpful else "not_helpful"
    selected["feedback"] = feedback_str

    # Multi-stage validation before promoting to persistent retrieval memory
    if helpful and not selected.get("escalated") and _is_safe_for_memory(selected["question"], selected["answer"], selected.get("citations", [])):
        selected["review_status"] = "approved"
        try:
            ProductIndex().add_approved_memory(
                selected["product_id"],
                selected["question"],
                selected["answer"],
                [item.get("url", "") for item in selected.get("citations", []) if item.get("url")],
            )
        except Exception:
            pass
    elif not helpful:
        selected["review_status"] = "rejected"
    else:
        selected["review_status"] = "unverified"

    INTERACTIONS_FILE.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    # Sync feedback with SQLite database
    try:
        get_db().update_interaction_feedback(interaction_id, feedback_str, selected["review_status"])
    except Exception:
        pass

    return True
