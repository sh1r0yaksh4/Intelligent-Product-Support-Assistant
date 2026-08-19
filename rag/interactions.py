from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime

from .config import INTERACTIONS_FILE, ensure_directories
from .indexer import ProductIndex
from .models import ChatResult


def record_interaction(product_id: str, question: str, result: ChatResult) -> str:
    ensure_directories()
    interaction_id = str(uuid.uuid4())
    row = {
        "id": interaction_id,
        "created_at": datetime.now(UTC).isoformat(),
        "product_id": product_id,
        "product_name": result.product_name,
        "question": question,
        "answer": result.answer,
        "citations": [citation.__dict__ for citation in result.citations],
        "escalated": result.escalated,
        "used_search": result.used_search,
        "feedback": None,
        "review_status": "pending",
    }
    with INTERACTIONS_FILE.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")
    return interaction_id


def submit_feedback(interaction_id: str, helpful: bool) -> bool:
    if not INTERACTIONS_FILE.exists():
        return False
    rows = [json.loads(line) for line in INTERACTIONS_FILE.read_text(encoding="utf-8").splitlines() if line.strip()]
    selected = next((row for row in rows if row["id"] == interaction_id), None)
    if not selected:
        return False
    selected["feedback"] = "helpful" if helpful else "not_helpful"
    # Memory remains separate from official documentation and cannot overwrite it.
    if helpful and not selected["escalated"] and selected["citations"]:
        selected["review_status"] = "approved"
        ProductIndex().add_approved_memory(
            selected["product_id"],
            selected["question"],
            selected["answer"],
            [item.get("url", "") for item in selected.get("citations", []) if item.get("url")],
        )
    elif not helpful:
        selected["review_status"] = "rejected"
    INTERACTIONS_FILE.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return True
