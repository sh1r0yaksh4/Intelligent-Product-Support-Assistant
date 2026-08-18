from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime

from database.db import get_db
from .config import INTERACTIONS_FILE, ensure_directories
from .indexer import ProductIndex
from .models import ChatResult


def record_interaction(product_id: str, question: str, result: ChatResult, user_id: str = "guest") -> str:
    ensure_directories()
    interaction_id = str(uuid.uuid4())
    now_iso = datetime.now(UTC).isoformat()

    citations_list = [citation.__dict__ for citation in result.citations]
    citations_json = json.dumps(citations_list)

    # Insert into SQLite Database
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """INSERT INTO interactions
               (id, user_id, product_id, product_name, hardware_version, question, answer, citations_json, escalated, used_search, review_status)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);""",
            (
                interaction_id,
                user_id,
                product_id,
                result.product_name or "",
                result.hardware_version or "",
                question,
                result.answer,
                citations_json,
                1 if result.escalated else 0,
                1 if result.used_search else 0,
                "pending",
            ),
        )

    # Append to JSONL log
    row = {
        "id": interaction_id,
        "user_id": user_id,
        "created_at": now_iso,
        "product_id": product_id,
        "product_name": result.product_name,
        "hardware_version": result.hardware_version,
        "question": question,
        "answer": result.answer,
        "citations": citations_list,
        "escalated": result.escalated,
        "used_search": result.used_search,
        "feedback": None,
        "review_status": "pending",
    }
    with INTERACTIONS_FILE.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")

    return interaction_id


def submit_feedback(interaction_id: str, helpful: bool, comment: str = "") -> bool:
    """Submit user feedback for an interaction. If helpful and backed by citations, promote to approved memory store."""
    feedback_id = str(uuid.uuid4())

    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """INSERT INTO feedback (id, interaction_id, helpful, comment)
               VALUES (?, ?, ?, ?);""",
            (feedback_id, interaction_id, 1 if helpful else 0, comment),
        )

        cursor.execute(
            """SELECT product_id, product_name, hardware_version, question, answer, citations_json, escalated
               FROM interactions WHERE id = ?;""",
            (interaction_id,),
        )
        row = cursor.fetchone()
        if row:
            status = "approved" if (helpful and not row["escalated"]) else "rejected"
            cursor.execute("UPDATE interactions SET review_status = ? WHERE id = ?;", (status, interaction_id))

            if helpful and not row["escalated"]:
                citations = json.loads(row["citations_json"] or "[]")
                urls = [c["url"] for c in citations if c.get("url")]
                ProductIndex().add_approved_memory(
                    product_id=row["product_id"],
                    question=row["question"],
                    answer=row["answer"],
                    source_urls=urls,
                    hardware_version=row["hardware_version"] or "",
                    model=row["product_name"] or "",
                )

    # Update JSONL log file
    if INTERACTIONS_FILE.exists():
        rows = [json.loads(line) for line in INTERACTIONS_FILE.read_text(encoding="utf-8").splitlines() if line.strip()]
        for item in rows:
            if item["id"] == interaction_id:
                item["feedback"] = "helpful" if helpful else "not_helpful"
                item["review_status"] = "approved" if (helpful and not item["escalated"]) else "rejected"
        INTERACTIONS_FILE.write_text("".join(json.dumps(line) + "\n" for line in rows), encoding="utf-8")

    return True
