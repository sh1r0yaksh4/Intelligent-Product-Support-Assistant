"""End-to-End verification script for Intelligent Product Support Assistant."""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure UTF-8 output encoding if possible
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from database.db import get_db
from rag.chatbot import ProductAssistant
from rag.interactions import record_interaction, submit_feedback
from rag.retriever import Retriever


def run_e2e_tests() -> None:
    print("\n" + "=" * 80)
    print("[E2E] Running End-to-End System Flow Verification")
    print("=" * 80)

    assistant = ProductAssistant()
    db = get_db()

    # Test 1: Grounded Answer & Citation Retrieval
    print("\n[Test 1] Query: 'How do I factory reset my TP-Link Archer AX21 V1 router?'")
    result1 = assistant.answer(
        "How do I factory reset my TP-Link Archer AX21 V1 router?",
        active_product="TP-Link Archer AX21",
        active_version="V1",
    )
    print(f"  * Escalated: {result1.escalated}")
    print(f"  * Answer: {result1.answer[:160]}...")
    print(f"  * Citations: {[c.title for c in result1.citations]}")
    assert len(result1.citations) > 0, "Expected at least one citation"

    # Test 2: Database & Audit Recording
    print("\n[Test 2] Recording interaction and database synchronization")
    inter_id = record_interaction("tp-link-archer-ax21", "How do I factory reset my TP-Link Archer AX21 V1 router?", result1)
    print(f"  * Recorded interaction ID: {inter_id}")
    assert inter_id is not None

    # Test 3: Feedback Loop & Memory Promotion
    print("\n[Test 3] Submitting helpful feedback for verified memory promotion")
    feedback_ok = submit_feedback(inter_id, helpful=True)
    print(f"  * Feedback processed: {feedback_ok}")
    assert feedback_ok is True

    # Test 4: Memory Retrieval
    print("\n[Test 4] Querying secondary verified memory collection")
    retriever = Retriever()
    mem_chunks = retriever.retrieve_approved_memory("factory reset router V1", "tp-link-archer-ax21")
    print(f"  * Retrieved memory chunks: {len(mem_chunks)}")

    # Test 5: Safe Escalation on Hallucination Trap
    print("\n[Test 5] Query: 'Does the router support direct satellite orbital communication?'")
    result_unsupported = assistant.answer(
        "Does the router support direct satellite orbital communication?",
        active_product="TP-Link Archer AX21",
    )
    print(f"  * Escalated: {result_unsupported.escalated}")
    print(f"  * Answer: {result_unsupported.answer}")
    assert result_unsupported.escalated is True, "Expected safe escalation on hallucination trap"

    # Test 6: Database State Verification
    print("\n[Test 6] Verifying SQLite database state")
    products = db.list_products()
    docs = db.list_documents()
    print(f"  * Total Products in DB: {len(products)}")
    print(f"  * Total Documents in DB: {len(docs)}")
    assert len(products) >= 3

    print("\n" + "=" * 80)
    print("[SUCCESS] All End-to-End System Tests Passed Successfully!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    run_e2e_tests()
