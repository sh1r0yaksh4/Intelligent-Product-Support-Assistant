from pathlib import Path
from database.db import Database


def test_database_initialization_and_product_crud(tmp_path: Path) -> None:
    db_file = tmp_path / "test_assistant.db"
    db = Database(db_path=db_file)
    assert db_file.exists()

    db.upsert_product("test-router", "Test Router AX", manufacturer="TP-Link", model="AX100")
    product = db.get_product("test-router")
    assert product is not None
    assert product["name"] == "Test Router AX"
    assert product["manufacturer"] == "TP-Link"
    assert product["model"] == "AX100"

    products = db.list_products()
    assert len(products) == 1


def test_database_conversation_and_message_crud(tmp_path: Path) -> None:
    db_file = tmp_path / "test_assistant.db"
    db = Database(db_path=db_file)

    db.upsert_conversation("conv-1", "Reset Instructions", active_product="Test Router")
    conv = db.get_conversation("conv-1")
    assert conv is not None
    assert conv["title"] == "Reset Instructions"
    assert conv["active_product"] == "Test Router"
    assert conv["archived"] is False

    msg_id = db.add_message(
        conversation_id="conv-1",
        role="user",
        content="How do I reset?",
    )
    assert msg_id > 0

    assistant_msg_id = db.add_message(
        conversation_id="conv-1",
        role="assistant",
        content="Hold reset for 10 seconds.",
        citations=[{"title": "Manual", "url": "https://example.com"}],
        escalated=False,
    )
    assert assistant_msg_id > 0

    messages = db.get_messages("conv-1")
    assert len(messages) == 2
    assert messages[0]["role"] == "user"
    assert messages[1]["role"] == "assistant"
    assert len(messages[1]["citations"]) == 1
    assert messages[1]["citations"][0]["title"] == "Manual"


def test_database_document_and_interaction_tracking(tmp_path: Path) -> None:
    db_file = tmp_path / "test_assistant.db"
    db = Database(db_path=db_file)

    db.upsert_product("sony-xm5", "Sony WH-1000XM5")
    db.record_document(
        document_id="manual",
        product_id="sony-xm5",
        filename="manual.md",
        source_type="md",
        total_chunks=8,
    )

    docs = db.list_documents("sony-xm5")
    assert len(docs) == 1
    assert docs[0]["filename"] == "manual.md"
    assert docs[0]["total_chunks"] == 8

    db.record_interaction(
        interaction_id="inter-101",
        product_id="sony-xm5",
        product_name="Sony WH-1000XM5",
        question="How to pair?",
        answer="Hold power 7 sec.",
        citations=[{"title": "Sony Guide", "url": "https://sony.com"}],
        escalated=False,
        used_search=False,
    )

    updated = db.update_interaction_feedback("inter-101", "helpful", "approved")
    assert updated is True
