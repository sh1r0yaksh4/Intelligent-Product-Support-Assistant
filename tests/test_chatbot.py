from pathlib import Path
from rag.chatbot import ProductAssistant, _context, _document_citations, _escalation
from rag.config import SUPPORT_URL
from rag.models import ChatResult, Citation, RetrievedChunk


def test_escalation_formatting() -> None:
    result = _escalation("Archer AX21", "V2")
    assert result.escalated is True
    assert "Archer AX21" in result.answer
    assert "V2" in result.answer
    assert len(result.citations) == 1
    assert result.citations[0].url == SUPPORT_URL


def test_document_citations_deduplication() -> None:
    chunks = [
        RetrievedChunk(text="Chunk 1", metadata={"source_name": "Manual.pdf", "source_url": "https://example.com/doc", "section": "Setup"}, score=0.9),
        RetrievedChunk(text="Chunk 2", metadata={"source_name": "Manual.pdf", "source_url": "https://example.com/doc", "section": "Setup 2"}, score=0.8),
        RetrievedChunk(text="Chunk 3", metadata={"source_name": "Guide.md", "source_url": "https://example.com/guide", "section": "FAQ"}, score=0.7),
    ]
    citations = _document_citations(chunks)
    assert len(citations) == 2
    assert citations[0].title == "Manual.pdf"
    assert citations[0].url == "https://example.com/doc"
    assert citations[1].title == "Guide.md"


def test_context_assembly() -> None:
    chunks = [
        RetrievedChunk(text="Press power button.", metadata={"source_name": "Manual.pdf", "section": "Power", "page": 1, "hardware_version": "V1"}, score=0.9)
    ]
    memory = [
        RetrievedChunk(text="Verified answer text.", metadata={"product_id": "test"}, score=0.95)
    ]
    ctx = _context(chunks, memory)
    assert "=== Approved Historical Memory ===" in ctx
    assert "Verified answer text." in ctx
    assert "=== Official Product Documentation Evidence ===" in ctx
    assert "Press power button." in ctx


def test_standalone_query_construction() -> None:
    history = [
        {"role": "user", "content": "How do I turn on the router?"},
        {"role": "assistant", "content": "Press the power button."},
    ]
    query = ProductAssistant._standalone_query("Where is the reset button?", history, "Archer AX21", "V2")
    assert "Archer AX21" in query
    assert "version V2" in query
    assert "How do I turn on the router?" in query
    assert "Where is the reset button?" in query


def test_record_interaction_and_feedback(tmp_path: Path, monkeypatch) -> None:
    import json
    from rag import config, interactions

    log_file = tmp_path / "test_interactions.jsonl"
    monkeypatch.setattr(config, "INTERACTIONS_FILE", log_file)
    monkeypatch.setattr(interactions, "INTERACTIONS_FILE", log_file)

    res = ChatResult(
        answer="Press power button.",
        citations=[Citation(title="Guide", url="https://example.com/doc")],
        product_name="Demo Product",
        escalated=False,
    )
    interaction_id = interactions.record_interaction("demo-product", "How to power on?", res)
    assert log_file.exists()

    rows = [json.loads(line) for line in log_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) == 1
    assert rows[0]["id"] == interaction_id
    assert rows[0]["product_id"] == "demo-product"
    assert rows[0]["question"] == "How to power on?"
    assert rows[0]["answer"] == "Press power button."
    assert rows[0]["review_status"] == "pending"

    ok = interactions.submit_feedback(interaction_id, helpful=False)
    assert ok is True

    rows = [json.loads(line) for line in log_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert rows[0]["feedback"] == "not_helpful"
    assert rows[0]["review_status"] == "rejected"


def test_clarification_result_structure() -> None:
    result = ChatResult(
        answer="Which model of headphones are you using?",
        clarification_needed=True,
        product_name="Sony Headphones",
    )
    assert result.clarification_needed is True
    assert result.escalated is False
    assert len(result.citations) == 0
    assert result.answer == "Which model of headphones are you using?"


def test_answer_triggers_clarification(monkeypatch) -> None:
    from rag import chatbot

    def mock_assess(message, history=None, active_product=None, active_version=None):
        return {
            "action": "clarify",
            "product_name": "",
            "manufacturer": "",
            "model": "",
            "hardware_version": "",
            "confidence": 0.3,
            "clarification_question": "What brand and model is your router?",
            "reasoning": "Missing product information",
        }

    monkeypatch.setattr(chatbot, "assess_conversation", mock_assess)

    assistant = ProductAssistant()
    result = assistant.answer("my wifi is not working")

    assert result.clarification_needed is True
    assert result.answer == "What brand and model is your router?"
    assert result.escalated is False
    assert len(result.citations) == 0
