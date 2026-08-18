from pathlib import Path
from rag.loader import load_file


def test_faq_question_and_answer_stay_together(tmp_path: Path) -> None:
    file_path = tmp_path / "router_faq.txt"
    file_path.write_text("Q: How do I reset it?\nA: Hold Reset for ten seconds.\n", encoding="utf-8")

    chunks = load_file(file_path, "demo-router", hardware_version="V1")

    assert len(chunks) == 1
    assert "How do I reset it?" in chunks[0].text
    assert "Hold Reset" in chunks[0].text
    assert chunks[0].metadata["document_type"] == "faq"
    assert chunks[0].metadata["hardware_version"] == "V1"


def test_markdown_heading_is_retained(tmp_path: Path) -> None:
    file_path = tmp_path / "guide.md"
    file_path.write_text("# Setup\nConnect the WAN port to your modem.", encoding="utf-8")

    chunks = load_file(file_path, "demo-router")

    assert chunks[0].metadata["section"] == "Setup"
