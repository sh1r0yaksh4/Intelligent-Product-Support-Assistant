from rag.chatbot import _escalation
from rag.config import SUPPORT_URL


def test_escalation_formatting() -> None:
    result = _escalation("Archer AX21", "V2")
    assert result.escalated is True
    assert "Archer AX21" in result.answer
    assert "V2" in result.answer
    assert len(result.citations) == 1
    assert result.citations[0].url == SUPPORT_URL
