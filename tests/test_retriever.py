from pathlib import Path
from rag import retriever as retriever_module
from rag.models import RetrievedChunk
from rag.retriever import Retriever


def test_retriever_empty_collection_returns_empty_list(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(retriever_module, "CHROMA_DIR", tmp_path / "empty_chroma")
    retriever = Retriever()
    results = retriever.retrieve_documents("reset router", "non-existent-product-id")
    assert isinstance(results, list)
    assert len(results) == 0


def test_retrieved_chunk_score_calculation() -> None:
    chunk = RetrievedChunk(text="Sample instruction", metadata={"product_id": "demo"}, score=0.85)
    assert chunk.score == 0.85
    assert chunk.metadata["product_id"] == "demo"
