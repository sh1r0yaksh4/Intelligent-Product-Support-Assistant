from __future__ import annotations

import chromadb
from chromadb.config import Settings

from .config import CHROMA_DIR, DOCUMENT_COLLECTION, TOP_K
from .gemini import embed_texts
from .models import RetrievedChunk


class Retriever:
    def __init__(self) -> None:
        self.client = chromadb.PersistentClient(
            path=str(CHROMA_DIR), settings=Settings(anonymized_telemetry=False)
        )

    def retrieve_documents(self, query: str, product_id: str, top_k: int = TOP_K) -> list[RetrievedChunk]:
        try:
            collection = self.client.get_collection(DOCUMENT_COLLECTION)
        except Exception:
            return []
        if collection.count() == 0:
            return []
        embedding = embed_texts([query], "RETRIEVAL_QUERY")[0]
        result = collection.query(
            query_embeddings=[embedding], n_results=top_k, where={"product_id": product_id}
        )
        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]
        return [
            RetrievedChunk(text=text, metadata=metadata, score=max(0.0, 1 - distance))
            for text, metadata, distance in zip(documents, metadatas, distances)
        ]

