from __future__ import annotations

import uuid
from typing import Iterable

import chromadb
from chromadb.config import Settings

from .config import CHROMA_DIR, DOCUMENT_COLLECTION, MEMORY_COLLECTION, ensure_directories
from .gemini import embed_texts
from .models import DocumentChunk


class ProductIndex:
    def __init__(self) -> None:
        ensure_directories()
        self.client = chromadb.PersistentClient(
            path=str(CHROMA_DIR), settings=Settings(anonymized_telemetry=False)
        )

    def _collection(self, name: str):
        return self.client.get_or_create_collection(name, metadata={"hnsw:space": "cosine"})

    def index_documents(self, chunks: Iterable[DocumentChunk], product_id: str) -> int:
        values = list(chunks)
        if not values:
            return 0
        collection = self._collection(DOCUMENT_COLLECTION)
        existing = collection.get(where={"product_id": product_id})
        if existing.get("ids"):
            collection.delete(ids=existing["ids"])
        for start in range(0, len(values), 40):
            batch = values[start : start + 40]
            embeddings = embed_texts([item.text for item in batch], "RETRIEVAL_DOCUMENT")
            collection.add(
                ids=[str(uuid.uuid4()) for _ in batch],
                documents=[item.text for item in batch],
                embeddings=embeddings,
                metadatas=[item.metadata for item in batch],
            )
        return len(values)

    def add_approved_memory(self, product_id: str, question: str, answer: str, source_urls: list[str]) -> None:
        text = f"Customer question: {question}\nVerified helpful answer: {answer}"
        collection = self._collection(MEMORY_COLLECTION)
        collection.add(
            ids=[str(uuid.uuid4())],
            documents=[text],
            embeddings=embed_texts([text], "RETRIEVAL_DOCUMENT"),
            metadatas=[{"product_id": product_id, "source_urls": " | ".join(source_urls), "approved": True}],
        )

