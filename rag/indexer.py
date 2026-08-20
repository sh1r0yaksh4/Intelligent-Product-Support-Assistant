from __future__ import annotations

import uuid
from typing import Iterable

import chromadb
from chromadb.config import Settings

from .config import CHROMA_DIR, DOCUMENT_COLLECTION, FAQ_COLLECTION, MEMORY_COLLECTION, ensure_directories
from .grok import embed_texts
from .models import DocumentChunk


class ProductIndex:
    def __init__(self) -> None:
        ensure_directories()
        self.client = chromadb.PersistentClient(
            path=str(CHROMA_DIR), settings=Settings(anonymized_telemetry=False)
        )

    def _collection(self, name: str):
        return self.client.get_or_create_collection(name, metadata={"hnsw:space": "cosine"})

    def index_documents(self, chunks: Iterable[DocumentChunk], product_id: str, source_id: str | None = None) -> int:
        values = list(chunks)
        if not values:
            return 0
        collection = self._collection(DOCUMENT_COLLECTION)
        
        # Clear previous chunks for this product/source if needed
        where_clause = {"product_id": product_id}
        if source_id:
            where_clause["source_id"] = source_id
            
        try:
            existing = collection.get(where=where_clause)
            if existing and existing.get("ids"):
                collection.delete(ids=existing["ids"])
        except Exception:
            pass

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

    def delete_source_chunks(self, product_id: str, source_id: str) -> None:
        collection = self._collection(DOCUMENT_COLLECTION)
        try:
            existing = collection.get(where={"$and": [{"product_id": product_id}, {"source_id": source_id}]})
            if existing and existing.get("ids"):
                collection.delete(ids=existing["ids"])
        except Exception:
            pass

    def add_approved_memory(
        self,
        product_id: str,
        question: str,
        answer: str,
        source_urls: list[str],
        hardware_version: str = "",
        model: str = "",
    ) -> None:
        text = f"Customer question: {question}\nVerified helpful answer: {answer}"
        collection = self._collection(MEMORY_COLLECTION)
        collection.add(
            ids=[str(uuid.uuid4())],
            documents=[text],
            embeddings=embed_texts([text], "RETRIEVAL_DOCUMENT"),
            metadatas=[
                {
                    "product_id": product_id,
                    "model": model,
                    "hardware_version": hardware_version,
                    "source_urls": " | ".join(source_urls),
                    "approved": True,
                }
            ],
        )

    def index_faq_pairs(self, faq_path: "Path") -> int:
        """Index general support FAQ Q&A pairs into the FAQ collection."""
        import json
        from pathlib import Path as _Path

        faq_path = _Path(faq_path)
        if not faq_path.exists():
            return 0

        pairs = [json.loads(line) for line in faq_path.read_text("utf-8").splitlines() if line.strip()]
        if not pairs:
            return 0

        collection = self._collection(FAQ_COLLECTION)

        # Clear existing FAQ data for clean re-index
        if collection.count() > 0:
            existing = collection.get()
            if existing and existing.get("ids"):
                collection.delete(ids=existing["ids"])

        import time
        for start in range(0, len(pairs), 50):
            batch = pairs[start : start + 50]
            texts = [f"Q: {p['question']}\nA: {p['answer']}" for p in batch]
            embeddings = embed_texts(texts, "RETRIEVAL_DOCUMENT")
            collection.add(
                ids=[str(uuid.uuid4()) for _ in batch],
                documents=texts,
                embeddings=embeddings,
                metadatas=[{
                    "category": p.get("category", "general"),
                    "intent": p.get("intent", ""),
                    "source": p.get("source", "faq"),
                    "question": p["question"],
                    "domain": "general_support",
                } for p in batch],
            )
            time.sleep(2.0)
        return len(pairs)
