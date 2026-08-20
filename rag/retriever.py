from __future__ import annotations

import chromadb
from chromadb.config import Settings

from .config import CHROMA_DIR, DOCUMENT_COLLECTION, FAQ_COLLECTION, FAQ_TOP_K, MEMORY_COLLECTION, TOP_K
from .grok import embed_texts
from .models import RetrievedChunk


class Retriever:
    def __init__(self) -> None:
        self.client = chromadb.PersistentClient(
            path=str(CHROMA_DIR), settings=Settings(anonymized_telemetry=False)
        )

    def retrieve_documents(
        self, query: str, product_id: str, hardware_version: str = "", top_k: int = TOP_K
    ) -> list[RetrievedChunk]:
        """Query Chroma DB product_documents collection with metadata filtering."""
        try:
            collection = self.client.get_collection(DOCUMENT_COLLECTION)
        except Exception:
            return []
        if collection.count() == 0:
            return []

        embedding = embed_texts([query], "RETRIEVAL_QUERY")[0]

        # Metadata filtering
        where_filter: dict = {"product_id": product_id}
        if hardware_version and hardware_version.strip():
            where_filter = {"$and": [{"product_id": product_id}, {"hardware_version": hardware_version.strip()}]}

        try:
            result = collection.query(
                query_embeddings=[embedding], n_results=top_k, where=where_filter
            )
        except Exception:
            # Fallback to product_id filter if hardware_version exact match returns zero or errors
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

    def retrieve_approved_memory(
        self, query: str, product_id: str, top_k: int = 2
    ) -> list[RetrievedChunk]:
        """Secondary retrieval from approved_interactions memory collection."""
        try:
            collection = self.client.get_collection(MEMORY_COLLECTION)
        except Exception:
            return []
        if collection.count() == 0:
            return []

        embedding = embed_texts([query], "RETRIEVAL_QUERY")[0]
        try:
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
        except Exception:
            return []

    def retrieve_faq(self, query: str, top_k: int = FAQ_TOP_K) -> list[RetrievedChunk]:
        """Query general_support_faqs collection — no product/version filtering."""
        try:
            collection = self.client.get_collection(FAQ_COLLECTION)
        except Exception:
            return []
        if collection.count() == 0:
            return []

        embedding = embed_texts([query], "RETRIEVAL_QUERY")[0]
        try:
            result = collection.query(query_embeddings=[embedding], n_results=top_k)
            documents = result.get("documents", [[]])[0]
            metadatas = result.get("metadatas", [[]])[0]
            distances = result.get("distances", [[]])[0]
            return [
                RetrievedChunk(text=text, metadata=metadata, score=max(0.0, 1 - distance))
                for text, metadata, distance in zip(documents, metadatas, distances)
            ]
        except Exception:
            return []
