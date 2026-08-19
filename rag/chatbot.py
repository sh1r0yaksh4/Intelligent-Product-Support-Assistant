from __future__ import annotations

from .config import MIN_DOCUMENT_SIMILARITY, SUPPORT_URL, slugify
from .gemini import (
    assess_conversation,
    generate_grounded_answer,
    identify_product_structured,
    search_grounded_answer,
)
from .models import ChatResult, Citation, ProductEntity, RetrievedChunk
from .retriever import Retriever

SYSTEM_INSTRUCTIONS = """You are a careful, version-aware product-support assistant.
Use ONLY the supplied verified evidence. Never invent a specification, procedure, reset sequence, or product feature.
If the evidence does not answer the question or if incompatible hardware versions are mixed, respond exactly with NOT_FOUND: followed by a brief explanation.
Give concise, step-by-step instructions. Cite the exact source title, section, or page number when answering from product documents."""


def _context(chunks: list[RetrievedChunk], memory_chunks: list[RetrievedChunk] | None = None) -> str:
    parts = []
    if memory_chunks:
        parts.append("=== Approved Historical Memory ===")
        for chunk in memory_chunks:
            parts.append(f"[Verified Prior Q&A]\n{chunk.text}")

    parts.append("=== Official Product Documentation Evidence ===")
    for chunk in chunks:
        parts.append(
            f"[Source: {chunk.metadata.get('source_name', 'Doc')} | section: {chunk.metadata.get('section', 'General')} | page: {chunk.metadata.get('page', '')} | ver: {chunk.metadata.get('hardware_version', '')}]\n{chunk.text}"
        )
    return "\n\n".join(parts)


def _document_citations(chunks: list[RetrievedChunk]) -> list[Citation]:
    seen: set[tuple[str, str]] = set()
    citations: list[Citation] = []
    for chunk in chunks:
        title = chunk.metadata.get("source_name") or chunk.metadata.get("source_title") or "Product Manual"
        url = chunk.metadata.get("source_url", "")
        key = (title, url)
        if key in seen:
            continue
        seen.add(key)
        citations.append(Citation(title=title, url=url, section=chunk.metadata.get("section"), source_type="document"))
    return citations[:3]


def _escalation(product_name: str | None, hardware_version: str | None = None) -> ChatResult:
    p_label = f" for {product_name}" if product_name else ""
    v_label = f" (hardware version {hardware_version})" if hardware_version else ""
    return ChatResult(
        answer=(
            f"I couldn’t verify an answer{p_label}{v_label} from approved documentation. "
            "To prevent incorrect hardware instructions, please verify your model/version or check official manufacturer support."
        ),
        citations=[Citation(title="Find official manufacturer support", url=SUPPORT_URL, source_type="support")],
        escalated=True,
        product_name=product_name,
        hardware_version=hardware_version,
    )


class ProductAssistant:
    def __init__(self) -> None:
        self.retriever = Retriever()

    def answer(
        self,
        question: str,
        history: list[dict[str, str]] | None = None,
        active_product: str | None = None,
        active_version: str | None = None,
        user_style: str | None = None,
    ) -> ChatResult:
        history_list = history or []

        # Step 1: Assess conversation — clarify or answer?
        assessment = assess_conversation(question, history_list, active_product, active_version)
        if assessment.get("action") == "clarify" and assessment.get("clarification_question"):
            return ChatResult(
                answer=assessment["clarification_question"],
                clarification_needed=True,
                product_name=assessment.get("product_name") or active_product,
                hardware_version=assessment.get("hardware_version") or active_version,
            )

        product_name = assessment.get("product_name") or active_product
        hardware_version = assessment.get("hardware_version") or active_version or ""
        product_id = slugify(product_name or "general")

        # Step 2: Formulate query
        retrieval_query = self._standalone_query(question, history_list, product_name, hardware_version)

        # Step 3: Local Document Vector Retrieval (Version-Aware)
        local_chunks = self.retriever.retrieve_documents(retrieval_query, product_id, hardware_version)
        memory_chunks = self.retriever.retrieve_approved_memory(retrieval_query, product_id, top_k=2)

        style_instruction = f"\nUser style preference: {user_style}\n" if user_style else ""

        # Step 4: Relevance & Score Check
        if local_chunks and local_chunks[0].score >= MIN_DOCUMENT_SIMILARITY:
            prompt = f"""{SYSTEM_INSTRUCTIONS}
{style_instruction}
Product: {product_name or 'unknown'}
Hardware Version: {hardware_version or 'unspecified'}
Conversation context: {history_list[-4:]}

Evidence:
{_context(local_chunks, memory_chunks)}

Question: {question}
Answer:"""
            response = generate_grounded_answer(prompt)
            if response.startswith("NOT_FOUND:"):
                return _escalation(product_name, hardware_version)
            return ChatResult(
                answer=response,
                citations=_document_citations(local_chunks),
                product_name=product_name,
                hardware_version=hardware_version,
                used_memory=bool(memory_chunks),
            )

        # Step 5: Web Search Grounding Fallback
        research_prompt = f"""Research this product-support question using Google Search grounding.
{style_instruction}
Product: {product_name or 'not specified'}
Hardware Version: {hardware_version or 'not specified'}
Question: {question}

Use official manufacturer product pages, user manuals, and support portals as the evidence standard.
Do not answer from generic forums, reseller listings, or unverified blogs.
If you cannot find sufficient official evidence for a safe answer, begin your response exactly with NOT_FOUND:. Keep instructions concise."""

        response, citations = search_grounded_answer(research_prompt)
        if not response or response.startswith("NOT_FOUND:") or not citations:
            return _escalation(product_name, hardware_version)

        return ChatResult(
            answer=response,
            citations=citations[:3],
            product_name=product_name,
            hardware_version=hardware_version,
            used_search=True,
        )

    @staticmethod
    def _standalone_query(
        question: str, history: list[dict[str, str]], product_name: str | None, hardware_version: str | None
    ) -> str:
        recent = " ".join(turn["content"] for turn in history[-4:] if turn["role"] == "user")
        parts = [product_name or "", f"version {hardware_version}" if hardware_version else "", recent, question]
        return " ".join(part for part in parts if part)
