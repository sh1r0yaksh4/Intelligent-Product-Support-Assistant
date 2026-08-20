from __future__ import annotations

import re

from .config import MIN_DOCUMENT_SIMILARITY, MIN_FAQ_SIMILARITY, SUPPORT_URL, slugify
from .grok import (
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

FAQ_SYSTEM_INSTRUCTIONS = """You are a helpful customer support assistant.
Answer using ONLY the provided FAQ evidence. Give clear, concise answers.
If the FAQ evidence does not cover the question, respond exactly with NOT_FOUND: followed by a brief explanation.
Do not invent policies, deadlines, or procedures not present in the evidence."""


def _context(
    chunks: list[RetrievedChunk],
    memory_chunks: list[RetrievedChunk] | None = None,
    visual_info: str = "",
) -> str:
    parts = []
    if visual_info:
        parts.append(f"=== Visual Hardware Inspection Finding ===\n{visual_info}")

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


def _faq_context(chunks: list[RetrievedChunk]) -> str:
    """Format FAQ chunks as evidence without hardware metadata."""
    parts = ["=== General Support FAQ Evidence ==="]
    for chunk in chunks:
        category = chunk.metadata.get("category", "general").replace("_", " ").title()
        parts.append(f"[Category: {category}]\n{chunk.text}")
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


def _faq_citations(chunks: list[RetrievedChunk]) -> list[Citation]:
    """Build category-based citations for FAQ answers."""
    seen: set[str] = set()
    citations: list[Citation] = []
    for chunk in chunks:
        category = chunk.metadata.get("category", "general").replace("_", " ").title()
        if category in seen:
            continue
        seen.add(category)
        citations.append(Citation(title=category, url="", source_type="faq"))
    return citations[:3]


def _escalation(product_name: str | None, hardware_version: str | None = None) -> ChatResult:
    p_label = f" for {product_name}" if product_name else ""
    v_label = f" (hardware version {hardware_version})" if hardware_version else ""
    return ChatResult(
        answer=(
            f"I couldn't verify an answer{p_label}{v_label} from approved documentation. "
            "To prevent incorrect hardware instructions, please verify your model/version or check official manufacturer support."
        ),
        citations=[Citation(title="Find official manufacturer support", url=SUPPORT_URL, source_type="support")],
        escalated=True,
        product_name=product_name,
        hardware_version=hardware_version,
    )


def _general_escalation() -> ChatResult:
    """Escalation for general support questions not covered by FAQ data."""
    return ChatResult(
        answer=(
            "I couldn't find a verified answer in our support FAQ. "
            "Please contact our customer support team directly for assistance with your inquiry."
        ),
        citations=[Citation(title="Contact Customer Support", url=SUPPORT_URL, source_type="support")],
        escalated=True,
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

        # Step 1: Assess conversation — clarify, classify domain, or answer?
        assessment = assess_conversation(question, history_list, active_product, active_version)
        if not active_product and assessment.get("action") == "clarify" and assessment.get("clarification_question"):
            return ChatResult(
                answer=assessment["clarification_question"],
                clarification_needed=True,
                product_name=assessment.get("product_name") or active_product,
                hardware_version=assessment.get("hardware_version") or active_version,
            )

        question_domain = assessment.get("question_domain", "product_support")
        style_instruction = f"\nUser style preference: {user_style}\n" if user_style else ""

        # ── General Support Path ────────────────────────────────────────
        if question_domain == "general_support":
            return self._answer_general_support(question, history_list, style_instruction)

        # ── Product Support Path (unchanged) ────────────────────────────
        return self._answer_product_support(
            question, history_list, assessment, active_product, active_version, style_instruction
        )

    def _answer_general_support(
        self, question: str, history_list: list[dict[str, str]], style_instruction: str
    ) -> ChatResult:
        """Handle account, order, shipping, returns, refund questions via FAQ collection."""
        # Step 1: FAQ Vector Retrieval
        faq_chunks = self.retriever.retrieve_faq(question)

        if faq_chunks and faq_chunks[0].score >= MIN_FAQ_SIMILARITY:
            prompt = f"""{FAQ_SYSTEM_INSTRUCTIONS}
{style_instruction}
Conversation context: {history_list[-4:]}

Evidence:
{_faq_context(faq_chunks)}

Question: {question}
Answer:"""
            response = generate_grounded_answer(prompt)
            if not response.startswith("NOT_FOUND:"):
                return ChatResult(
                    answer=response,
                    citations=_faq_citations(faq_chunks),
                )

        # Step 2: Web Search Grounding Fallback (general support)
        research_prompt = f"""Research this general customer support question using Google Search grounding.
{style_instruction}
Question: {question}

Look for official company FAQ pages, help centers, and customer service documentation.
If you cannot find a clear, verified answer, begin your response exactly with NOT_FOUND:. Keep the answer concise."""

        response, citations = search_grounded_answer(research_prompt)
        if not response or response.startswith("NOT_FOUND:") or not citations:
            return _general_escalation()

        return ChatResult(
            answer=response,
            citations=citations[:3],
            used_search=True,
        )

    def _answer_product_support(
        self,
        question: str,
        history_list: list[dict[str, str]],
        assessment: dict,
        active_product: str | None,
        active_version: str | None,
        style_instruction: str,
    ) -> ChatResult:
        """Handle hardware troubleshooting questions via product_docs + approved_memory."""
        product_name = assessment.get("product_name") or active_product
        hardware_version = assessment.get("hardware_version") or active_version or ""
        product_id = slugify(product_name or "general")

        # Step 2: Formulate query
        is_summary_request = bool(re.search(
            r"\b(summary|summarize|overview|spec|specs|specification|specifications|feature|features|detail|details|info|information|explain|describe|what is this|about|about it|help)\b",
            question,
            re.I,
        ))
        if is_summary_request:
            retrieval_query = f"{product_name or ''} overview introduction summary specifications features guide"
        else:
            retrieval_query = self._standalone_query(question, history_list, product_name, hardware_version)

        # Step 3: Local Document Vector Retrieval (Version-Aware)
        local_chunks = self.retriever.retrieve_documents(retrieval_query, product_id, hardware_version)
        memory_chunks = self.retriever.retrieve_approved_memory(retrieval_query, product_id, top_k=2)

        # Step 4: Relevance & Score Check
        if local_chunks and (local_chunks[0].score >= MIN_DOCUMENT_SIMILARITY or is_summary_request or active_product):
            prompt = f"""You are an expert product support assistant.
Answer the user's question clearly, thoroughly, and helpfully using the provided documentation evidence.
Synthesize relevant details, specifications, regulatory information, features, steps, gestures, settings, and procedures found in the evidence.
Use clear bullet points and bold headings for easy readability.
{style_instruction}
Product: {product_name or 'unknown'}
Hardware Version: {hardware_version or 'unspecified'}

Evidence:
{_context(local_chunks, memory_chunks)}

Question: {question}
Answer:"""
            response = generate_grounded_answer(prompt)
            clean_answer = response.removeprefix("NOT_FOUND:").strip()
            if clean_answer and not response.startswith("NOT_FOUND:"):
                return ChatResult(
                    answer=clean_answer,
                    citations=_document_citations(local_chunks),
                    product_name=product_name,
                    hardware_version=hardware_version,
                    used_memory=bool(memory_chunks),
                )
            if active_product:
                return _escalation(product_name, hardware_version)

        if active_product:
            return _escalation(product_name, hardware_version)

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
