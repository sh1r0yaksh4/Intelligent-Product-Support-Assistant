from __future__ import annotations

from .config import MIN_DOCUMENT_SIMILARITY, SUPPORT_URL, slugify
from .gemini import generate_grounded_answer, identify_product, search_grounded_answer
from .models import ChatResult, Citation, RetrievedChunk
from .retriever import Retriever

SYSTEM_INSTRUCTIONS = """You are a careful product-support assistant.
Use ONLY the supplied evidence. Never invent a specification, procedure, or product feature.
If the evidence does not answer the question, respond exactly with NOT_FOUND: followed by a brief explanation.
Give concise, natural-language steps. Cite the source title and section when answering from product documents."""


def _context(chunks: list[RetrievedChunk]) -> str:
    return "\n\n".join(
        f"[Source: {chunk.metadata.get('source_title')} | section: {chunk.metadata.get('section')} | page: {chunk.metadata.get('page')}]\n{chunk.text}"
        for chunk in chunks
    )


def _document_citations(chunks: list[RetrievedChunk]) -> list[Citation]:
    seen: set[tuple[str, str]] = set()
    citations: list[Citation] = []
    for chunk in chunks:
        key = (chunk.metadata.get("source_title", "Source"), chunk.metadata.get("source_url", ""))
        if key in seen:
            continue
        seen.add(key)
        citations.append(Citation(title=key[0], url=key[1], section=chunk.metadata.get("section")))
    return citations[:3]


def _escalation(product_name: str | None) -> ChatResult:
    label = f" for {product_name}" if product_name else ""
    return ChatResult(
        answer=(f"I couldn’t verify an answer{label} from reliable documentation. "
                "Please check the manufacturer’s official support site or provide the product manual."),
        citations=[Citation(title="Find official manufacturer support", url=SUPPORT_URL, source_type="support")],
        escalated=True,
        product_name=product_name,
    )


class ProductAssistant:
    def __init__(self) -> None:
        self.retriever = Retriever()

    def answer(self, question: str, history: list[dict[str, str]], active_product: str | None) -> ChatResult:
        product_name = identify_product(question, active_product)
        product_id = slugify(product_name or "general")
        retrieval_query = self._standalone_query(question, history, product_name)
        local_chunks = self.retriever.retrieve_documents(retrieval_query, product_id)

        if local_chunks and local_chunks[0].score >= MIN_DOCUMENT_SIMILARITY:
            prompt = f"""{SYSTEM_INSTRUCTIONS}

Product: {product_name or 'unknown'}
Conversation context: {history[-4:]}

Evidence:
{_context(local_chunks)}

Question: {question}
Answer:"""
            response = generate_grounded_answer(prompt)
            if response.startswith("NOT_FOUND:"):
                return _escalation(product_name)
            return ChatResult(answer=response, citations=_document_citations(local_chunks), product_name=product_name)

        # Research in the same chat turn rather than forcing a product-setup form.
        research_prompt = f"""Research this product-support question using Google Search.
Product (if known): {product_name or 'not identified'}.
Question: {question}

Use manufacturer product pages, manuals, and official support pages as the evidence standard.
Do not answer from retailer listings, forums, or guesses. If you cannot find enough official
evidence for a safe answer, begin your response exactly with NOT_FOUND:. Keep the answer concise."""
        response, citations = search_grounded_answer(research_prompt)
        if not response or response.startswith("NOT_FOUND:") or not citations:
            return _escalation(product_name)
        return ChatResult(answer=response, citations=citations[:3], product_name=product_name, used_search=True)

    @staticmethod
    def _standalone_query(question: str, history: list[dict[str, str]], product_name: str | None) -> str:
        recent = " ".join(turn["content"] for turn in history[-4:] if turn["role"] == "user")
        return " ".join(part for part in (product_name or "", recent, question) if part)

