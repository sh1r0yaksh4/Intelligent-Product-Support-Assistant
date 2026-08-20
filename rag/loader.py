from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

from pypdf import PdfReader

from .models import DocumentChunk
from .ocr import extract_ocr_text_from_pdf

MAX_WORDS = 400
OVERLAP_WORDS = 50


def clean_text(text: str) -> str:
    # Replace non-ascii bullet characters with standard ASCII dashes
    text = (
        text.replace("\u25cf", "- ")
        .replace("\u2022", "- ")
        .replace("\u25cb", "- ")
        .replace("\u25a0", "- ")
        .replace("\u2013", "-")
        .replace("\u2014", "-")
        .replace("\u2018", "'")
        .replace("\u2019", "'")
        .replace("\u201c", '"')
        .replace("\u201d", '"')
    )
    return re.sub(r"\s+", " ", text).strip()


def _chunk_words(text: str) -> list[str]:
    words = clean_text(text).split()
    if not words:
        return []
    if len(words) <= MAX_WORDS:
        return [" ".join(words)]
    chunks: list[str] = []
    step = MAX_WORDS - OVERLAP_WORDS
    for start in range(0, len(words), step):
        chunk = words[start : start + MAX_WORDS]
        if chunk:
            chunks.append(" ".join(chunk))
        if start + MAX_WORDS >= len(words):
            break
    return chunks


def _faq_blocks(text: str) -> list[tuple[str, str]]:
    """Keep Q/A pairs together for documents written as FAQ text."""
    pattern = re.compile(
        r"(?:^|\n)\s*(?:Q(?:uestion)?\s*[:.-])\s*(.+?)\s*\n\s*"
        r"(?:A(?:nswer)?\s*[:.-])\s*(.+?)(?=(?:\n\s*Q(?:uestion)?\s*[:.-])|\Z)",
        re.I | re.S,
    )
    return [(clean_text(question), clean_text(answer)) for question, answer in pattern.findall(text)]


def _markdown_sections(text: str) -> Iterable[tuple[str, str]]:
    heading = "General information"
    lines: list[str] = []
    for line in text.splitlines():
        match = re.match(r"^\s*(?:#{1,6}\s+|\d+(?:\.\d+)*\s+)(.+?)\s*$", line)
        if match:
            if clean_text("\n".join(lines)):
                yield heading, "\n".join(lines)
            heading = match.group(1).strip()
            lines = []
        else:
            lines.append(line)
    if clean_text("\n".join(lines)):
        yield heading, "\n".join(lines)


def _read_pdf(path: Path) -> list[tuple[str, str, int]]:
    reader = PdfReader(str(path))
    extracted: list[tuple[str, str, int]] = []
    total_text_len = 0
    for page_number, page in enumerate(reader.pages, start=1):
        text = clean_text(page.extract_text() or "")
        if text:
            extracted.append((f"Page {page_number}", text, page_number))
            total_text_len += len(text)

    # Scanned PDF fallback if extracted text is effectively empty
    if total_text_len < 50:
        ocr_results = extract_ocr_text_from_pdf(path)
        if ocr_results:
            extracted = [(f"Page {p_num} (OCR)", text, p_num) for p_num, text in ocr_results]

    return extracted


def _detect_version_in_text(text: str) -> str:
    """Extract hardware version (V1, V2, Rev A) from heading or text if present."""
    match = re.search(r"\b(V\d+(?:\.\d+)?|Rev\s*[A-Z0-9]+)\b", text, re.I)
    if match:
        v = match.group(1).upper()
        return re.sub(r"\.0+$", "", v)
    return ""


def load_file(
    path: Path,
    product_id: str,
    source_url: str = "",
    manufacturer: str = "",
    model: str = "",
    hardware_version: str = "",
    revision: str = "",
    source_id: str = "",
    approval_status: str = "APPROVED",
) -> list[DocumentChunk]:
    """Return chunks with complete source and product version metadata from a PDF, TXT, or Markdown file."""
    extension = path.suffix.lower()
    base_metadata = {
        "product_id": product_id,
        "manufacturer": manufacturer,
        "model": model,
        "hardware_version": hardware_version,
        "revision": revision,
        "source_id": source_id or path.stem,
        "source_name": path.name,
        "source_title": path.stem.replace("_", " "),
        "source_url": source_url,
        "source_type": extension.lstrip("."),
        "document_type": "faq" if "faq" in path.stem.lower() else "document",
        "approval_status": approval_status,
    }
    chunks: list[DocumentChunk] = []

    if extension == ".pdf":
        sections = _read_pdf(path)
    elif extension in {".txt", ".md"}:
        text = path.read_text(encoding="utf-8", errors="ignore")
        faq_pairs = _faq_blocks(text)
        if faq_pairs:
            sections = [(question, f"Question: {question}\nAnswer: {answer}", 0) for question, answer in faq_pairs]
            base_metadata["document_type"] = "faq"
        else:
            sections = [(title, body, 0) for title, body in _markdown_sections(text)]
    else:
        return []

    for section, body, page in sections:
        section_hw = hardware_version or _detect_version_in_text(section) or _detect_version_in_text(body[:100])
        # Detect table of contents, index pages, or page number list pages
        is_toc = (
            (bool(re.search(r"\.{4,}|\bTable of contents\b", body, re.I)) and len(re.findall(r"\.{3,}", body)) >= 3)
            or bool(re.search(r"^\s*Index\b", body, re.I))
            or (page and page >= 68 and "c05633409" in path.name.lower())
        )

        for index, part in enumerate(_chunk_words(body)):
            metadata = dict(base_metadata)
            metadata.update({
                "section": section,
                "hardware_version": section_hw,
                "page": page or "",
                "page_number": page or 0,
                "chunk_index": index,
                "is_toc": is_toc,
            })
            chunks.append(DocumentChunk(text=part, metadata=metadata))
    return chunks
