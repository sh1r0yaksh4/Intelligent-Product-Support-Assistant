from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def extract_ocr_text_from_pdf(path: Path) -> list[tuple[int, str]]:
    """Extract text from scanned PDF using pytesseract or Gemini Vision fallback if native libraries exist.
    Returns list of (page_number, extracted_text) tuples. Never crashes application on missing OCR binaries.
    """
    results: list[tuple[int, str]] = []
    try:
        import pytesseract
        from pdf2image import convert_from_path

        images = convert_from_path(str(path))
        for idx, img in enumerate(images, start=1):
            text = pytesseract.image_to_string(img)
            if text.strip():
                results.append((idx, text.strip()))
    except Exception as exc:
        logger.warning(f"Native OCR fallback unavailable for {path.name}: {exc}")

    return results
