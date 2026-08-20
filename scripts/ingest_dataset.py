"""Comprehensive, idempotent dataset ingestion pipeline for Intelligent Product Support Assistant.

Supports:
- CSV (e.g. question/answer FAQ datasets, customer support records)
- JSON (arrays of documents or QA pairs)
- JSONL (line-delimited JSON objects)
- Markdown (.md) and Plaintext (.txt) manuals
- PDF (.pdf) user guides

Features:
- Schema validation & normalization
- Duplicate detection & text cleaning
- Metadata enrichment (manufacturer, model, hardware_version, source_type, category)
- Intelligent chunking
- Idempotent ChromaDB vector indexing and SQLite document recording
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from database.db import get_db
from rag.config import slugify
from rag.grok import GrokUnavailable
from rag.indexer import ProductIndex
from rag.loader import clean_text, load_file
from rag.models import DocumentChunk
from rag.sources import product_directory

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def normalize_record_text(raw_text: str) -> str:
    """Normalize whitespace and strip noisy formatting."""
    return clean_text(raw_text)


def load_csv_dataset(
    file_path: Path,
    product_name: str,
    product_id: str,
    hardware_version: str = "",
    manufacturer: str = "",
    model: str = "",
) -> list[DocumentChunk]:
    """Parse CSV dataset with flexible header detection (question/answer, text, title/body)."""
    chunks: list[DocumentChunk] = []
    seen_texts: set[str] = set()

    with file_path.open("r", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return []

        # Normalize field names to lowercase
        field_map = {fn.lower().strip(): fn for fn in reader.fieldnames if fn}

        q_key = next((field_map[k] for k in ["question", "query", "instruction", "prompt", "title"] if k in field_map), None)
        a_key = next((field_map[k] for k in ["answer", "response", "output", "body", "solution", "content"] if k in field_map), None)
        text_key = next((field_map[k] for k in ["text", "chunk", "content", "document"] if k in field_map), None)
        hw_key = next((field_map[k] for k in ["hardware_version", "hw_version", "version", "revision"] if k in field_map), None)
        prod_key = next((field_map[k] for k in ["product", "product_name", "model"] if k in field_map), None)
        cat_key = next((field_map[k] for k in ["category", "topic", "tag", "domain"] if k in field_map), None)

        for idx, row in enumerate(reader):
            hw_ver = row.get(hw_key, "").strip() if hw_key else hardware_version
            p_name = row.get(prod_key, "").strip() if prod_key else product_name
            cat = row.get(cat_key, "").strip() if cat_key else "General Support"

            if q_key and a_key and row.get(q_key) and row.get(a_key):
                q = normalize_record_text(row[q_key])
                a = normalize_record_text(row[a_key])
                if not q or not a or len(q) < 5:
                    continue
                content = f"Question: {q}\nAnswer: {a}"
                section = q[:60]
            elif text_key and row.get(text_key):
                content = normalize_record_text(row[text_key])
                if not content or len(content) < 15:
                    continue
                section = f"Record {idx + 1}"
            else:
                continue

            # Deduplication
            content_key = content.lower()
            if content_key in seen_texts:
                continue
            seen_texts.add(content_key)

            metadata = {
                "product_id": slugify(p_name) if p_name else product_id,
                "product_name": p_name or product_name,
                "manufacturer": manufacturer,
                "model": model,
                "hardware_version": hw_ver,
                "source_id": file_path.stem,
                "source_name": file_path.name,
                "source_title": file_path.stem.replace("_", " "),
                "source_url": "",
                "source_type": "csv",
                "document_type": "faq" if (q_key and a_key) else "dataset",
                "category": cat,
                "section": section,
                "chunk_index": idx,
                "approval_status": "APPROVED",
            }
            chunks.append(DocumentChunk(text=content, metadata=metadata))

    return chunks


def load_json_or_jsonl(
    file_path: Path,
    product_name: str,
    product_id: str,
    hardware_version: str = "",
    manufacturer: str = "",
    model: str = "",
) -> list[DocumentChunk]:
    """Parse JSON or JSONL records into clean DocumentChunk objects."""
    chunks: list[DocumentChunk] = []
    seen_texts: set[str] = set()

    is_jsonl = file_path.suffix.lower() == ".jsonl"
    records: list[dict[str, Any]] = []

    try:
        content = file_path.read_text(encoding="utf-8", errors="ignore")
        if is_jsonl:
            for line in content.splitlines():
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        else:
            parsed = json.loads(content)
            if isinstance(parsed, list):
                records = parsed
            elif isinstance(parsed, dict) and "data" in parsed and isinstance(parsed["data"], list):
                records = parsed["data"]
            else:
                records = [parsed]
    except Exception as exc:
        logger.error(f"Failed to read JSON/JSONL file {file_path.name}: {exc}")
        return []

    for idx, rec in enumerate(records):
        if not isinstance(rec, dict):
            continue

        q = normalize_record_text(str(rec.get("question") or rec.get("instruction") or rec.get("query") or ""))
        a = normalize_record_text(str(rec.get("answer") or rec.get("response") or rec.get("output") or rec.get("text") or ""))
        hw_ver = str(rec.get("hardware_version") or rec.get("version") or hardware_version).strip()
        p_name = str(rec.get("product_name") or rec.get("product") or product_name).strip()
        cat = str(rec.get("category") or rec.get("topic") or "General").strip()

        if q and a:
            text_body = f"Question: {q}\nAnswer: {a}"
            section = q[:60]
        elif a:
            text_body = a
            section = f"Section {idx + 1}"
        else:
            continue

        if len(text_body) < 15 or text_body.lower() in seen_texts:
            continue
        seen_texts.add(text_body.lower())

        metadata = {
            "product_id": slugify(p_name) if p_name else product_id,
            "product_name": p_name or product_name,
            "manufacturer": manufacturer,
            "model": model,
            "hardware_version": hw_ver,
            "source_id": file_path.stem,
            "source_name": file_path.name,
            "source_title": file_path.stem.replace("_", " "),
            "source_url": str(rec.get("url") or ""),
            "source_type": "json" if not is_jsonl else "jsonl",
            "document_type": "faq" if q else "document",
            "category": cat,
            "section": section,
            "chunk_index": idx,
            "approval_status": "APPROVED",
        }
        chunks.append(DocumentChunk(text=text_body, metadata=metadata))

    return chunks


def ingest_file(
    file_path: Path,
    product_name: str,
    hardware_version: str = "",
    manufacturer: str = "",
    model: str = "",
) -> int:
    """Ingest a single file (CSV, JSON, JSONL, MD, TXT, PDF) idempotently."""
    if not file_path.exists():
        logger.error(f"File not found: {file_path}")
        return 0

    product_id = slugify(product_name)
    ext = file_path.suffix.lower()

    if ext == ".csv":
        chunks = load_csv_dataset(file_path, product_name, product_id, hardware_version, manufacturer, model)
    elif ext in {".json", ".jsonl"}:
        chunks = load_json_or_jsonl(file_path, product_name, product_id, hardware_version, manufacturer, model)
    elif ext in {".pdf", ".txt", ".md"}:
        chunks = load_file(
            file_path,
            product_id=product_id,
            manufacturer=manufacturer,
            model=model,
            hardware_version=hardware_version,
            source_id=file_path.stem,
        )
    else:
        logger.warning(f"Unsupported file format: {ext}")
        return 0

    if not chunks:
        logger.warning(f"No valid chunks extracted from {file_path.name}")
        return 0

    # Record in SQLite Database
    db = get_db()
    db.upsert_product(product_id, product_name, manufacturer=manufacturer, model=model)
    db.record_document(
        document_id=file_path.stem,
        product_id=product_id,
        filename=file_path.name,
        file_path=str(file_path),
        source_type=ext.lstrip("."),
        total_chunks=len(chunks),
    )

    # Index in ChromaDB
    try:
        indexed_count = ProductIndex().index_documents(chunks, product_id, source_id=file_path.stem)
        logger.info(f"Successfully indexed {indexed_count} chunks for '{product_name}' from {file_path.name}")
        return indexed_count
    except GrokUnavailable as exc:
        logger.error(f"Embedding error: {exc}")
        raise


def ingest_directory(dir_path: Path, product_name: str | None = None) -> int:
    """Ingest all supported files in a directory."""
    if not dir_path.exists():
        logger.error(f"Directory not found: {dir_path}")
        return 0

    total_indexed = 0
    for file_path in dir_path.glob("**/*"):
        if file_path.is_file() and file_path.suffix.lower() in {".pdf", ".txt", ".md", ".csv", ".json", ".jsonl"}:
            if file_path.name.startswith("."):
                continue
            # Determine product name from parent dir if not provided
            p_name = product_name or file_path.parent.name.replace("-", " ").title()
            if p_name.lower() in {"products", "data", "examples"}:
                p_name = file_path.stem.replace("-", " ").replace("_", " ").title()

            try:
                count = ingest_file(file_path, p_name)
                total_indexed += count
            except Exception as exc:
                logger.error(f"Failed to ingest {file_path.name}: {exc}")

    return total_indexed


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest datasets and product documents into the vector store and database.")
    parser.add_argument("--file", "-f", type=Path, help="Path to single file to ingest")
    parser.add_argument("--dir", "-d", type=Path, help="Path to directory of files to ingest")
    parser.add_argument("--product", "-p", type=str, default="", help="Product name")
    parser.add_argument("--version", "-v", type=str, default="", help="Hardware version (e.g. V1, V2)")
    parser.add_argument("--manufacturer", "-m", type=str, default="", help="Manufacturer name")
    parser.add_argument("--model", type=str, default="", help="Product model")

    args = parser.parse_args()

    if args.file:
        product_name = args.product or args.file.stem.replace("-", " ").title()
        ingest_file(
            args.file,
            product_name,
            hardware_version=args.version,
            manufacturer=args.manufacturer,
            model=args.model,
        )
    elif args.dir:
        ingest_directory(args.dir, product_name=args.product or None)
    else:
        # Default: Ingest all products in data/products and data/examples
        logger.info("Ingesting all bundled datasets and product manuals...")
        products_dir = ROOT_DIR / "data" / "products"
        examples_dir = ROOT_DIR / "data" / "examples"

        count1 = ingest_directory(examples_dir) if examples_dir.exists() else 0
        count2 = ingest_directory(products_dir) if products_dir.exists() else 0
        logger.info(f"Ingestion complete. Total chunks indexed: {count1 + count2}")


if __name__ == "__main__":
    main()
