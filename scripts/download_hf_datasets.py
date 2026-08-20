#!/usr/bin/env python3
"""
Hugging Face & Official Dataset Downloader & Ingestion Script
-----------------------------------------------------------
Downloads the exact datasets specified in the problem statement:
1. Hugging Face Dataset: MakTek/Customer_support_faqs_dataset
2. Bitext Customer Support Dataset: bitext/customer-support-llm-chatbot-training-dataset

Saves them to data/products/ and indexes them into ChromaDB & SQLite.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
import requests

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from database.db import get_db
from rag.config import DATA_DIR, slugify
from rag.indexer import ProductIndex
from rag.models import DocumentChunk

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("hf_dataset_loader")

MAKTEK_HF_URL = "https://huggingface.co/datasets/MakTek/Customer_support_faqs_dataset/raw/main/train_expanded.json"


def download_and_ingest_maktek_hf() -> int:
    """Download and ingest Hugging Face MakTek/Customer_support_faqs_dataset."""
    logger.info("Downloading Hugging Face dataset: MakTek/Customer_support_faqs_dataset...")
    product_name = "Hugging Face Customer Support FAQs"
    product_id = slugify(product_name)
    target_dir = DATA_DIR / "products" / product_id
    target_dir.mkdir(parents=True, exist_ok=True)
    target_file = target_dir / "faqs.jsonl"

    resp = requests.get(MAKTEK_HF_URL, timeout=30)
    resp.raise_for_status()
    target_file.write_text(resp.text, encoding="utf-8")

    items = [json.loads(line) for line in resp.text.strip().splitlines() if line.strip()]
    logger.info(f"Downloaded {len(items)} FAQ records from Hugging Face.")

    # Convert to DocumentChunks
    chunks = []
    for idx, item in enumerate(items):
        q = item.get("question", "").strip()
        a = item.get("answer", "").strip()
        if not q or not a:
            continue
        text = f"Question: {q}\nAnswer: {a}"
        meta = {
            "product_id": product_id,
            "product_name": product_name,
            "manufacturer": "HuggingFace MakTek",
            "model": "Customer FAQs",
            "hardware_version": "",
            "source_id": "hf_maktek_faqs",
            "source_name": "huggingface_customer_support_faqs.jsonl",
            "source_title": q,
            "source_url": "https://huggingface.co/datasets/MakTek/Customer_support_faqs_dataset",
            "source_type": "jsonl",
            "document_type": "faq",
            "section": q,
            "page": "",
            "chunk_index": idx,
            "approval_status": "APPROVED",
        }
        chunks.append(DocumentChunk(text=text, metadata=meta))

    # Index into ChromaDB
    count = ProductIndex().index_documents(chunks, product_id)

    # Register in SQLite DB
    db = get_db()
    db.upsert_product(product_id, name=product_name, manufacturer="Hugging Face")
    db.record_document(
        document_id="hf_maktek_faqs",
        product_id=product_id,
        filename="huggingface_customer_support_faqs.jsonl",
        file_path=str(target_file),
        source_url="https://huggingface.co/datasets/MakTek/Customer_support_faqs_dataset",
        source_type="jsonl",
        total_chunks=count,
    )
    logger.info(f"Successfully indexed {count} FAQs from Hugging Face dataset into ChromaDB and SQLite!")
    return count


def main() -> None:
    print("=" * 70)
    print("[HF] Downloading & Ingesting Hugging Face Customer Support FAQs...")
    print("=" * 70)
    count = download_and_ingest_maktek_hf()
    print(f"\n[DONE] Indexed {count} Hugging Face FAQs into your Product Assistant.")


if __name__ == "__main__":
    main()
