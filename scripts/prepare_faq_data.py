"""Download MakTek and Bitext FAQ datasets, prepare, and write to data/faq/general_support_faqs.jsonl.

Usage:
    python3 scripts/prepare_faq_data.py

Requires network access on first run to download datasets from HuggingFace.
Uses only `requests` (already a project dependency) — no extra deps needed.
"""

from __future__ import annotations

import csv
import io
import json
import random
import sys
from pathlib import Path

import requests

ROOT_DIR = Path(__file__).resolve().parents[1]
FAQ_DIR = ROOT_DIR / "data" / "faq"
OUTPUT_FILE = FAQ_DIR / "general_support_faqs.jsonl"

# --- HuggingFace raw file URLs ---
MAKTEK_URL = (
    "https://huggingface.co/datasets/MakTek/Customer_support_faqs_dataset/resolve/main/train_expanded.json"
)
BITEXT_URL = (
    "https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset"
    "/resolve/main/Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv"
)

# Deterministic placeholder replacements for Bitext template tokens
PLACEHOLDER_MAP = {
    "{{Order Number}}": "#ORD-29481",
    "{{order_number}}": "#ORD-29481",
    "{{Invoice Number}}": "#INV-7723",
    "{{invoice_number}}": "#INV-7723",
    "{{Account Number}}": "ACC-881204",
    "{{account_number}}": "ACC-881204",
    "{{Tracking Number}}": "TRK-5519832",
    "{{tracking_number}}": "TRK-5519832",
    "{{Money Amount}}": "$45.99",
    "{{money_amount}}": "$45.99",
    "{{Delivery Address}}": "742 Evergreen Terrace, Springfield",
    "{{delivery_address}}": "742 Evergreen Terrace, Springfield",
    "{{Shipping Address}}": "742 Evergreen Terrace, Springfield",
    "{{shipping_address}}": "742 Evergreen Terrace, Springfield",
    "{{Email}}": "user@example.com",
    "{{email}}": "user@example.com",
    "{{Phone Number}}": "(555) 123-4567",
    "{{phone_number}}": "(555) 123-4567",
    "{{Product Name}}": "Wireless Bluetooth Headphones",
    "{{product_name}}": "Wireless Bluetooth Headphones",
    "{{Company Name}}": "our company",
    "{{company_name}}": "our company",
    "{{Store Name}}": "our store",
    "{{store_name}}": "our store",
    "{{Customer Name}}": "Alex",
    "{{customer_name}}": "Alex",
    "{{Name}}": "Alex",
    "{{name}}": "Alex",
    "{{Date}}": "August 5, 2025",
    "{{date}}": "August 5, 2025",
    "{{Promo Code}}": "SAVE20",
    "{{promo_code}}": "SAVE20",
    "{{Coupon Code}}": "SAVE20",
    "{{coupon_code}}": "SAVE20",
    "{{Refund Amount}}": "$45.99",
    "{{refund_amount}}": "$45.99",
}


def _replace_placeholders(text: str) -> str:
    """Replace all {{template}} tokens with realistic example values."""
    for placeholder, replacement in PLACEHOLDER_MAP.items():
        text = text.replace(placeholder, replacement)
    return text


def _download(url: str, label: str) -> bytes:
    """Download a file from a URL with progress indication."""
    print(f"  Downloading {label}...")
    response = requests.get(url, timeout=60, headers={"User-Agent": "ProductSupportAssistant/1.0"})
    response.raise_for_status()
    print(f"  Downloaded {len(response.content):,} bytes")
    return response.content


def prepare_maktek() -> list[dict]:
    """Download and normalize all 200 MakTek FAQ pairs."""
    raw = _download(MAKTEK_URL, "MakTek dataset")
    text = raw.decode("utf-8", errors="replace")

    # Try parsing as single JSON first, fall back to JSONL
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            if "train" in data:
                data = data["train"]
            elif "data" in data:
                data = data["data"]
            elif "question" in data and "answer" in data:
                questions = data["question"]
                answers = data["answer"]
                data = [{"question": q, "answer": a} for q, a in zip(questions, answers)]
    except json.JSONDecodeError:
        # JSONL format: one JSON object per line
        data = [json.loads(line) for line in text.splitlines() if line.strip()]

    pairs = []
    for item in data:
        q = (item.get("question") or item.get("instruction") or "").strip()
        a = (item.get("answer") or item.get("response") or "").strip()
        if q and a:
            pairs.append({
                "question": q,
                "answer": a,
                "category": "general",
                "intent": "",
                "source": "maktek",
            })

    print(f"  MakTek: {len(pairs)} pairs extracted")
    return pairs


def prepare_bitext(target_per_intent: int = 18) -> list[dict]:
    """Download Bitext, stratified sample ~18 per intent (~486 total), replace placeholders."""
    raw = _download(BITEXT_URL, "Bitext dataset")

    # Parse CSV
    text = raw.decode("utf-8", errors="replace")
    reader = csv.DictReader(io.StringIO(text))
    rows = list(reader)
    print(f"  Bitext: {len(rows)} total rows loaded")

    # Group by intent
    by_intent: dict[str, list[dict]] = {}
    for row in rows:
        intent = (row.get("intent") or "unknown").strip()
        by_intent.setdefault(intent, []).append(row)

    print(f"  Bitext: {len(by_intent)} intents found: {sorted(by_intent.keys())}")

    # Stratified sample
    random.seed(42)  # Reproducible
    sampled: list[dict] = []
    for intent, intent_rows in sorted(by_intent.items()):
        n = min(target_per_intent, len(intent_rows))
        chosen = random.sample(intent_rows, n)
        for row in chosen:
            q = _replace_placeholders((row.get("instruction") or "").strip())
            a = _replace_placeholders((row.get("response") or "").strip())
            category = (row.get("category") or "general").strip()
            if q and a:
                sampled.append({
                    "question": q,
                    "answer": a,
                    "category": category.lower(),
                    "intent": intent,
                    "source": "bitext",
                })

    print(f"  Bitext: {len(sampled)} pairs sampled ({target_per_intent} per intent)")
    return sampled


def main() -> None:
    print("Preparing general support FAQ data...\n")

    maktek_pairs = prepare_maktek()
    print()
    bitext_pairs = prepare_bitext()

    combined = maktek_pairs + bitext_pairs
    print(f"\nTotal combined: {len(combined)} pairs")

    # Write output
    FAQ_DIR.mkdir(parents=True, exist_ok=True)
    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        for pair in combined:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    print(f"Written to: {OUTPUT_FILE}")
    print("Done.")


if __name__ == "__main__":
    main()
