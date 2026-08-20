"""Load bundled example product documents into the local index and database for demo purposes."""

from __future__ import annotations

import sys
from pathlib import Path

# Provide helpful guidance if run outside the virtual environment
try:
    import chromadb
    import dotenv
except ImportError:
    print("\n[Error] Dependencies not found in current Python environment.")
    print("Please activate the virtual environment first:")
    print("    .venv\\Scripts\\activate (Windows) or source .venv/bin/activate")
    print("Or run directly using:")
    print("    .venv/bin/python3 scripts/load_demo.py\n")
    sys.exit(1)

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from rag.grok import GrokUnavailable
from scripts.ingest_dataset import ingest_file

DEMO_DATASETS = [
    {
        "name": "TP-Link Archer AX21",
        "file": ROOT_DIR / "data" / "products" / "tp-link-archer-ax21" / "tp-link-archer-ax21.md",
        "manufacturer": "TP-Link",
        "model": "Archer AX21",
    },
    {
        "name": "Sony WH-1000XM5",
        "file": ROOT_DIR / "data" / "products" / "sony-wh-1000xm5" / "manual.md",
        "manufacturer": "Sony",
        "model": "WH-1000XM5",
    },
    {
        "name": "Ecobee Smart Thermostat Premium",
        "file": ROOT_DIR / "data" / "products" / "ecobee-smart-thermostat" / "manual.md",
        "manufacturer": "Ecobee",
        "model": "Smart Thermostat Premium",
    },
    {
        "name": "General Electronics & Networking FAQs",
        "file": ROOT_DIR / "data" / "products" / "general-support-faqs" / "faqs.csv",
        "manufacturer": "General",
        "model": "Support FAQs",
    },
]


def main() -> None:
    print("\n=== Initializing Demo Product Knowledge Base ===")
    total_chunks = 0

    for item in DEMO_DATASETS:
        name = item["name"]
        file_path = item["file"]

        if not file_path.exists():
            print(f"  [Skip] {name}: File {file_path.name} not found.")
            continue

        try:
            count = ingest_file(
                file_path=file_path,
                product_name=name,
                manufacturer=item["manufacturer"],
                model=item["model"],
            )
            print(f"  [Indexed] {name}: {count} chunks indexed.")
            total_chunks += count
        except GrokUnavailable as exc:
            print(f"\n[Error] {exc}")
            print("Set your GROK_API_KEY in .env before indexing documents.\n")
            sys.exit(1)
        except Exception as exc:
            print(f"  [Error] Failed to index {name}: {exc}")

    print(f"\nDemo knowledge base initialized ({total_chunks} total chunks indexed).")
    print("Run UI with: streamlit run app.py\n")


if __name__ == "__main__":
    main()
