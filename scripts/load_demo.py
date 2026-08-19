"""Load bundled example product documents into the local index for demo purposes."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

# Provide helpful guidance if run outside the virtual environment
try:
    import chromadb
    import dotenv
except ImportError:
    print("\n[Error] Dependencies not found in current Python environment.")
    print("Please activate the virtual environment first:")
    print("    source .venv/bin/activate")
    print("Or run directly using:")
    print("    .venv/bin/python3 scripts/load_demo.py\n")
    sys.exit(1)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag.config import slugify
from rag.gemini import GeminiUnavailable
from rag.indexer import ProductIndex
from rag.loader import load_file
from rag.sources import product_directory

EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "data" / "examples"

DEMO_PRODUCTS = [
    {"name": "TP-Link Archer AX21", "file": "tp-link-archer-ax21.md"},
]


def main() -> None:
    if not EXAMPLES_DIR.exists():
        print(f"Examples directory not found: {EXAMPLES_DIR}")
        sys.exit(1)

    for product in DEMO_PRODUCTS:
        name = product["name"]
        product_id = slugify(name)
        source_path = EXAMPLES_DIR / product["file"]

        if not source_path.exists():
            print(f"  Skipping {name}: {source_path.name} not found.")
            continue

        # Copy into the product workspace
        dest_dir = product_directory(product_id)
        dest_path = dest_dir / source_path.name
        shutil.copy2(source_path, dest_path)

        # Chunk and index
        chunks = load_file(dest_path, product_id)
        try:
            count = ProductIndex().index_documents(chunks, product_id)
            print(f"  Indexed {count} chunks for {name}.")
        except GeminiUnavailable as exc:
            print(f"\n[Error] {exc}")
            print("Set your GEMINI_API_KEY in .env before indexing documents.\n")
            sys.exit(1)

    print("Demo products loaded. Run: streamlit run app.py")


if __name__ == "__main__":
    main()
