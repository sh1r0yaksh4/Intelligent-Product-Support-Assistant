from __future__ import annotations

import argparse
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
    print("    .venv/bin/python3 scripts/reindex.py <product_name>\n")
    sys.exit(1)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag.config import slugify
from rag.grok import GrokUnavailable
from rag.indexer import ProductIndex
from rag.loader import load_file
from rag.sources import product_directory, source_urls


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild a product's local document index.")
    parser.add_argument("product", help="Product name used in the chat workspace")
    args = parser.parse_args()
    product_id = slugify(args.product)
    directory = product_directory(product_id)
    urls = source_urls(product_id)
    files = [path for path in directory.iterdir() if path.suffix.lower() in {".pdf", ".txt", ".md"}]
    chunks = [chunk for path in files for chunk in load_file(path, product_id, urls.get(path.name, ""))]
    try:
        count = ProductIndex().index_documents(chunks, product_id)
        print(f"Indexed {count} chunks for {args.product}.")
    except GrokUnavailable as exc:
        print(f"\n[Error] {exc}")
        print("Set your GROK_API_KEY in .env before indexing documents.\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
