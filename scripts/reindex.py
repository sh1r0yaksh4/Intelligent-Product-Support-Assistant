from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag.config import slugify
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
    count = ProductIndex().index_documents(chunks, product_id)
    print(f"Indexed {count} chunks for {args.product}.")


if __name__ == "__main__":
    main()

