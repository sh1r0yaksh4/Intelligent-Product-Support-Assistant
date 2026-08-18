from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from database.db import init_db
from rag.admin import create_product
from rag.config import slugify
from rag.indexer import ProductIndex
from rag.loader import load_file
from rag.sources import product_directory, source_urls


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild a product's local document index.")
    parser.add_argument("product", help="Product name used in the chat workspace")
    parser.add_argument("--manufacturer", default="Generic", help="Manufacturer name")
    parser.add_argument("--model", default="", help="Model number")
    parser.add_argument("--version", default="", help="Hardware version")
    args = parser.parse_args()

    init_db()
    product_id = slugify(args.product)
    create_product(args.manufacturer, args.product, args.model or args.product, args.version)

    directory = product_directory(product_id)
    urls = source_urls(product_id)
    files = [path for path in directory.iterdir() if path.suffix.lower() in {".pdf", ".txt", ".md"}]

    chunks = [
        chunk
        for path in files
        for chunk in load_file(
            path=path,
            product_id=product_id,
            source_url=urls.get(path.name, ""),
            manufacturer=args.manufacturer,
            model=args.model or args.product,
            hardware_version=args.version,
            approval_status="APPROVED",
        )
    ]

    count = ProductIndex().index_documents(chunks, product_id)
    print(f"Successfully indexed {count} chunks for {args.product} (ID: {product_id}).")


if __name__ == "__main__":
    main()
