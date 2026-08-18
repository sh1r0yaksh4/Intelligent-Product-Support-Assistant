from __future__ import annotations

import uuid
from pathlib import Path
from typing import List

from database.db import get_db
from database.models import ProductRecord, SourceRecord
from .config import slugify
from .indexer import ProductIndex
from .loader import load_file
from .sources import product_directory, source_urls


def create_product(manufacturer: str, name: str, model: str, hardware_version: str = "", description: str = "") -> ProductRecord:
    """Create a new product workspace entry in SQLite database."""
    product_id = slugify(name or f"{manufacturer}-{model}")
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """INSERT OR REPLACE INTO products (id, manufacturer, name, model, hardware_version, description)
               VALUES (?, ?, ?, ?, ?, ?);""",
            (product_id, manufacturer.strip(), name.strip(), model.strip(), hardware_version.strip(), description.strip()),
        )
    return ProductRecord(
        id=product_id,
        manufacturer=manufacturer,
        name=name,
        model=model,
        hardware_version=hardware_version,
        description=description,
    )


def list_products() -> list[ProductRecord]:
    """Return list of all registered products."""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id, manufacturer, name, model, hardware_version, description, created_at FROM products ORDER BY name ASC;")
        rows = cursor.fetchall()
        return [
            ProductRecord(
                id=r["id"],
                manufacturer=r["manufacturer"],
                name=r["name"],
                model=r["model"],
                hardware_version=r["hardware_version"],
                description=r["description"],
                created_at=r["created_at"],
            )
            for r in rows
        ]


def list_sources(product_id: str | None = None, status: str | None = None) -> list[SourceRecord]:
    """Retrieve filtered sources from database."""
    query = "SELECT id, product_id, source_name, source_type, source_url, file_path, status, chunk_count, created_at FROM sources WHERE 1=1"
    params = []
    if product_id:
        query += " AND product_id = ?"
        params.append(product_id)
    if status:
        query += " AND status = ?"
        params.append(status)
    query += " ORDER BY created_at DESC;"

    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute(query, params)
        rows = cursor.fetchall()
        return [
            SourceRecord(
                id=r["id"],
                product_id=r["product_id"],
                source_name=r["source_name"],
                source_type=r["source_type"],
                source_url=r["source_url"],
                file_path=r["file_path"],
                status=r["status"],
                chunk_count=r["chunk_count"],
                created_at=r["created_at"],
            )
            for r in rows
        ]


def approve_source(source_id: str, user_id: str = "admin") -> int:
    """Approve a pending source and index its chunks in Chroma vector store."""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id, product_id, source_name, file_path, source_url FROM sources WHERE id = ?;", (source_id,))
        source = cursor.fetchone()
        if not source:
            raise ValueError("Source record not found.")

        product_id = source["product_id"]
        file_path = Path(source["file_path"])

        # Fetch product metadata
        cursor.execute("SELECT manufacturer, name, model, hardware_version FROM products WHERE id = ?;", (product_id,))
        prod = cursor.fetchone()
        mfg = prod["manufacturer"] if prod else ""
        mdl = prod["model"] if prod else ""
        hver = prod["hardware_version"] if prod else ""

        # Chunk and embed
        urls = source_urls(product_id)
        chunks = load_file(
            path=file_path,
            product_id=product_id,
            source_url=source["source_url"] or urls.get(file_path.name, ""),
            manufacturer=mfg,
            model=mdl,
            hardware_version=hver,
            source_id=source_id,
            approval_status="APPROVED",
        )

        indexed_count = ProductIndex().index_documents(chunks, product_id, source_id)

        cursor.execute(
            "UPDATE sources SET status = 'APPROVED', chunk_count = ? WHERE id = ?;",
            (indexed_count, source_id),
        )
        cursor.execute(
            "INSERT INTO audit_logs (id, user_id, action, details) VALUES (?, ?, ?, ?);",
            (str(uuid.uuid4()), user_id, "APPROVE_SOURCE", f"Approved {source['source_name']} ({indexed_count} chunks)"),
        )
    return indexed_count


def reject_source(source_id: str, user_id: str = "admin") -> None:
    """Reject a source and remove any indexed chunks from Chroma vector store."""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT product_id, source_name FROM sources WHERE id = ?;", (source_id,))
        source = cursor.fetchone()
        if source:
            ProductIndex().delete_source_chunks(source["product_id"], source_id)
            cursor.execute("UPDATE sources SET status = 'REJECTED', chunk_count = 0 WHERE id = ?;", (source_id,))
            cursor.execute(
                "INSERT INTO audit_logs (id, user_id, action, details) VALUES (?, ?, ?, ?);",
                (str(uuid.uuid4()), user_id, "REJECT_SOURCE", f"Rejected {source['source_name']}"),
            )


def delete_source(source_id: str, user_id: str = "admin") -> None:
    """Delete a source file and its DB entry."""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT product_id, file_path, source_name FROM sources WHERE id = ?;", (source_id,))
        source = cursor.fetchone()
        if source:
            ProductIndex().delete_source_chunks(source["product_id"], source_id)
            if source["file_path"]:
                p = Path(source["file_path"])
                if p.exists():
                    p.unlink(missing_ok=True)
            cursor.execute("DELETE FROM sources WHERE id = ?;", (source_id,))
            cursor.execute(
                "INSERT INTO audit_logs (id, user_id, action, details) VALUES (?, ?, ?, ?);",
                (str(uuid.uuid4()), user_id, "DELETE_SOURCE", f"Deleted {source['source_name']}"),
            )
