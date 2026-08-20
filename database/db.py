from __future__ import annotations

import json
import sqlite3
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rag.config import DATA_DIR, ensure_directories

DB_PATH = DATA_DIR / "assistant.db"

_LOCAL = threading.local()


class Database:
    def __init__(self, db_path: Path = DB_PATH) -> None:
        self.db_path = db_path
        ensure_directories()
        self.init_db()

    def _get_connection(self) -> sqlite3.Connection:
        if not hasattr(_LOCAL, "connections"):
            _LOCAL.connections = {}
        path_str = str(self.db_path)
        if path_str not in _LOCAL.connections:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(path_str, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA foreign_keys=ON;")
            _LOCAL.connections[path_str] = conn
        return _LOCAL.connections[path_str]

    def init_db(self) -> None:
        conn = self._get_connection()
        with conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS products (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    manufacturer TEXT,
                    model TEXT,
                    description TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    product_id TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    file_path TEXT,
                    source_url TEXT,
                    source_type TEXT,
                    total_chunks INTEGER DEFAULT 0,
                    status TEXT DEFAULT 'INDEXED',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY (product_id) REFERENCES products(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    active_product TEXT,
                    archived INTEGER DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    citations_json TEXT,
                    escalated INTEGER DEFAULT 0,
                    used_search INTEGER DEFAULT 0,
                    used_memory INTEGER DEFAULT 0,
                    interaction_id TEXT,
                    feedback TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS interactions (
                    id TEXT PRIMARY KEY,
                    product_id TEXT NOT NULL,
                    product_name TEXT,
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    citations_json TEXT,
                    escalated INTEGER DEFAULT 0,
                    used_search INTEGER DEFAULT 0,
                    feedback TEXT,
                    review_status TEXT DEFAULT 'pending',
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS knowledge_chunks (
                    chunk_id TEXT PRIMARY KEY,
                    document_id TEXT,
                    product_id TEXT NOT NULL,
                    hardware_version TEXT,
                    section TEXT,
                    page INTEGER,
                    chunk_index INTEGER,
                    text_content TEXT,
                    created_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_docs_product ON documents(product_id);
                CREATE INDEX IF NOT EXISTS idx_msgs_conv ON messages(conversation_id);
                CREATE INDEX IF NOT EXISTS idx_interactions_prod ON interactions(product_id);
                CREATE INDEX IF NOT EXISTS idx_chunks_prod ON knowledge_chunks(product_id);
                """
            )

    # --- Product Operations ---
    def upsert_product(
        self, product_id: str, name: str, manufacturer: str = "", model: str = "", description: str = ""
    ) -> None:
        now = datetime.now(UTC).isoformat()
        conn = self._get_connection()
        with conn:
            conn.execute(
                """
                INSERT INTO products (id, name, manufacturer, model, description, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    name = excluded.name,
                    manufacturer = CASE WHEN excluded.manufacturer != '' THEN excluded.manufacturer ELSE products.manufacturer END,
                    model = CASE WHEN excluded.model != '' THEN excluded.model ELSE products.model END,
                    description = CASE WHEN excluded.description != '' THEN excluded.description ELSE products.description END,
                    updated_at = excluded.updated_at;
                """,
                (product_id, name, manufacturer, model, description, now, now),
            )

    def get_product(self, product_id: str) -> dict[str, Any] | None:
        conn = self._get_connection()
        cursor = conn.execute("SELECT * FROM products WHERE id = ?", (product_id,))
        row = cursor.fetchone()
        return dict(row) if row else None

    def list_products(self) -> list[dict[str, Any]]:
        conn = self._get_connection()
        cursor = conn.execute("SELECT * FROM products ORDER BY name ASC")
        return [dict(row) for row in cursor.fetchall()]

    # --- Document Operations ---
    def record_document(
        self,
        document_id: str,
        product_id: str,
        filename: str,
        file_path: str = "",
        source_url: str = "",
        source_type: str = "",
        total_chunks: int = 0,
    ) -> None:
        now = datetime.now(UTC).isoformat()
        conn = self._get_connection()
        with conn:
            conn.execute(
                """
                INSERT INTO documents (id, product_id, filename, file_path, source_url, source_type, total_chunks, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    total_chunks = excluded.total_chunks,
                    file_path = excluded.file_path,
                    source_url = excluded.source_url,
                    source_type = excluded.source_type,
                    updated_at = excluded.updated_at;
                """,
                (document_id, product_id, filename, file_path, source_url, source_type, total_chunks, now, now),
            )

    def list_documents(self, product_id: str | None = None) -> list[dict[str, Any]]:
        conn = self._get_connection()
        if product_id:
            cursor = conn.execute("SELECT * FROM documents WHERE product_id = ? ORDER BY created_at DESC", (product_id,))
        else:
            cursor = conn.execute("SELECT * FROM documents ORDER BY created_at DESC")
        return [dict(row) for row in cursor.fetchall()]

    # --- Conversation Operations ---
    def upsert_conversation(self, conv_id: str, title: str, active_product: str | None = None, archived: bool = False) -> None:
        now = datetime.now(UTC).isoformat()
        conn = self._get_connection()
        with conn:
            conn.execute(
                """
                INSERT INTO conversations (id, title, active_product, archived, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    title = excluded.title,
                    active_product = excluded.active_product,
                    archived = excluded.archived,
                    updated_at = excluded.updated_at;
                """,
                (conv_id, title, active_product, 1 if archived else 0, now, now),
            )

    def get_conversation(self, conv_id: str) -> dict[str, Any] | None:
        conn = self._get_connection()
        cursor = conn.execute("SELECT * FROM conversations WHERE id = ?", (conv_id,))
        row = cursor.fetchone()
        if not row:
            return None
        conv = dict(row)
        conv["archived"] = bool(conv["archived"])
        conv["messages"] = self.get_messages(conv_id)
        return conv

    def list_conversations(self, include_archived: bool = False) -> list[dict[str, Any]]:
        conn = self._get_connection()
        if include_archived:
            cursor = conn.execute("SELECT * FROM conversations ORDER BY updated_at DESC")
        else:
            cursor = conn.execute("SELECT * FROM conversations WHERE archived = 0 ORDER BY updated_at DESC")
        results = []
        for row in cursor.fetchall():
            conv = dict(row)
            conv["archived"] = bool(conv["archived"])
            conv["messages"] = self.get_messages(conv["id"])
            results.append(conv)
        return results

    def delete_conversation(self, conv_id: str) -> None:
        conn = self._get_connection()
        with conn:
            conn.execute("DELETE FROM conversations WHERE id = ?", (conv_id,))

    # --- Message Operations ---
    def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        citations: list[dict] | None = None,
        escalated: bool = False,
        used_search: bool = False,
        used_memory: bool = False,
        interaction_id: str | None = None,
    ) -> int:
        now = datetime.now(UTC).isoformat()
        citations_str = json.dumps(citations or [])
        conn = self._get_connection()
        with conn:
            cursor = conn.execute(
                """
                INSERT INTO messages (conversation_id, role, content, citations_json, escalated, used_search, used_memory, interaction_id, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (conversation_id, role, content, citations_str, 1 if escalated else 0, 1 if used_search else 0, 1 if used_memory else 0, interaction_id, now),
            )
            conn.execute("UPDATE conversations SET updated_at = ? WHERE id = ?", (now, conversation_id))
            return cursor.lastrowid or 0

    def get_messages(self, conversation_id: str) -> list[dict[str, Any]]:
        conn = self._get_connection()
        cursor = conn.execute(
            "SELECT * FROM messages WHERE conversation_id = ? ORDER BY id ASC", (conversation_id,)
        )
        messages = []
        for row in cursor.fetchall():
            msg = dict(row)
            msg["citations"] = json.loads(msg["citations_json"] or "[]")
            msg["escalated"] = bool(msg["escalated"])
            msg["used_search"] = bool(msg["used_search"])
            msg["used_memory"] = bool(msg["used_memory"])
            messages.append(msg)
        return messages

    # --- Interaction & Feedback Operations ---
    def record_interaction(
        self,
        interaction_id: str,
        product_id: str,
        product_name: str | None,
        question: str,
        answer: str,
        citations: list[dict],
        escalated: bool,
        used_search: bool,
    ) -> None:
        now = datetime.now(UTC).isoformat()
        conn = self._get_connection()
        with conn:
            conn.execute(
                """
                INSERT INTO interactions (id, product_id, product_name, question, answer, citations_json, escalated, used_search, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    answer = excluded.answer,
                    citations_json = excluded.citations_json,
                    escalated = excluded.escalated,
                    used_search = excluded.used_search;
                """,
                (
                    interaction_id,
                    product_id,
                    product_name or "",
                    question,
                    answer,
                    json.dumps(citations),
                    1 if escalated else 0,
                    1 if used_search else 0,
                    now,
                ),
            )

    def update_interaction_feedback(self, interaction_id: str, feedback: str, review_status: str) -> bool:
        conn = self._get_connection()
        with conn:
            cursor = conn.execute(
                """
                UPDATE interactions SET feedback = ?, review_status = ? WHERE id = ?
                """,
                (feedback, review_status, interaction_id),
            )
            return cursor.rowcount > 0


_DB_INSTANCE: Database | None = None


def get_db() -> Database:
    global _DB_INSTANCE
    if _DB_INSTANCE is None:
        _DB_INSTANCE = Database()
    return _DB_INSTANCE
