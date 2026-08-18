from __future__ import annotations

import hashlib
import uuid
from typing import Optional

from database.db import get_db, init_db
from database.models import UserRecord


def hash_password(password: str) -> str:
    """Hash password securely using SHA-256 with salt."""
    salt = "product_support_assistant_salt_2026"
    return hashlib.sha256((password + salt).encode("utf-8")).hexdigest()


def seed_default_users() -> None:
    """Seed default Admin and Customer accounts if not present."""
    init_db()
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM users;")
        count = cursor.fetchone()[0]
        if count == 0:
            admin_id = str(uuid.uuid4())
            admin_pass = hash_password("admin123")
            cursor.execute(
                "INSERT INTO users (id, username, password_hash, role) VALUES (?, ?, ?, ?);",
                (admin_id, "admin", admin_pass, "ADMIN"),
            )
            customer_id = str(uuid.uuid4())
            customer_pass = hash_password("customer123")
            cursor.execute(
                "INSERT INTO users (id, username, password_hash, role) VALUES (?, ?, ?, ?);",
                (customer_id, "customer", customer_pass, "CUSTOMER"),
            )


def authenticate_user(username: str, password_raw: str) -> Optional[UserRecord]:
    """Authenticate username and password against SQLite database."""
    seed_default_users()
    hashed = hash_password(password_raw)
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, username, password_hash, role, created_at FROM users WHERE username = ? AND password_hash = ?;",
            (username.strip().lower(), hashed),
        )
        row = cursor.fetchone()
        if row:
            return UserRecord(
                id=row["id"],
                username=row["username"],
                password_hash=row["password_hash"],
                role=row["role"],
                created_at=row["created_at"],
            )
    return None


def register_user(username: str, password_raw: str, role: str = "CUSTOMER") -> UserRecord:
    """Register a new user in SQLite database."""
    seed_default_users()
    hashed = hash_password(password_raw)
    user_id = str(uuid.uuid4())
    clean_username = username.strip().lower()
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO users (id, username, password_hash, role) VALUES (?, ?, ?, ?);",
            (user_id, clean_username, hashed, role.upper()),
        )
    return UserRecord(id=user_id, username=clean_username, password_hash=hashed, role=role.upper())
