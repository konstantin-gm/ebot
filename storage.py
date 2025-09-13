import os
import sqlite3
from datetime import datetime, timezone
from typing import Optional, List, Tuple, Dict, Any


def _connect(db_path: str) -> sqlite3.Connection:
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: str) -> None:
    conn = _connect(db_path)
    try:
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                user_id INTEGER PRIMARY KEY,
                chat_id INTEGER NOT NULL,
                tariff REAL DEFAULT 0.0,
                created_at TEXT NOT NULL
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS readings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                month_key TEXT NOT NULL,
                reading_value REAL,
                photo_file_id TEXT,
                tariff_applied REAL,
                entered_at TEXT NOT NULL,
                UNIQUE(user_id, month_key),
                FOREIGN KEY(user_id) REFERENCES users(user_id)
            )
            """
        )
        conn.commit()
    finally:
        conn.close()


def get_or_create_user(db_path: str, user_id: int, chat_id: int) -> Dict[str, Any]:
    conn = _connect(db_path)
    try:
        cur = conn.cursor()
        cur.execute("SELECT * FROM users WHERE user_id=?", (user_id,))
        row = cur.fetchone()
        if row:
            # Update chat_id if changed
            if row["chat_id"] != chat_id:
                cur.execute("UPDATE users SET chat_id=? WHERE user_id=?", (chat_id, user_id))
                conn.commit()
            return dict(row)
        now = datetime.now(timezone.utc).isoformat()
        cur.execute(
            "INSERT INTO users (user_id, chat_id, tariff, created_at) VALUES (?, ?, ?, ?)",
            (user_id, chat_id, 0.0, now),
        )
        conn.commit()
        return {
            "user_id": user_id,
            "chat_id": chat_id,
            "tariff": 0.0,
            "created_at": now,
        }
    finally:
        conn.close()


def set_tariff(db_path: str, user_id: int, tariff: float) -> None:
    conn = _connect(db_path)
    try:
        cur = conn.cursor()
        cur.execute("UPDATE users SET tariff=? WHERE user_id=?", (tariff, user_id))
        conn.commit()
    finally:
        conn.close()


def get_tariff(db_path: str, user_id: int) -> float:
    conn = _connect(db_path)
    try:
        cur = conn.cursor()
        cur.execute("SELECT tariff FROM users WHERE user_id=?", (user_id,))
        row = cur.fetchone()
        return float(row[0]) if row else 0.0
    finally:
        conn.close()


def month_key_for(dt: datetime) -> str:
    return f"{dt.year:04d}-{dt.month:02d}-01"


def record_reading(
    db_path: str,
    user_id: int,
    month_key: str,
    reading_value: Optional[float],
    photo_file_id: Optional[str],
    tariff_applied: Optional[float],
) -> None:
    conn = _connect(db_path)
    try:
        cur = conn.cursor()
        now = datetime.now(timezone.utc).isoformat()
        # Upsert by user_id + month_key
        cur.execute(
            "SELECT id FROM readings WHERE user_id=? AND month_key=?",
            (user_id, month_key),
        )
        row = cur.fetchone()
        if row:
            cur.execute(
                """
                UPDATE readings
                SET reading_value=COALESCE(?, reading_value),
                    photo_file_id=COALESCE(?, photo_file_id),
                    tariff_applied=COALESCE(?, tariff_applied),
                    entered_at=?
                WHERE id=?
                """,
                (reading_value, photo_file_id, tariff_applied, now, row[0]),
            )
        else:
            cur.execute(
                """
                INSERT INTO readings (user_id, month_key, reading_value, photo_file_id, tariff_applied, entered_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (user_id, month_key, reading_value, photo_file_id, tariff_applied, now),
            )
        conn.commit()
    finally:
        conn.close()


def get_reading_for_month(db_path: str, user_id: int, month_key: str) -> Optional[Dict[str, Any]]:
    conn = _connect(db_path)
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT * FROM readings WHERE user_id=? AND month_key=?",
            (user_id, month_key),
        )
        row = cur.fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def get_last_reading_before_month(db_path: str, user_id: int, month_key: str) -> Optional[Dict[str, Any]]:
    conn = _connect(db_path)
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT * FROM readings
            WHERE user_id=? AND month_key < ? AND reading_value IS NOT NULL
            ORDER BY month_key DESC
            LIMIT 1
            """,
            (user_id, month_key),
        )
        row = cur.fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def get_history(db_path: str, user_id: int, limit: int = 12) -> List[Dict[str, Any]]:
    conn = _connect(db_path)
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT * FROM readings
            WHERE user_id=?
            ORDER BY month_key DESC
            LIMIT ?
            """,
            (user_id, limit),
        )
        rows = cur.fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def list_users(db_path: str) -> List[Dict[str, Any]]:
    conn = _connect(db_path)
    try:
        cur = conn.cursor()
        cur.execute("SELECT * FROM users")
        rows = cur.fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()

