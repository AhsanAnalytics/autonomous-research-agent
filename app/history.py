import sqlite3
import json
from datetime import datetime, timezone

DB_PATH = "runs.db"   # a single-file SQLite database in the project root


def _connect():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row   # lets us read columns by name, like a dict
    return conn


def init_db():
    """Create the runs table if it does not exist yet. Safe to call anytime."""
    with _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS runs (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                goal       TEXT NOT NULL,
                answer     TEXT,
                steps      INTEGER,
                detail     TEXT
            )
            """
        )


def save_run(goal, answer, steps=None, detail=None):
    """Insert one run; return its new id."""
    init_db()
    with _connect() as conn:
        cur = conn.execute(
            "INSERT INTO runs (created_at, goal, answer, steps, detail) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                datetime.now(timezone.utc).isoformat(timespec="seconds"),
                goal,
                answer,
                steps,
                json.dumps(detail) if detail is not None else None,
            ),
        )
        return cur.lastrowid


def list_runs(limit=10):
    """Return the most recent runs, newest first."""
    init_db()
    with _connect() as conn:
        rows = conn.execute(
            "SELECT id, created_at, goal, answer, steps "
            "FROM runs ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]
