"""Local SQLite persistence for signed-in travellers.

Only what a signed-in user needs is stored: their identity and the ordered list
of cities they have visited. No tokens, no recommendation logs, no analytics.

SQLite rather than PostgreSQL is a deliberate choice, in line with the project
brief: the data is a handful of rows per user, it lives beside the parquet
dataset, and requiring a database server would add an operational dependency to
a project whose selling point is that it runs on a laptop with nothing else
installed. ``docker-compose.yml`` still ships an optional Postgres profile for
anyone who wants it.

Anonymous use never touches this module -- the flow keeps its history in
browser memory, so the product works fully without sign-in.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from src.utils.logging_utils import get_logger

LOGGER = get_logger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    sub            TEXT PRIMARY KEY,
    email          TEXT,
    name           TEXT,
    picture        TEXT,
    created_at     TEXT NOT NULL DEFAULT (datetime('now')),
    last_seen_at   TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS trips (
    sub            TEXT NOT NULL,
    destination_id TEXT NOT NULL,
    position       INTEGER NOT NULL,
    added_at       TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (sub, destination_id),
    FOREIGN KEY (sub) REFERENCES users(sub) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS trips_by_user ON trips(sub, position);

CREATE TABLE IF NOT EXISTS preferences (
    sub            TEXT PRIMARY KEY,
    interests      TEXT NOT NULL DEFAULT '',
    duration_days  INTEGER,
    budget         TEXT,
    updated_at     TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (sub) REFERENCES users(sub) ON DELETE CASCADE
);
"""


class TravellerStore:
    """A tiny SQLite-backed store for user identity, trips and preferences."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript(SCHEMA)
        LOGGER.info("Traveller store ready at %s", self.path)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        # Enforce the ON DELETE CASCADE above; SQLite ignores it otherwise.
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    # ------------------------------------------------------------- users
    def upsert_user(self, user: Dict[str, Any]) -> None:
        """Create or refresh a user record from verified ID-token claims."""
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO users (sub, email, name, picture)
                VALUES (:sub, :email, :name, :picture)
                ON CONFLICT(sub) DO UPDATE SET
                    email = excluded.email,
                    name = excluded.name,
                    picture = excluded.picture,
                    last_seen_at = datetime('now')
                """,
                {
                    "sub": user["sub"],
                    "email": user.get("email", ""),
                    "name": user.get("name", ""),
                    "picture": user.get("picture", ""),
                },
            )

    def get_user(self, sub: str) -> Optional[Dict[str, Any]]:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM users WHERE sub = ?", (sub,)).fetchone()
        return dict(row) if row else None

    def delete_user(self, sub: str) -> None:
        """Remove a user and everything belonging to them."""
        with self._connect() as connection:
            connection.execute("DELETE FROM users WHERE sub = ?", (sub,))

    # ------------------------------------------------------------- trips
    def get_trips(self, sub: str) -> List[str]:
        """Return the user's destination ids in the order they added them."""
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT destination_id FROM trips WHERE sub = ? ORDER BY position", (sub,)
            ).fetchall()
        return [row["destination_id"] for row in rows]

    def set_trips(self, sub: str, destination_ids: List[str]) -> List[str]:
        """Replace the user's history with ``destination_ids``.

        Replacing wholesale rather than diffing keeps ordering unambiguous:
        the client owns the list, and position is simply the index.
        """
        unique: List[str] = list(dict.fromkeys(destination_ids))
        with self._connect() as connection:
            connection.execute("DELETE FROM trips WHERE sub = ?", (sub,))
            connection.executemany(
                "INSERT INTO trips (sub, destination_id, position) VALUES (?, ?, ?)",
                [(sub, destination_id, index) for index, destination_id in enumerate(unique)],
            )
        return unique

    # ------------------------------------------------------- preferences
    def get_preferences(self, sub: str) -> Dict[str, Any]:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM preferences WHERE sub = ?", (sub,)
            ).fetchone()
        if not row:
            return {"interests": [], "duration_days": None, "budget": None}
        return {
            "interests": [i for i in (row["interests"] or "").split(",") if i],
            "duration_days": row["duration_days"],
            "budget": row["budget"],
        }

    def set_preferences(
        self,
        sub: str,
        *,
        interests: Optional[List[str]] = None,
        duration_days: Optional[int] = None,
        budget: Optional[str] = None,
    ) -> Dict[str, Any]:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO preferences (sub, interests, duration_days, budget)
                VALUES (:sub, :interests, :duration_days, :budget)
                ON CONFLICT(sub) DO UPDATE SET
                    interests = excluded.interests,
                    duration_days = excluded.duration_days,
                    budget = excluded.budget,
                    updated_at = datetime('now')
                """,
                {
                    "sub": sub,
                    "interests": ",".join(interests or []),
                    "duration_days": duration_days,
                    "budget": budget,
                },
            )
        return self.get_preferences(sub)
