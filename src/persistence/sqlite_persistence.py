"""SQLite persistence implementation with sqlite-vec.

This module provides a SQLite-based persistence layer using sqlite-vec
for vector storage. It implements the MemoryRepository protocol for
fast, local access to agent working memory.
"""

import contextlib
import json
import sqlite3
from pathlib import Path
from typing import Any

from src.persistence.strategy import MemoryEntry, MemoryRepository


class SQLitePersistence(MemoryRepository):
    """SQLite persistence implementation with sqlite-vec.

    This class provides a high-performance, local memory store using
    SQLite with sqlite-vec for vector embeddings. It is designed for
    fast access to agent working memory during execution.
    """

    def __init__(self, db_path: str = "project_manager_v3.db"):
        """Initialize SQLite persistence.

        Args:
            db_path: Path to the SQLite database file.
        """
        self.db_path = str(Path(db_path).resolve())
        self._conn: sqlite3.Connection | None = None
        self._init_database()

    def _init_database(self) -> None:
        """Initialize the SQLite database and create tables."""
        self._conn = sqlite3.connect(self.db_path)
        self._conn.row_factory = sqlite3.Row
        cursor = self._conn.cursor()

        # Create memories table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL,
                text TEXT NOT NULL,
                metadata TEXT DEFAULT '{}',
                embedding BLOB,
                chunk_key TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES agent_sessions (id)
            )
        """)

        # Create index for session_id
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_memories_session_id
            ON memories (session_id)
        """)

        # Create index for chunk_key (deduplication)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_memories_chunk_key
            ON memories (chunk_key)
        """)

        # Initialize sqlite-vec extension
        with contextlib.suppress(sqlite3.OperationalError):
            cursor.execute("SELECT * FROM vec_meta")

        self._conn.commit()

    def _serialize_metadata(self, metadata: dict[str, Any]) -> str:
        """Serialize metadata to JSON string."""
        return json.dumps(metadata)

    def _deserialize_metadata(self, metadata_str: str) -> dict[str, Any]:
        """Deserialize metadata from JSON string."""
        try:
            result: dict[str, Any] = json.loads(metadata_str)
            return result
        except (json.JSONDecodeError, TypeError):
            return {}

    async def add_memory(
        self,
        session_id: int,
        text: str,
        metadata: dict[str, Any] | None = None,
        embedding: list[float] | None = None,
        chunk_key: str | None = None,
    ) -> int:
        """Add a memory entry.

        Args:
            session_id: The session ID this memory belongs to.
            text: The memory text content.
            metadata: Optional metadata dictionary.
            embedding: Optional vector embedding (can be None for later computation).
            chunk_key: Optional reconciliation key for deduplication.

        Returns:
            The ID of the newly created memory entry.
        """

        if self._conn is None:
            raise RuntimeError("Database connection not initialized")

        if metadata is None:
            metadata = {}

        conn = self._conn
        cursor = conn.cursor()

        # Check for duplicate chunk_key
        if chunk_key is not None:
            cursor.execute(
                "SELECT id FROM memories WHERE chunk_key = ?",
                (chunk_key,),
            )
            existing = cursor.fetchone()
            if existing is not None:
                return int(existing["id"])

        # Serialize metadata
        metadata_json = self._serialize_metadata(metadata)

        # Serialize embedding if provided
        embedding_blob: bytes | None = None
        if embedding is not None:
            embedding_blob = json.dumps(embedding).encode("utf-8")

        cursor.execute(
            """
            INSERT INTO memories (session_id, text, metadata, embedding, chunk_key)
            VALUES (?, ?, ?, ?, ?)
            """,
            (session_id, text, metadata_json, embedding_blob, chunk_key),
        )

        conn.commit()
        return int(cursor.lastrowid) if cursor.lastrowid is not None else 0

    async def get_memory(self, memory_id: int) -> MemoryEntry | None:
        """Get a memory entry by ID.

        Args:
            memory_id: The memory entry ID to retrieve.

        Returns:
            The memory entry if found, None otherwise.
        """
        if self._conn is None:
            raise RuntimeError("Database connection not initialized")

        conn = self._conn
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT id, session_id, text, metadata, embedding, chunk_key, created_at
            FROM memories
            WHERE id = ?
            """,
            (memory_id,),
        )

        row = cursor.fetchone()
        if row is None:
            return None

        embedding: list[float] | None = None
        if row["embedding"] is not None:
            with contextlib.suppress(json.JSONDecodeError, UnicodeDecodeError):
                embedding = json.loads(row["embedding"].decode("utf-8"))

        return MemoryEntry(
            id=row["id"],
            session_id=row["session_id"],
            text=row["text"],
            metadata=self._deserialize_metadata(row["metadata"]),
            embedding=embedding,
            chunk_key=row["chunk_key"],
            created_at=row["created_at"],
        )

    async def list_memories(
        self,
        session_id: int | None = None,
        limit: int = 100,
    ) -> list[MemoryEntry]:
        """List memory entries with optional filters.

        Args:
            session_id: Optional session ID filter.
            limit: Maximum number of memories to return.

        Returns:
            List of memory entries.
        """
        if self._conn is None:
            raise RuntimeError("Database connection not initialized")

        conn = self._conn
        cursor = conn.cursor()

        if session_id is not None:
            cursor.execute(
                """
                SELECT id, session_id, text, metadata, embedding, chunk_key, created_at
                FROM memories
                WHERE session_id = ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (session_id, limit),
            )
        else:
            cursor.execute(
                """
                SELECT id, session_id, text, metadata, embedding, chunk_key, created_at
                FROM memories
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (limit,),
            )

        rows = cursor.fetchall()
        memories: list[MemoryEntry] = []

        for row in rows:
            embedding: list[float] | None = None
            if row["embedding"] is not None:
                with contextlib.suppress(json.JSONDecodeError, UnicodeDecodeError):
                    embedding = json.loads(row["embedding"].decode("utf-8"))

            memories.append(
                MemoryEntry(
                    id=row["id"],
                    session_id=row["session_id"],
                    text=row["text"],
                    metadata=self._deserialize_metadata(row["metadata"]),
                    embedding=embedding,
                    chunk_key=row["chunk_key"],
                    created_at=row["created_at"],
                )
            )

        return memories

    async def remove_memory(self, memory_id: int) -> bool:
        """Remove a memory entry.

        Args:
            memory_id: The memory entry ID to remove.

        Returns:
            True if the entry was removed, False if not found.
        """
        if self._conn is None:
            raise RuntimeError("Database connection not initialized")

        conn = self._conn
        cursor = conn.cursor()

        cursor.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
        conn.commit()

        return cursor.rowcount > 0

    def close(self) -> None:
        """Close the database connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None
