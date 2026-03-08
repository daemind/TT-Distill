"""Hybrid persistence implementation combining Postgres and SQLite.

This module provides a hybrid persistence layer that combines Postgres
(with pgvector) for global state and SQLite (with sqlite-vec) for
local agent memory, with Event-Driven synchronization between layers.
"""

from typing import Any

from src.persistence.postgres_persistence import PostgresPersistence
from src.persistence.sqlite_persistence import SQLitePersistence
from src.persistence.strategy import (
    MemoryRepository,
    PersistenceStrategy,
    SessionRepository,
    VectorSearchRepository,
)
from src.persistence.sync_pipeline import VectorSyncPipeline


class HybridPersistence(PersistenceStrategy):
    """Hybrid persistence combining Postgres and SQLite.

    This class provides a unified persistence interface that combines:
    - Postgres with pgvector for global state and vector search
    - SQLite with sqlite-vec for local agent memory
    - Event-Driven synchronization between layers

    The architecture follows the separation of concerns principle:
    - Orchestration strategy is decoupled from memory storage
    - Local memory provides fast access during agent execution
    - Global state provides persistence and cross-session search
    """

    def __init__(
        self,
        postgres_url: str = "postgres://localhost/project_manager",
        sqlite_path: str = "project_manager_v3.db",
    ):
        """Initialize hybrid persistence.

        Args:
            postgres_url: Postgres database URL.
            sqlite_path: Path to the SQLite database for local memory.
        """
        self.postgres = PostgresPersistence(postgres_url, sqlite_path)
        self.sqlite = SQLitePersistence(sqlite_path)
        self.sync_pipeline = VectorSyncPipeline()
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the hybrid persistence layer."""
        await self.postgres.initialize()
        await self.sync_pipeline.start()
        self._initialized = True

    async def close(self) -> None:
        """Close the hybrid persistence layer."""
        await self.sync_pipeline.stop()
        await self.postgres.close()
        self._initialized = False

    @property
    def session_repo(self) -> SessionRepository:
        """Get the session repository (Postgres)."""
        return self.postgres.session_repo

    @property
    def memory_repo(self) -> MemoryRepository:
        """Get the memory repository (SQLite)."""
        return self.sqlite

    @property
    def vector_repo(self) -> VectorSearchRepository:
        """Get the vector search repository (Postgres)."""
        return self.postgres.vector_repo

    async def add_memory_with_sync(
        self,
        session_id: int,
        text: str,
        metadata: dict[str, Any] | None = None,
        embedding: list[float] | None = None,
        chunk_key: str | None = None,
    ) -> int:
        """Add a memory entry with automatic synchronization.

        This method adds the memory to SQLite (local) and schedules
        it for synchronization to Postgres (global) via the Event-Driven
        sync pipeline.

        Args:
            session_id: The session ID this memory belongs to.
            text: The memory text content.
            metadata: Optional metadata dictionary.
            embedding: Optional vector embedding (computed at source).
            chunk_key: Optional reconciliation key for deduplication.

        Returns:
            The ID of the newly created memory entry.
        """
        # Add to SQLite first (fast local operation)
        memory_id = await self.sqlite.add_memory(
            session_id=session_id,
            text=text,
            metadata=metadata,
            embedding=embedding,
            chunk_key=chunk_key,
        )

        # Schedule synchronization to Postgres
        if embedding is not None:
            await self.sync_pipeline.queue_sync(
                session_id=session_id,
                text=text,
                metadata=metadata or {},
                embedding=embedding,
                chunk_key=chunk_key,
            )

        return memory_id
