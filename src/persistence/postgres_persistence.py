"""Postgres persistence implementation with pgvector.

This module provides a Postgres-based persistence layer using pgvector
for vector storage. It implements the SessionRepository and
VectorSearchRepository protocols for global state management and
global vector search across all agents.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from src.persistence.sqlite_persistence import SQLitePersistence
from src.persistence.strategy import (
    MemoryRepository,
    PersistenceStrategy,
    SessionCompleteArgs,
    SessionCreateArgs,
    SessionData,
    SessionRepository,
    VectorSearchRepository,
    VectorSearchResult,
)

try:
    import asyncpg
    ASYNCPG_AVAILABLE = True
except ImportError:
    ASYNCPG_AVAILABLE = False
    asyncpg = None


class PostgresPersistence(PersistenceStrategy):
    """Postgres persistence implementation with pgvector.

    This class provides a global state store using Postgres with pgvector
    for vector embeddings. It implements all repository protocols and
    delegates local memory operations to a SQLitePersistence instance.
    """

    def __init__(
        self,
        db_url: str = "postgres://localhost/project_manager",
        sqlite_path: str = "project_manager_v3.db",
    ) -> None:
        """Initialize Postgres persistence.

        Args:
            db_url: Postgres database URL.
            sqlite_path: Path to the SQLite database for local memory.
        """
        self.db_url = db_url
        self.sqlite_path = sqlite_path
        self._pool: asyncpg.Pool | None = None
        self._sqlite: SQLitePersistence | None = None

    async def initialize(self) -> None:
        """Initialize the Postgres persistence layer."""
        if not ASYNCPG_AVAILABLE or asyncpg is None:
            raise RuntimeError(
                "asyncpg is not installed. "
                "Install with: pip install asyncpg"
            )

        # Create connection pool
        self._pool = await asyncpg.create_pool(self.db_url)

        # Initialize SQLite for local memory
        self._sqlite = SQLitePersistence(self.sqlite_path)

        # Create tables and indexes
        async with self._pool.acquire() as conn:
            # Create projects table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS projects (
                    id SERIAL PRIMARY KEY,
                    name TEXT NOT NULL,
                    path TEXT NOT NULL,
                    status TEXT DEFAULT 'pending',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Create agent_sessions table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS agent_sessions (
                    id SERIAL PRIMARY KEY,
                    project_id INTEGER REFERENCES projects(id),
                    agent_name TEXT NOT NULL,
                    role TEXT NOT NULL,
                    task TEXT,
                    prompt TEXT,
                    result TEXT,
                    reasoning_content TEXT,
                    success BOOLEAN,
                    tokens_used INTEGER,
                    parent_session_id INTEGER REFERENCES agent_sessions(id),
                    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    completed_at TIMESTAMP
                )
            """)

            # Create task_events table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS task_events (
                    id SERIAL PRIMARY KEY,
                    project_id INTEGER REFERENCES projects(id),
                    task_id TEXT,
                    event_type TEXT,
                    description TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Create project_metrics table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS project_metrics (
                    id SERIAL PRIMARY KEY,
                    project_id INTEGER REFERENCES projects(id),
                    metric_key TEXT,
                    metric_value REAL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Create model_registry table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS model_registry (
                    id SERIAL PRIMARY KEY,
                    model_name TEXT NOT NULL,
                    model_id TEXT NOT NULL UNIQUE,
                    task_type TEXT NOT NULL,
                    input_format TEXT,
                    estimated_cost_kw REAL NOT NULL,
                    description TEXT,
                    version TEXT DEFAULT '1.0.0',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Create blueprints table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS blueprints (
                    blueprint_id TEXT PRIMARY KEY,
                    project_id INTEGER REFERENCES projects(id),
                    version INTEGER DEFAULT 1,
                    title TEXT NOT NULL,
                    description TEXT NOT NULL,
                    features TEXT,
                    validation_status TEXT DEFAULT 'pending',
                    validated_by TEXT,
                    validated_at TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Create validation_checkpoints table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS validation_checkpoints (
                    checkpoint_id TEXT PRIMARY KEY,
                    blueprint_id TEXT REFERENCES blueprints(blueprint_id),
                    checkpoint_name TEXT NOT NULL,
                    checkpoint_type TEXT NOT NULL,
                    status TEXT DEFAULT 'pending',
                    user_feedback TEXT,
                    approved_by TEXT,
                    approved_at TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Create vectors table with pgvector
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS vectors (
                    id SERIAL PRIMARY KEY,
                    session_id INTEGER REFERENCES agent_sessions(id),
                    text TEXT NOT NULL,
                    metadata JSONB NOT NULL DEFAULT '{}',
                    embedding vector(2880),
                    chunk_key TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Create index for session_id
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_vectors_session_id
                ON vectors (session_id)
            """)

            # Create index for chunk_key (deduplication)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_vectors_chunk_key
                ON vectors (chunk_key)
            """)

            # Create HNSW index for vector similarity search
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_vectors_embedding
                ON vectors USING hnsw (embedding vector_cosine_ops)
            """)

    async def close(self) -> None:
        """Close the Postgres persistence layer."""
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

        if self._sqlite is not None:
            self._sqlite.close()
            self._sqlite = None

    @property
    def session_repo(self) -> SessionRepository:
        """Get the session repository (wrapper for async methods)."""
        return _SyncSessionRepository(self)  # type: ignore[return-value]

    @property
    def memory_repo(self) -> MemoryRepository:
        """Get the memory repository (delegates to SQLite)."""
        if self._sqlite is None:
            raise RuntimeError("Persistence not initialized")
        return self._sqlite

    @property
    def vector_repo(self) -> VectorSearchRepository:
        """Get the vector search repository (wrapper for async methods)."""
        return _SyncVectorSearchRepository(self)

    async def create_session(self, args: SessionCreateArgs) -> int:
        """Create a new agent session.

        Args:
            args: Session creation arguments.

        Returns:
            The ID of the newly created session.
        """
        if self._pool is None:
            raise RuntimeError("Persistence not initialized")

        async with self._pool.acquire() as conn:
            result = await conn.fetchrow(
                """
                INSERT INTO agent_sessions (
                    project_id, agent_name, role, task, prompt, parent_session_id
                )
                VALUES ($1, $2, $3, $4, $5, $6)
                RETURNING id
                """,
                args.project_id,
                args.agent_name,
                args.role,
                args.task,
                args.prompt,
                args.parent_session_id,
            )
            return int(result["id"])

    async def complete_session(self, args: SessionCompleteArgs) -> None:
        """Complete an agent session with results.

        Args:
            args: Session completion arguments.
        """
        if self._pool is None:
            raise RuntimeError("Persistence not initialized")

        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE agent_sessions
                SET result = $1, reasoning_content = $2, success = $3,
                    tokens_used = $4, completed_at = CURRENT_TIMESTAMP
                WHERE id = $5
                """,
                args.result,
                args.reasoning,
                args.success,
                args.tokens_used,
                args.session_id,
            )

    async def get_session(self, session_id: int) -> SessionData | None:
        """Get a session by ID.

        Args:
            session_id: The session ID to retrieve.

        Returns:
            The session data if found, None otherwise.
        """
        if self._pool is None:
            raise RuntimeError("Persistence not initialized")

        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT id, project_id, agent_name, role, task, prompt, result,
                       reasoning_content, success, tokens_used, parent_session_id,
                       started_at, completed_at
                FROM agent_sessions
                WHERE id = $1
                """,
                session_id,
            )

            if row is None:
                return None

            return SessionData(
                id=int(row["id"]),
                project_id=int(row["project_id"]),
                agent_name=row["agent_name"],
                role=row["role"],
                task=row["task"],
                prompt=row["prompt"],
                result=row["result"],
                reasoning_content=row["reasoning_content"],
                success=row["success"],
                tokens_used=int(row["tokens_used"]) if row["tokens_used"] else None,
                parent_session_id=int(row["parent_session_id"]) if row["parent_session_id"] else None,
                started_at=row["started_at"].isoformat() if row["started_at"] else "",
                completed_at=row["completed_at"].isoformat() if row["completed_at"] else None,
            )

    async def list_sessions(
        self,
        project_id: int | None = None,
        agent_name: str | None = None,
        limit: int = 100,
    ) -> list[SessionData]:
        """List sessions with optional filters.

        Args:
            project_id: Optional project ID filter.
            agent_name: Optional agent name filter.
            limit: Maximum number of sessions to return.

        Returns:
            List of session data.
        """
        if self._pool is None:
            raise RuntimeError("Persistence not initialized")

        async with self._pool.acquire() as conn:
            if project_id is not None and agent_name is not None:
                rows = await conn.fetch(
                    """
                    SELECT id, project_id, agent_name, role, task, prompt, result,
                           reasoning_content, success, tokens_used, parent_session_id,
                           started_at, completed_at
                    FROM agent_sessions
                    WHERE project_id = $1 AND agent_name = $2
                    ORDER BY started_at DESC
                    LIMIT $3
                    """,
                    project_id,
                    agent_name,
                    limit,
                )
            elif project_id is not None:
                rows = await conn.fetch(
                    """
                    SELECT id, project_id, agent_name, role, task, prompt, result,
                           reasoning_content, success, tokens_used, parent_session_id,
                           started_at, completed_at
                    FROM agent_sessions
                    WHERE project_id = $1
                    ORDER BY started_at DESC
                    LIMIT $2
                    """,
                    project_id,
                    limit,
                )
            elif agent_name is not None:
                rows = await conn.fetch(
                    """
                    SELECT id, project_id, agent_name, role, task, prompt, result,
                           reasoning_content, success, tokens_used, parent_session_id,
                           started_at, completed_at
                    FROM agent_sessions
                    WHERE agent_name = $1
                    ORDER BY started_at DESC
                    LIMIT $2
                    """,
                    agent_name,
                    limit,
                )
            else:
                rows = await conn.fetch(
                    """
                    SELECT id, project_id, agent_name, role, task, prompt, result,
                           reasoning_content, success, tokens_used, parent_session_id,
                           started_at, completed_at
                    FROM agent_sessions
                    ORDER BY started_at DESC
                    LIMIT $1
                    """,
                    limit,
                )

            return [
                SessionData(
                    id=int(row["id"]),
                    project_id=int(row["project_id"]),
                    agent_name=row["agent_name"],
                    role=row["role"],
                    task=row["task"],
                    prompt=row["prompt"],
                    result=row["result"],
                    reasoning_content=row["reasoning_content"],
                    success=row["success"],
                    tokens_used=int(row["tokens_used"]) if row["tokens_used"] else None,
                    parent_session_id=int(row["parent_session_id"]) if row["parent_session_id"] else None,
                    started_at=row["started_at"].isoformat() if row["started_at"] else "",
                    completed_at=row["completed_at"].isoformat() if row["completed_at"] else None,
                )
                for row in rows
            ]

    async def search(
        self,
        query_embedding: list[float],
        session_id: int | None = None,
        top_k: int = 10,
        min_similarity: float = 0.5,
    ) -> list[VectorSearchResult]:
        """Perform vector similarity search.

        Args:
            query_embedding: The query vector embedding.
            session_id: Optional session ID filter.
            top_k: Maximum number of results to return.
            min_similarity: Minimum similarity threshold.

        Returns:
            List of search results sorted by similarity.
        """
        if self._pool is None:
            raise RuntimeError("Persistence not initialized")

        async with self._pool.acquire() as conn:
            if session_id is not None:
                rows = await conn.fetch(
                    """
                    SELECT id, session_id, text, metadata, embedding, chunk_key,
                           created_at,
                           1 - (embedding <=> $1) AS similarity
                    FROM vectors
                    WHERE session_id = $2
                    ORDER BY embedding <=> $1
                    LIMIT $3
                    """,
                    query_embedding,
                    session_id,
                    top_k,
                )
            else:
                rows = await conn.fetch(
                    """
                    SELECT id, session_id, text, metadata, embedding, chunk_key,
                           created_at,
                           1 - (embedding <=> $1) AS similarity
                    FROM vectors
                    ORDER BY embedding <=> $1
                    LIMIT $2
                    """,
                    query_embedding,
                    top_k,
                )

            results: list[VectorSearchResult] = []
            for row in rows:
                similarity = float(row["similarity"])
                if similarity >= min_similarity:
                    embedding: list[float] | None = None
                    if row["embedding"] is not None:
                        embedding = list(row["embedding"])

                    results.append(
                        VectorSearchResult(
                            id=int(row["id"]),
                            session_id=int(row["session_id"]),
                            text=row["text"],
                            metadata=json.loads(row["metadata"]) if row["metadata"] else {},
                            similarity=similarity,
                            embedding=embedding,
                        )
                    )

            return results

    async def search_by_text(
        self, query: str, limit: int = 10, session_id: int | None = None
    ) -> list[VectorSearchResult]:
        """Perform vector similarity search by text query.

        Args:
            query: The text query to search for.
            limit: Maximum number of results to return.
            session_id: Optional session ID filter.

        Returns:
            List of search results sorted by similarity.
        """
        if self._pool is None:
            raise RuntimeError("Persistence not initialized")

        # For text-based search, we need to embed the query first
        # For now, we return all matching entries for the session (or all if session_id is None)
        async with self._pool.acquire() as conn:
            if session_id is not None:
                rows = await conn.fetch(
                    """
                    SELECT id, session_id, text, metadata, embedding, chunk_key,
                           created_at
                    FROM vectors
                    WHERE session_id = $1
                    ORDER BY created_at DESC
                    LIMIT $2
                    """,
                    session_id,
                    limit,
                )
            else:
                rows = await conn.fetch(
                    """
                    SELECT id, session_id, text, metadata, embedding, chunk_key,
                           created_at
                    FROM vectors
                    ORDER BY created_at DESC
                    LIMIT $1
                    """,
                    limit,
                )

            return [
                VectorSearchResult(
                    id=int(row["id"]),
                    session_id=int(row["session_id"]),
                    text=row["text"],
                    metadata=json.loads(row["metadata"]) if row["metadata"] else {},
                    similarity=1.0,
                    embedding=None,
                )
                for row in rows
            ]

    async def add_vector(
        self,
        session_id: int,
        text: str,
        metadata: dict[str, Any],
        embedding: list[float],
        chunk_key: str | None = None,
    ) -> int:
        """Add a vector entry for global search.

        Args:
            session_id: The session ID this vector belongs to.
            text: The vector text content.
            metadata: Metadata dictionary.
            embedding: The vector embedding.
            chunk_key: Optional reconciliation key for deduplication.

        Returns:
            The ID of the newly created vector entry.
        """
        if self._pool is None:
            raise RuntimeError("Persistence not initialized")

        async with self._pool.acquire() as conn:
            # Check for duplicate chunk_key
            if chunk_key is not None:
                existing = await conn.fetchval(
                    "SELECT id FROM vectors WHERE chunk_key = $1",
                    chunk_key,
                )
                if existing is not None:
                    return int(existing)

            result = await conn.fetchrow(
                """
                INSERT INTO vectors (session_id, text, metadata, embedding, chunk_key)
                VALUES ($1, $2, $3, $4, $5)
                RETURNING id
                """,
                session_id,
                text,
                json.dumps(metadata),
                embedding,
                chunk_key,
            )
            return int(result["id"])

    async def remove_vector(self, vector_id: int) -> bool:
        """Remove a vector entry.

        Args:
            vector_id: The vector entry ID to remove.

        Returns:
            True if the entry was removed, False if not found.
        """
        if self._pool is None:
            raise RuntimeError("Persistence not initialized")

        async with self._pool.acquire() as conn:
            result = await conn.execute(
                "DELETE FROM vectors WHERE id = $1",
                vector_id,
            )
            return int(result.split()[1]) == 1


class _SyncSessionRepository:
    """Wrapper to provide synchronous interface for async SessionRepository methods."""

    def __init__(self, postgres: PostgresPersistence) -> None:
        self._postgres = postgres

    def create_session(self, args: SessionCreateArgs) -> int:
        """Create a new agent session (synchronous wrapper)."""
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self._postgres.create_session(args))

    def complete_session(self, args: SessionCompleteArgs) -> None:
        """Complete an agent session (synchronous wrapper)."""
        loop = asyncio.get_event_loop()
        loop.run_until_complete(self._postgres.complete_session(args))

    def get_session(self, session_id: int) -> SessionData | None:
        """Get a session by ID (synchronous wrapper)."""
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self._postgres.get_session(session_id))

    def list_sessions(
        self,
        project_id: int | None = None,
        agent_name: str | None = None,
        limit: int = 100,
    ) -> list[SessionData]:
        """List sessions (synchronous wrapper)."""
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(
            self._postgres.list_sessions(project_id, agent_name, limit)
        )


class _SyncVectorSearchRepository:
    """Wrapper to provide synchronous interface for async VectorSearchRepository methods."""

    def __init__(self, postgres: PostgresPersistence) -> None:
        self._postgres = postgres

    def search(
        self,
        query_embedding: list[float],
        session_id: int | None = None,
        top_k: int = 10,
        min_similarity: float = 0.5,
    ) -> list[VectorSearchResult]:
        """Perform vector search (synchronous wrapper)."""
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(
            self._postgres.search(query_embedding, session_id, top_k, min_similarity)
        )

    def search_by_text(
        self, query: str, limit: int = 10, session_id: int | None = None
    ) -> list[VectorSearchResult]:
        """Perform vector search by text query (synchronous wrapper)."""
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(
            self._postgres.search_by_text(query, limit, session_id)
        )

    def add_vector(
        self,
        session_id: int,
        text: str,
        metadata: dict[str, Any],
        embedding: list[float],
        chunk_key: str | None = None,
    ) -> int:
        """Add a vector entry (synchronous wrapper)."""
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(
            self._postgres.add_vector(session_id, text, metadata, embedding, chunk_key)
        )

    def remove_vector(self, vector_id: int) -> bool:
        """Remove a vector entry (synchronous wrapper)."""
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self._postgres.remove_vector(vector_id))
