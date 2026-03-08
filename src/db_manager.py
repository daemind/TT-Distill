import json
import os
import sqlite3
import time
from pathlib import Path
from typing import Any, cast

from src.logger import get_logger
from src.persistence.strategy import (
    MemoryEntry,
    MemoryRepository,
    PersistenceStrategy,
    SessionCompleteArgs,
    SessionCreateArgs,
    SessionRepository,
    VectorSearchRepository,
    VectorSearchResult,
)


class DBManager(PersistenceStrategy):
    """
    Manages the SQLite database for Project Manager v3.
    Follows SOC by handling only data persistence.

    Implements the PersistenceStrategy interface to provide backward compatibility
    while enabling the hybrid persistence architecture (Postgres + SQLite/svec).
    """

    def __init__(self, db_path: str = "project_manager_v3.db") -> None:
        if db_path == ":memory:":
            self.db_path = ":memory:"
        else:
            self.db_path = str(Path(db_path).resolve())
        self._conn: sqlite3.Connection | None = (
            None  # Persistent connection for :memory: mode
        )
        self._initialize_db()

    # SessionRepository implementation
    async def create_session(self, args: SessionCreateArgs) -> int:
        """Starts a new agent session."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO agent_sessions (project_id, agent_name, role, task, parent_session_id) VALUES (?, ?, ?, ?, ?)",
                (
                    args.project_id,
                    args.agent_name,
                    args.role,
                    args.task,
                    args.parent_session_id,
                ),
            )
            return cast(int, cursor.lastrowid)

    async def complete_session(self, args: SessionCompleteArgs) -> None:
        """Finalizes an agent session."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """UPDATE agent_sessions
                   SET result = ?, reasoning_content = ?, success = ?, completed_at = CURRENT_TIMESTAMP
                   WHERE id = ?""",
                (args.result, args.reasoning, args.success, args.session_id),
            )

    async def get_session(self, session_id: int) -> dict[str, Any] | None:
        """Retrieves session details."""
        with self._get_connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM agent_sessions WHERE id = ?", (session_id,))
            row = cursor.fetchone()
            return dict(row) if row else None

    async def list_sessions(
        self,
        project_id: int | None = None,
        agent_name: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Retrieves agent sessions, optionally filtered."""
        with self._get_connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            query = "SELECT * FROM agent_sessions"
            params: list[Any] = []
            conditions: list[str] = []

            if project_id is not None:
                conditions.append("project_id = ?")
                params.append(project_id)
            if agent_name is not None:
                conditions.append("agent_name = ?")
                params.append(agent_name)

            if conditions:
                query += " WHERE " + " AND ".join(conditions)

            query += " ORDER BY started_at DESC LIMIT ?"
            params.append(limit)

            cursor.execute(query, tuple(params))
            return [dict(row) for row in cursor.fetchall()]

    async def list_sessions_by_project(self, project_id: int) -> list[dict[str, Any]]:
        """Retrieves agent sessions for a specific project."""
        with self._get_connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM agent_sessions WHERE project_id = ? ORDER BY started_at DESC",
                (project_id,),
            )
            return [dict(row) for row in cursor.fetchall()]

    # MemoryRepository implementation
    async def add_memory(
        self, session_id: int, text: str, metadata: dict[str, Any] | None = None, chunk_key: str | None = None
    ) -> int:
        """Adds a memory entry to the database."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """INSERT INTO memories (session_id, text, metadata, chunk_key)
                   VALUES (?, ?, ?, ?)""",
                (session_id, text, json.dumps(metadata) if metadata else None, chunk_key),
            )
            return cast(int, cursor.lastrowid)

    async def get_memory(self, memory_id: int) -> MemoryEntry | None:
        """Retrieves a memory entry by ID."""
        with self._get_connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM memories WHERE id = ?", (memory_id,))
            row = cursor.fetchone()
            if row:
                metadata = {}
                if row["metadata"]:
                    try:
                        metadata = json.loads(row["metadata"])
                    except json.JSONDecodeError:
                        metadata = {}
                return MemoryEntry(
                    id=row["id"],
                    session_id=row["session_id"],
                    text=row["text"],
                    metadata=metadata,
                    chunk_key=row["chunk_key"],
                    embedding=None,
                    created_at=row["created_at"],
                )
            return None

    async def list_memories(
        self, session_id: int | None = None
    ) -> list[MemoryEntry]:
        """Retrieves memory entries, optionally filtered by session_id."""
        with self._get_connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            if session_id:
                cursor.execute(
                    "SELECT * FROM memories WHERE session_id = ? ORDER BY created_at DESC",
                    (session_id,),
                )
            else:
                cursor.execute("SELECT * FROM memories ORDER BY created_at DESC")
            rows = cursor.fetchall()
            memories: list[MemoryEntry] = []
            for row in rows:
                metadata = {}
                if row["metadata"]:
                    try:
                        metadata = json.loads(row["metadata"])
                    except json.JSONDecodeError:
                        metadata = {}
                memories.append(
                    MemoryEntry(
                        id=row["id"],
                        session_id=row["session_id"],
                        text=row["text"],
                        metadata=metadata,
                        chunk_key=row["chunk_key"],
                        embedding=None,
                        created_at=row["created_at"],
                    )
                )
            return memories

    async def remove_memory(self, memory_id: int) -> bool:
        """Removes a memory entry by ID."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
            return cursor.rowcount > 0

    # VectorSearchRepository implementation
    async def search(
        self,
        query_embedding: list[float],
        session_id: int | None = None,
        top_k: int = 10,
        min_similarity: float = 0.5,
    ) -> list[VectorSearchResult]:
        """Performs a similarity search on memory entries."""
        # SQLite doesn't support vector similarity natively, so we return all
        # matching entries for the session (or all if session_id is None)
        with self._get_connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            if session_id:
                cursor.execute(
                    "SELECT id, session_id, text, metadata, chunk_key FROM memories WHERE session_id = ? ORDER BY created_at DESC LIMIT ?",
                    (session_id, top_k),
                )
            else:
                cursor.execute(
                    "SELECT id, session_id, text, metadata, chunk_key FROM memories ORDER BY created_at DESC LIMIT ?",
                    (top_k,),
                )

            rows = cursor.fetchall()
            results: list[VectorSearchResult] = []
            for row in rows:
                meta = {}
                if row["metadata"]:
                    try:
                        meta = json.loads(row["metadata"])
                    except json.JSONDecodeError:
                        meta = {}
                results.append(
                    VectorSearchResult(
                        id=row["id"],
                        session_id=row["session_id"],
                        text=row["text"],
                        metadata=meta,
                        similarity=1.0,
                        embedding=None,
                    )
                )
            return results

    async def search_by_text(
        self, query: str, limit: int = 10, session_id: int | None = None
    ) -> list[VectorSearchResult]:
        """Performs a similarity search on memory entries by text query."""
        # SQLite doesn't support vector similarity natively, so we return all
        # matching entries for the session (or all if session_id is None)
        with self._get_connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            if session_id:
                cursor.execute(
                    "SELECT id, session_id, text, metadata, chunk_key FROM memories WHERE session_id = ? ORDER BY created_at DESC LIMIT ?",
                    (session_id, limit),
                )
            else:
                cursor.execute(
                    "SELECT id, session_id, text, metadata, chunk_key FROM memories ORDER BY created_at DESC LIMIT ?",
                    (limit,),
                )

            rows = cursor.fetchall()
            results: list[VectorSearchResult] = []
            for row in rows:
                meta = {}
                if row["metadata"]:
                    try:
                        meta = json.loads(row["metadata"])
                    except json.JSONDecodeError:
                        meta = {}
                results.append(
                    VectorSearchResult(
                        id=row["id"],
                        session_id=row["session_id"],
                        text=row["text"],
                        metadata=meta,
                        similarity=1.0,
                        embedding=None,
                    )
                )
            return results

    async def add_vector(self, entry: MemoryEntry) -> None:
        """Adds a vector embedding to the database."""
        # For now, we store the embedding as a BLOB in the memories table
        # The embedding should be computed by the embedding service before calling this
        # Note: This requires the embeddings column to be added to the schema
        # For backward compatibility, we skip this in pure SQLite mode
        pass

    async def remove_vector(self, chunk_key: str) -> bool:
        """Removes a vector embedding by chunk_key."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM memories WHERE chunk_key = ?", (chunk_key,))
            return cursor.rowcount > 0

    # Legacy methods for backward compatibility
    def _get_connection(self) -> sqlite3.Connection:
        """Creates a new database connection with concurrency handling."""
        logger = get_logger(__name__)

        # Ensure parent directory exists and use absolute path
        if self.db_path == ":memory:":
            abs_db_path = Path(":memory:")
            parent = None
        else:
            abs_db_path = Path(self.db_path).resolve()
            parent = abs_db_path.parent
            parent.mkdir(parents=True, exist_ok=True)

        max_retries = 3
        base_delay = 0.1
        for attempt in range(max_retries):
            try:
                # Check directory permissions first
                if parent and not os.access(parent, os.W_OK):
                    logger.error(
                        f"Permission denied: Directory {parent} is not writable."
                    )
                    raise PermissionError(f"Directory {parent} is not writable.")

                if self._conn:
                    return self._conn

                conn = sqlite3.connect(self.db_path, timeout=30.0)
                # Use simple journal mode for maximum compatibility
                try:
                    conn.execute("PRAGMA journal_mode = DELETE;")
                except sqlite3.OperationalError as e:
                    logger.warning(f"Could not set journal mode: {e}")

                conn.execute("PRAGMA synchronous = NORMAL;")
                logger.debug(f"Database connection established to {self.db_path}")

                if self.db_path == ":memory:":
                    self._conn = conn
                return conn
            except (sqlite3.OperationalError, PermissionError) as e:
                logger.info(f"Caught database error: {e}")
                error_str = str(e).lower()
                if (
                    "disk i/o" in error_str
                    or "unable to open" in error_str
                    or "permission" in error_str
                ):
                    if attempt == max_retries - 1:
                        logger.error(
                            f"Failed to connect after {max_retries} retries: {e}"
                        )
                        break
                    delay = base_delay * (2**attempt)
                    logger.warning(
                        f"Retriable error (attempt {attempt + 1}/{max_retries}), retrying in {delay:.2f}s: {e}"
                    )
                    time.sleep(delay)
                else:
                    logger.warning(f"Non-retryable database error: {e}")
                    raise
        raise RuntimeError("Failed to establish database connection after retries")

    def _initialize_db(self) -> None:
        """Creates the schema if it doesn't exist."""
        try:
            with self._get_connection() as conn:
                self._create_schema(conn)
        except Exception as e:
            get_logger(__name__).error(
                f"Failed to initialize disk DB at {self.db_path}: {e}"
            )
            get_logger(__name__).warning(
                "Falling back to IN-MEMORY database for this session."
            )
            self.db_path = ":memory:"
            with self._get_connection() as conn:
                self._create_schema(conn)

    def _create_schema(self, conn: sqlite3.Connection) -> None:
        """Internal helper to create schema on a connection."""
        cursor = conn.cursor()

        # Projects Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS projects (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                path TEXT NOT NULL,
                status TEXT DEFAULT 'pending',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Agent Sessions Table (Every LLM call)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS agent_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id INTEGER,
                agent_name TEXT NOT NULL,
                role TEXT NOT NULL,
                task TEXT,
                prompt TEXT,
                result TEXT,
                reasoning_content TEXT,
                success BOOLEAN,
                tokens_used INTEGER,
                parent_session_id INTEGER,
                started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                completed_at TIMESTAMP,
                FOREIGN KEY (project_id) REFERENCES projects (id)
            )
        """)

        # Migration: Add parent_session_id if missing from older DBs
        cursor.execute("PRAGMA table_info(agent_sessions)")
        columns = [col[1] for col in cursor.fetchall()]
        if "parent_session_id" not in columns:
            cursor.execute(
                "ALTER TABLE agent_sessions ADD COLUMN parent_session_id INTEGER"
            )

        # Task Events (Audit Trail)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS task_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id INTEGER,
                task_id TEXT,
                event_type TEXT,
                description TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (project_id) REFERENCES projects (id)
            )
        """)

        # Project Metrics
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS project_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id INTEGER,
                metric_key TEXT,
                metric_value REAL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (project_id) REFERENCES projects (id)
            )
        """)

        # Model Registry Table (for Hugging Face models)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS model_registry (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
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

        # Blueprints Table (for Progressive Blueprinting System)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS blueprints (
                blueprint_id TEXT PRIMARY KEY,
                project_id INTEGER NOT NULL,
                version INTEGER DEFAULT 1,
                title TEXT NOT NULL,
                description TEXT NOT NULL,
                features TEXT,
                validation_status TEXT DEFAULT 'pending',
                validated_by TEXT,
                validated_at TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (project_id) REFERENCES projects (id)
            )
        """)

        # Validation Checkpoints Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS validation_checkpoints (
                checkpoint_id TEXT PRIMARY KEY,
                blueprint_id TEXT NOT NULL,
                checkpoint_name TEXT NOT NULL,
                checkpoint_type TEXT NOT NULL,
                status TEXT DEFAULT 'pending',
                user_feedback TEXT,
                approved_by TEXT,
                approved_at TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (blueprint_id) REFERENCES blueprints (blueprint_id)
            )
        """)

        # Memories Table (for agent output indexing)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL,
                text TEXT NOT NULL,
                metadata TEXT,
                chunk_key TEXT UNIQUE NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES agent_sessions (id)
            )
        """)

        conn.commit()

    def create_project(self, name: str, path: str) -> int:
        """Registers a new project."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO projects (name, path) VALUES (?, ?)", (name, path)
            )
            return cast(int, cursor.lastrowid)

    def get_project_by_path(self, path: str) -> dict[str, Any] | None:
        """Retrieves project details by its filesystem path."""
        with self._get_connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM projects WHERE path = ?", (path,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def update_project_status(self, project_id: int, status: str) -> bool:
        """Updates project status."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE projects SET status = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (status, project_id),
            )
            return cursor.rowcount > 0

    def get_projects(self) -> list[dict[str, Any]]:
        """Retrieves all registered projects."""
        with self._get_connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM projects")
            return [dict(row) for row in cursor.fetchall()]

    def get_project(self, project_id: int) -> dict[str, Any] | None:
        """Retrieves a single project by ID."""
        with self._get_connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM projects WHERE id = ?", (project_id,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def update_project(self, project_id: int, name: str, path: str) -> bool:
        """Updates project details."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE projects SET name = ?, path = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (name, path, project_id),
            )
            return cursor.rowcount > 0

    def delete_project(self, project_id: int) -> bool:
        """Deletes a project and its associated sessions/events."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            # Failsafe: Associated sessions/events are deleted via CASCADE in a real DB,
            # but here we'll do it explicitly or ensure the schema supports it.
            # Our current schema doesn't have ON DELETE CASCADE for all tables.
            cursor.execute(
                "DELETE FROM agent_sessions WHERE project_id = ?", (project_id,)
            )
            cursor.execute(
                "DELETE FROM task_events WHERE project_id = ?", (project_id,)
            )
            cursor.execute(
                "DELETE FROM project_metrics WHERE project_id = ?", (project_id,)
            )
            cursor.execute("DELETE FROM projects WHERE id = ?", (project_id,))
            return cursor.rowcount > 0

    # Legacy methods for backward compatibility
    def create_session_sync(
        self,
        project_id: int,
        agent_name: str,
        role: str,
        task: str,
        parent_session_id: int | None = None,
    ) -> int:
        """Starts a new agent session (legacy sync method)."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO agent_sessions (project_id, agent_name, role, task, parent_session_id) VALUES (?, ?, ?, ?, ?)",
                (project_id, agent_name, role, task, parent_session_id),
            )
            return cast(int, cursor.lastrowid)

    def get_session_sync(self, session_id: int) -> dict[str, Any] | None:
        """Retrieves session details (legacy sync method)."""
        with self._get_connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM agent_sessions WHERE id = ?", (session_id,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def complete_session_sync(
        self,
        session_id: int,
        result: str,
        success: bool,
        tokens: int = 0,
        reasoning: str | None = None,
    ) -> None:
        """Finalizes an agent session (legacy sync method)."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """UPDATE agent_sessions
                   SET result = ?, reasoning_content = ?, success = ?, tokens_used = ?, completed_at = CURRENT_TIMESTAMP
                   WHERE id = ?""",
                (result, reasoning, success, tokens, session_id),
            )

    def list_sessions_sync(self) -> list[dict[str, Any]]:
        """Retrieves all agent sessions (legacy sync method)."""
        with self._get_connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM agent_sessions ORDER BY started_at DESC")
            return [dict(row) for row in cursor.fetchall()]

    def list_sessions_by_project_sync(self, project_id: int) -> list[dict[str, Any]]:
        """Retrieves agent sessions for a specific project (legacy sync method)."""
        with self._get_connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM agent_sessions WHERE project_id = ? ORDER BY started_at DESC",
                (project_id,),
            )
            return [dict(row) for row in cursor.fetchall()]

    # PersistenceStrategy protocol implementation
    async def initialize(self) -> None:
        """Initialize the persistence strategy."""
        pass

    async def close(self) -> None:
        """Close the persistence strategy."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    @property
    def session_repo(self) -> SessionRepository:
        """Get the session repository."""
        return self  # type: ignore[return-value]

    @property
    def memory_repo(self) -> MemoryRepository:
        """Get the memory repository."""
        return self  # type: ignore[return-value]

    @property
    def vector_repo(self) -> VectorSearchRepository:
        """Get the vector search repository."""
        return self  # type: ignore[return-value]
