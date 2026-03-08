"""Abstract interfaces for the persistence layer.

This module defines the core protocols that separate the orchestration
strategy from the memory storage and retrieval mechanisms.

The interfaces follow SOLID principles:
- Single Responsibility: Each protocol has a single, well-defined responsibility
- Interface Segregation: Clients depend only on interfaces they use
- Dependency Inversion: High-level modules depend on abstractions, not concretions
"""

from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field


@dataclass
class SessionCreateArgs:
    """Arguments for creating a session."""

    project_id: int
    agent_name: str
    role: str
    task: str | None = None
    prompt: str | None = None
    parent_session_id: int | None = None


@dataclass
class SessionCompleteArgs:
    """Arguments for completing a session."""

    session_id: int
    result: str | None = None
    reasoning: str | None = None
    success: bool | None = None
    tokens_used: int | None = None


@dataclass
class VectorSearchArgs:
    """Arguments for vector search."""

    query: str
    limit: int = 10
    session_id: int | None = None


class SessionData(BaseModel):
    """Data model for agent session information."""

    id: int = Field(..., description="Session ID")
    project_id: int = Field(..., description="Project ID")
    agent_name: str = Field(..., description="Agent name")
    role: str = Field(..., description="Agent role")
    task: str | None = Field(None, description="Task description")
    prompt: str | None = Field(None, description="System prompt")
    result: str | None = Field(None, description="Session result")
    reasoning_content: str | None = Field(None, description="Reasoning content")
    success: bool | None = Field(None, description="Session success status")
    tokens_used: int | None = Field(None, description="Tokens used")
    parent_session_id: int | None = Field(None, description="Parent session ID")
    started_at: str = Field(..., description="Session start timestamp")
    completed_at: str | None = Field(None, description="Session completion timestamp")


class MemoryEntry(BaseModel):
    """Data model for memory entry information."""

    id: int = Field(..., description="Memory entry ID")
    session_id: int = Field(..., description="Session ID")
    text: str = Field(..., description="Memory text content")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Metadata")
    embedding: list[float] | None = Field(None, description="Vector embedding")
    chunk_key: str | None = Field(None, description="Reconciliation key for deduplication")
    created_at: str = Field(..., description="Creation timestamp")


class VectorSearchResult(BaseModel):
    """Data model for vector search results."""

    id: int = Field(..., description="Result ID")
    session_id: int = Field(..., description="Session ID")
    text: str = Field(..., description="Result text")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Metadata")
    similarity: float = Field(..., description="Similarity score")
    embedding: list[float] | None = Field(None, description="Vector embedding")


@runtime_checkable
class SessionRepository(Protocol):
    """Repository protocol for agent session management.

    This protocol defines the interface for creating, reading, updating,
    and deleting agent sessions. It is used by the orchestration engine
    to track agent execution state.
    """

    async def create_session(self, args: SessionCreateArgs) -> int:
        """Create a new agent session.

        Args:
            args: Session creation arguments.

        Returns:
            The ID of the newly created session.
        """
        ...

    async def complete_session(self, args: SessionCompleteArgs) -> None:
        """Complete an agent session with results.

        Args:
            args: Session completion arguments.
        """
        ...

    async def get_session(self, session_id: int) -> SessionData | None:
        """Get a session by ID.

        Args:
            session_id: The session ID to retrieve.

        Returns:
            The session data if found, None otherwise.
        """
        ...

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
        ...


@runtime_checkable
class MemoryRepository(Protocol):
    """Repository protocol for local agent memory.

    This protocol defines the interface for managing local memory entries
    using SQLite with sqlite-vec for vector storage. It provides fast,
    local access to agent working memory.
    """

    @abstractmethod
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
        ...

    @abstractmethod
    async def get_memory(self, memory_id: int) -> MemoryEntry | None:
        """Get a memory entry by ID.

        Args:
            memory_id: The memory entry ID to retrieve.

        Returns:
            The memory entry if found, None otherwise.
        """
        ...

    @abstractmethod
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
        ...

    @abstractmethod
    async def remove_memory(self, memory_id: int) -> bool:
        """Remove a memory entry.

        Args:
            memory_id: The memory entry ID to remove.

        Returns:
            True if the entry was removed, False if not found.
        """
        ...


@runtime_checkable
class VectorSearchRepository(Protocol):
    """Repository protocol for global vector search.

    This protocol defines the interface for vector similarity search
    across all sessions. It is used by Postgres with pgvector or
    external services like CocoIndex for global knowledge retrieval.
    """

    def search(
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
        ...

    def search_by_text(
        self,
        query: str,
        limit: int = 10,
        session_id: int | None = None,
    ) -> list[VectorSearchResult]:
        """Perform vector similarity search by text query.

        This method computes the embedding for the query text and performs
        similarity search. It is a convenience method for text-based queries.

        Args:
            query: The query text to search for.
            limit: Maximum number of results to return.
            session_id: Optional session ID filter.

        Returns:
            List of search results sorted by similarity.
        """
        ...

    def add_vector(
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
        ...

    def remove_vector(self, vector_id: int) -> bool:
        """Remove a vector entry.

        Args:
            vector_id: The vector entry ID to remove.

        Returns:
            True if the entry was removed, False if not found.
        """
        ...


@runtime_checkable
class PersistenceStrategy(Protocol):
    """Main persistence strategy protocol.

    This protocol combines all repository protocols into a single
    interface that represents a complete persistence strategy.
    It is used by the orchestration engine to interact with the
    persistence layer without knowing the underlying implementation.
    """

    @property
    @abstractmethod
    def session_repo(self) -> SessionRepository:
        """Get the session repository."""
        ...

    @property
    @abstractmethod
    def memory_repo(self) -> MemoryRepository:
        """Get the memory repository."""
        ...

    @property
    @abstractmethod
    def vector_repo(self) -> VectorSearchRepository:
        """Get the vector search repository."""
        ...

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the persistence strategy.

        This method should set up any required resources, such as
        database connections, schema creation, or index initialization.
        """
        ...

    @abstractmethod
    async def close(self) -> None:
        """Close the persistence strategy.

        This method should clean up any resources, such as database
        connections or background tasks.
        """
        ...
