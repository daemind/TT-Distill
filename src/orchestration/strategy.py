"""Minimal abstract interfaces for orchestration components.

This module defines only the necessary protocols for MACA and Brainstorming Salon.
"""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Protocol, runtime_checkable


class BrainstormingMode(Enum):
    """Brainstorming salon mode."""

    IDEA = "IDEA"
    CRITIQUE = "CRITIQUE"
    RESEARCH = "RESEARCH"
    RESOLUTION = "RESOLUTION"


@dataclass
class BrainstormingMessage:
    """Message in a brainstorming session."""

    session_id: str
    agent_name: str
    mode: BrainstormingMode
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    parent_message_id: str | None = None


@dataclass
class BrainstormingSession:
    """Brainstorming session for conflict resolution."""

    session_id: str
    title: str
    description: str
    status: str = "active"
    messages: list[BrainstormingMessage] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    resolved_at: datetime | None = None
    resolution: str | None = None


@runtime_checkable
class BrainstormingSalonProtocol(Protocol):
    """Protocol for brainstorming salon conflict resolution."""

    @abstractmethod
    async def create_session(
        self,
        title: str,
        description: str,
        participants: list[str],
    ) -> BrainstormingSession:
        """Create a new brainstorming session."""
        ...

    @abstractmethod
    async def add_message(
        self,
        session_id: str,
        agent_name: str,
        mode: BrainstormingMode,
        content: str,
        parent_message_id: str | None = None,
    ) -> BrainstormingMessage:
        """Add a message to a session."""
        ...

    @abstractmethod
    async def resolve_conflict(
        self,
        session_id: str,
        resolution: str,
    ) -> BrainstormingSession:
        """Resolve a conflict in a session."""
        ...

    @abstractmethod
    def get_session_messages(
        self,
        session_id: str,
    ) -> list[BrainstormingMessage]:
        """Get all messages in a session."""
        ...

    @abstractmethod
    async def auto_resolve(
        self,
        session_id: str,
        consensus_threshold: float = 0.7,
    ) -> bool:
        """Attempt automatic resolution based on consensus."""
        ...
