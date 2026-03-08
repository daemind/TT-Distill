"""Brainstorming salon implementation for conflict resolution.

This module provides the BrainstormingSalon class for IRC-style
conflict resolution with IDEA, CRITIQUE, RESEARCH, and RESOLUTION modes.
"""

from __future__ import annotations

import uuid

from .strategy import (
    BrainstormingMessage,
    BrainstormingMode,
    BrainstormingSalonProtocol,
    BrainstormingSession,
)

# Constants
MIN_MESSAGES_FOR_AUTO_RESOLVE = 2


class BrainstormingSalon(BrainstormingSalonProtocol):
    """Implementation of brainstorming salon for conflict resolution."""

    def __init__(self) -> None:
        """Initialize the brainstorming salon."""
        self._sessions: dict[str, BrainstormingSession] = {}

    async def create_session(
        self,
        title: str,
        description: str,
        participants: list[str],
    ) -> BrainstormingSession:
        """Create a new brainstorming session.

        Args:
            title: The session title.
            description: The session description.
            participants: List of participant agent names.

        Returns:
            The created brainstorming session.
        """
        session_id = str(uuid.uuid4())
        session = BrainstormingSession(
            session_id=session_id,
            title=title,
            description=description,
            status="active",
        )
        self._sessions[session_id] = session
        return session

    async def add_message(
        self,
        session_id: str,
        agent_name: str,
        mode: BrainstormingMode,
        content: str,
        parent_message_id: str | None = None,
    ) -> BrainstormingMessage:
        """Add a message to a session.

        Args:
            session_id: The session ID.
            agent_name: The name of the agent sending the message.
            mode: The brainstorming mode.
            content: The message content.
            parent_message_id: Optional parent message ID for threading.

        Returns:
            The created message.

        Raises:
            ValueError: If the session doesn't exist.
        """
        if session_id not in self._sessions:
            raise ValueError(f"Session '{session_id}' not found")

        session = self._sessions[session_id]

        message = BrainstormingMessage(
            session_id=session_id,
            agent_name=agent_name,
            mode=mode,
            content=content,
            parent_message_id=parent_message_id,
        )

        session.messages.append(message)
        return message

    async def resolve_conflict(
        self,
        session_id: str,
        resolution: str,
    ) -> BrainstormingSession:
        """Resolve a conflict in a session.

        Args:
            session_id: The session ID.
            resolution: The resolution text.

        Returns:
            The updated session.

        Raises:
            ValueError: If the session doesn't exist.
        """
        if session_id not in self._sessions:
            raise ValueError(f"Session '{session_id}' not found")

        session = self._sessions[session_id]
        session.status = "resolved"
        session.resolution = resolution
        return session

    def get_session_messages(
        self,
        session_id: str,
    ) -> list[BrainstormingMessage]:
        """Get all messages in a session.

        Args:
            session_id: The session ID.

        Returns:
            List of messages in the session.

        Raises:
            ValueError: If the session doesn't exist.
        """
        if session_id not in self._sessions:
            raise ValueError(f"Session '{session_id}' not found")

        return self._sessions[session_id].messages.copy()

    async def auto_resolve(
        self,
        session_id: str,
        consensus_threshold: float = 0.7,
    ) -> bool:
        """Attempt automatic resolution based on consensus.

        Args:
            session_id: The session ID.
            consensus_threshold: Required consensus ratio (0.0 to 1.0).

        Returns:
            True if auto-resolution succeeded, False otherwise.

        Raises:
            ValueError: If the session doesn't exist.
        """
        if session_id not in self._sessions:
            raise ValueError(f"Session '{session_id}' not found")

        session = self._sessions[session_id]

        if len(session.messages) < MIN_MESSAGES_FOR_AUTO_RESOLVE:
            return False

        resolution_messages = [
            msg for msg in session.messages if msg.mode == BrainstormingMode.RESOLUTION
        ]

        if len(resolution_messages) < MIN_MESSAGES_FOR_AUTO_RESOLVE:
            return False

        unique_resolutions: set[str] = set()
        for msg in resolution_messages:
            unique_resolutions.add(msg.content)

        consensus_ratio = len(unique_resolutions) / len(resolution_messages)

        if consensus_ratio >= consensus_threshold:
            resolution = resolution_messages[0].content
            await self.resolve_conflict(session_id, resolution)
            return True

        return False

    def get_session(self, session_id: str) -> BrainstormingSession | None:
        """Get a session by ID.

        Args:
            session_id: The session ID.

        Returns:
            The session if found, None otherwise.
        """
        return self._sessions.get(session_id)

    def list_sessions(self) -> list[BrainstormingSession]:
        """List all sessions.

        Returns:
            List of all sessions.
        """
        return list(self._sessions.values())

    def get_active_sessions(self) -> list[BrainstormingSession]:
        """Get all active sessions.

        Returns:
            List of active sessions.
        """
        return [
            session for session in self._sessions.values() if session.status == "active"
        ]

    def get_session_by_title(self, title: str) -> BrainstormingSession | None:
        """Get a session by title.

        Args:
            title: The session title.

        Returns:
            The session if found, None otherwise.
        """
        for session in self._sessions.values():
            if session.title == title:
                return session
        return None

    def delete_session(self, session_id: str) -> bool:
        """Delete a session.

        Args:
            session_id: The session ID.

        Returns:
            True if the session was deleted, False if it didn't exist.
        """
        if session_id not in self._sessions:
            return False

        del self._sessions[session_id]
        return True

    def get_message_count(self, session_id: str) -> int:
        """Get the message count for a session.

        Args:
            session_id: The session ID.

        Returns:
            The number of messages in the session.
        """
        session = self._sessions.get(session_id)
        if session is None:
            return 0
        return len(session.messages)

    def get_messages_by_mode(
        self,
        session_id: str,
        mode: BrainstormingMode,
    ) -> list[BrainstormingMessage]:
        """Get messages by mode in a session.

        Args:
            session_id: The session ID.
            mode: The mode to filter by.

        Returns:
            List of messages with the specified mode.
        """
        session = self._sessions.get(session_id)
        if session is None:
            return []

        return [msg for msg in session.messages if msg.mode == mode]

    def get_participants(self, session_id: str) -> list[str]:
        """Get unique participants in a session.

        Args:
            session_id: The session ID.

        Returns:
            List of unique participant names.
        """
        session = self._sessions.get(session_id)
        if session is None:
            return []

        return [msg.agent_name for msg in session.messages]

    def clear(self) -> None:
        """Clear all sessions."""
        self._sessions.clear()
