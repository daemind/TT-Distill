"""TT-Distill Trajectory Persistence Store.

This module provides persistent storage for latent trajectories,
enabling cross-session analysis and trajectory replay.

Storage options:
- JSON: Simple file-based storage
- SQLite: Structured storage with queries
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from src.orchestration.latent_trajectory import TrajectoryStep

logger = logging.getLogger(__name__)


class TrajectoryStore:
    """
    Persistent storage for latent trajectories.

    Storage options:
    - JSON: Simple file-based storage
    - SQLite: Structured storage with queries
    """

    def __init__(self, storage_path: Path | str) -> None:
        """Initialize trajectory store.

        Args:
            storage_path: Path to directory for storing trajectory files.
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        logger.debug(
            "TrajectoryStore: initialized at %s",
            self.storage_path,
        )

    # ── JSON Storage ─────────────────────────────────────────────────

    def save_trajectory(
        self,
        trajectory: list[TrajectoryStep],
        session_id: str,
    ) -> None:
        """Save trajectory to JSON file.

        Args:
            trajectory: List of TrajectoryStep objects.
            session_id: Unique identifier for the session.
        """
        file_path = self.storage_path / f"trajectory_{session_id}.json"

        data = {
            "session_id": session_id,
            "steps": [s.to_dict() for s in trajectory],
        }

        with file_path.open("w") as f:
            json.dump(data, f, indent=2)

        logger.debug(
            "TrajectoryStore: saved trajectory with %d steps to %s",
            len(trajectory),
            file_path,
        )

    def load_trajectory(self, session_id: str) -> list[TrajectoryStep]:
        """Load trajectory from JSON file.

        Args:
            session_id: Unique identifier for the session.

        Returns:
            List of TrajectoryStep objects, or empty list if not found.
        """
        file_path = self.storage_path / f"trajectory_{session_id}.json"

        if not file_path.exists():
            logger.debug(
                "TrajectoryStore: trajectory not found for session %s",
                session_id,
            )
            return []

        with file_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        return [TrajectoryStep.from_dict(s) for s in data["steps"]]

    def list_sessions(self) -> list[str]:
        """List all session IDs in the storage directory.

        Returns:
            List of session IDs.
        """
        sessions = []
        for file_path in self.storage_path.glob("trajectory_*.json"):
            # Extract session ID from filename
            session_id = file_path.stem.replace("trajectory_", "")
            sessions.append(session_id)
        return sessions

    def delete_trajectory(self, session_id: str) -> bool:
        """Delete a trajectory file.

        Args:
            session_id: Unique identifier for the session.

        Returns:
            True if deleted, False if not found.
        """
        file_path = self.storage_path / f"trajectory_{session_id}.json"
        if file_path.exists():
            file_path.unlink()
            logger.debug(
                "TrajectoryStore: deleted trajectory for session %s",
                session_id,
            )
            return True
        return False

    # ── Batch Operations ─────────────────────────────────────────────

    def save_multiple_trajectories(
        self,
        trajectories: dict[str, list[TrajectoryStep]],
    ) -> None:
        """Save multiple trajectories at once.

        Args:
            trajectories: Dict of session_id -> trajectory list.
        """
        for session_id, trajectory in trajectories.items():
            self.save_trajectory(trajectory, session_id)

    def load_multiple_trajectories(
        self,
        session_ids: list[str],
    ) -> dict[str, list[TrajectoryStep]]:
        """Load multiple trajectories.

        Args:
            session_ids: List of session IDs to load.

        Returns:
            Dict of session_id -> trajectory list.
        """
        return {
            session_id: self.load_trajectory(session_id)
            for session_id in session_ids
        }

    # ── Analysis Utilities ───────────────────────────────────────────

    def get_all_steps(self) -> list[TrajectoryStep]:
        """Get all steps from all sessions.

        Returns:
            List of all TrajectoryStep objects.
        """
        all_steps: list[TrajectoryStep] = []
        for session_id in self.list_sessions():
            all_steps.extend(self.load_trajectory(session_id))
        return all_steps

    def get_steps_by_adapter(self, adapter_name: str) -> list[TrajectoryStep]:
        """Get all steps that used a specific adapter across all sessions.

        Args:
            adapter_name: Name of the adapter to filter by.

        Returns:
            List of TrajectoryStep objects that used the adapter.
        """
        all_steps = self.get_all_steps()
        return [s for s in all_steps if s.adapter_used == adapter_name]

    def get_success_rate_by_adapter(
        self,
        adapter_name: str,
    ) -> float:
        """Get success rate for a specific adapter across all sessions.

        Args:
            adapter_name: Name of the adapter.

        Returns:
            Success rate (0.0 to 1.0), or 0.0 if no steps found.
        """
        steps = self.get_steps_by_adapter(adapter_name)
        if not steps:
            return 0.0
        return sum(1 for s in steps if s.success) / len(steps)
