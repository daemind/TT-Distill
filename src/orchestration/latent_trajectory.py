"""TT-Distill Latent Trajectory Buffer (Le Tampon de Trajectoire Latente).

This module implements short-term memory for tracking latent trajectories
between adapter swaps. It stores intermediate latent vectors (X_1, X_2, ...)
and maintains a chronological log of adapter usage.

Components:
- State: Stores intermediate latent vectors (X_1, X_2, ...)
- Action Log: Chronological list of adapter usage with weights

CRITICAL: Tensor Detachment
- Input latent_vector may be PyTorch/Metal tensor with computation graph
- MUST detach and convert to CPU numpy to prevent VRAM leaks (OOM)
- Use: latent_vector.detach().cpu().numpy().copy()
"""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TrajectoryStep:
    """Single step in the latent trajectory."""

    step_id: int
    latent_vector: np.ndarray  # X_t: intermediate latent state (CPU numpy)
    adapter_used: str  # Expert name
    adapter_weight: float  # Weight applied
    timestamp: datetime = field(default_factory=datetime.now)
    success: bool = False  # Whether this step led to solution

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "step_id": self.step_id,
            "latent_vector": self.latent_vector.tolist(),
            "adapter_used": self.adapter_used,
            "adapter_weight": self.adapter_weight,
            "timestamp": self.timestamp.isoformat(),
            "success": self.success,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrajectoryStep:
        """Create from dictionary."""
        return cls(
            step_id=data["step_id"],
            latent_vector=np.array(data["latent_vector"]),
            adapter_used=data["adapter_used"],
            adapter_weight=data["adapter_weight"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            success=data.get("success", False),
        )


class LatentTrajectoryBuffer:
    """
    Short-term memory for latent trajectory tracking.

    Components:
    - State: Stores intermediate latent vectors (X_1, X_2, ...)
    - Action Log: Chronological list of adapter usage

    Capacity:
    - Max steps: Configurable (default: 100)
    - Eviction: FIFO when capacity exceeded

    CRITICAL: Tensor Detachment to Prevent VRAM Leaks
    - Input latent_vector may be PyTorch/Metal tensor with computation graph
    - MUST detach and convert to CPU numpy to prevent VRAM leaks (OOM)
    - Use: latent_vector.detach().cpu().numpy().copy()
    """

    def __init__(self, max_steps: int = 100) -> None:
        """Initialize trajectory buffer.

        Args:
            max_steps: Maximum number of steps to retain (FIFO eviction).
        """
        self.max_steps = max_steps
        self._steps: list[TrajectoryStep] = []
        self._step_counter = 0

        logger.debug(
            "LatentTrajectoryBuffer: initialized with max_steps=%d",
            max_steps,
        )

    def record_step(
        self,
        latent_vector: np.ndarray,  # May be torch.Tensor or np.ndarray
        adapter_used: str,
        adapter_weight: float,
        success: bool = False,
    ) -> None:
        """
        Record a new trajectory step.

        CRITICAL: Tensor Detachment to Prevent VRAM Leaks
        - If input is torch.Tensor, detach from computation graph
        - Move to CPU and convert to numpy
        - Make explicit copy to ensure no shared memory

        Args:
            latent_vector: Can be torch.Tensor or np.ndarray.
                         If torch.Tensor, will be detached and converted.
            adapter_used: Name of the adapter/expert used.
            adapter_weight: Weight applied to the adapter.
            success: Whether this step led to a successful solution.
        """
        self._step_counter += 1

        # CRITICAL: Detach tensor from computation graph if PyTorch tensor
        # This prevents VRAM leaks from retained computation graphs
        try:
            import torch  # noqa: PLC0415

            if isinstance(latent_vector, torch.Tensor):
                latent_vector = latent_vector.detach().cpu().numpy().copy()
        except ImportError:
            pass

        try:
            import tensorflow as tf  # noqa: PLC0415

            if isinstance(latent_vector, tf.Tensor):
                latent_vector = latent_vector.numpy().copy()
        except ImportError:
            pass

        if hasattr(latent_vector, "__array__"):
            # JAX array or other array-like - convert via numpy
            latent_vector = np.asarray(latent_vector).copy()
        # If already np.ndarray, use as-is (already CPU memory)

        step = TrajectoryStep(
            step_id=self._step_counter,
            latent_vector=latent_vector,
            adapter_used=adapter_used,
            adapter_weight=adapter_weight,
            success=success,
        )

        self._steps.append(step)

        # Enforce capacity (FIFO eviction)
        if len(self._steps) > self.max_steps:
            self._steps.pop(0)

        logger.debug(
            "LatentTrajectoryBuffer: recorded step %d (adapter=%s, weight=%.2f)",
            self._step_counter,
            adapter_used,
            adapter_weight,
        )

    def get_trajectory(self) -> list[TrajectoryStep]:
        """Get full trajectory history."""
        return self._steps.copy()

    def get_last_n_steps(self, n: int) -> list[TrajectoryStep]:
        """Get last n steps.

        Args:
            n: Number of steps to retrieve.

        Returns:
            List of the last n TrajectoryStep objects.
        """
        return self._steps[-n:] if len(self._steps) >= n else self._steps.copy()

    def get_adapter_log(self) -> list[str]:
        """Get chronological adapter usage log.

        Returns:
            List of formatted strings: "Step N: AdapterName(weight)".
        """
        return [
            f"Step {s.step_id}: {s.adapter_used}({s.adapter_weight})"
            for s in self._steps
        ]

    def analyze_trajectory(self) -> dict[str, Any]:
        """
        Analyze trajectory for patterns.

        Returns:
            Dictionary with:
            - total_steps: Total number of steps
            - unique_adapters: List of unique adapter names
            - adapter_frequency: Dict of adapter -> count
            - success_rate: Ratio of successful steps
            - success_rate_by_adapter: Dict of adapter -> success rate
            - avg_weight: Average adapter weight
        """
        if not self._steps:
            return {
                "total_steps": 0,
                "unique_adapters": [],
                "adapter_frequency": {},
                "success_rate": 0.0,
                "success_rate_by_adapter": {},
                "avg_weight": 0.0,
            }

        adapters = [s.adapter_used for s in self._steps]
        successful = sum(1 for s in self._steps if s.success)

        # Calculate success rate per adapter
        adapter_stats: dict[str, dict[str, int]] = defaultdict(
            lambda: {"total": 0, "successful": 0}
        )
        for s in self._steps:
            adapter_stats[s.adapter_used]["total"] += 1
            if s.success:
                adapter_stats[s.adapter_used]["successful"] += 1

        success_rate_by_adapter: dict[str, float] = {
            adapter: stats["successful"] / stats["total"]
            for adapter, stats in adapter_stats.items()
        }

        return {
            "total_steps": len(self._steps),
            "unique_adapters": list(set(adapters)),
            "adapter_frequency": dict(Counter(adapters)),
            "success_rate": successful / len(self._steps),
            "success_rate_by_adapter": success_rate_by_adapter,
            "avg_weight": np.mean([s.adapter_weight for s in self._steps]),
        }

    def clear(self) -> None:
        """Clear all trajectory steps."""
        self._steps.clear()
        self._step_counter = 0
        logger.debug("LatentTrajectoryBuffer: cleared all steps")

    def get_success_rate(self) -> float:
        """Get success rate of trajectory steps.

        Returns:
            Ratio of successful steps (0.0 to 1.0).
        """
        if not self._steps:
            return 0.0
        return sum(1 for s in self._steps if s.success) / len(self._steps)

    def get_last_latent_vector(self) -> np.ndarray | None:
        """Get the last latent vector in the trajectory.

        Returns:
            Last latent vector or None if empty.
        """
        if not self._steps:
            return None
        return self._steps[-1].latent_vector

    def get_steps_by_adapter(self, adapter_name: str) -> list[TrajectoryStep]:
        """Get all steps that used a specific adapter.

        Args:
            adapter_name: Name of the adapter to filter by.

        Returns:
            List of TrajectoryStep objects that used the adapter.
        """
        return [s for s in self._steps if s.adapter_used == adapter_name]

    def get_steps_by_success(self, success: bool) -> list[TrajectoryStep]:
        """Get all steps with a specific success status.

        Args:
            success: Success status to filter by.

        Returns:
            List of TrajectoryStep objects with the specified success status.
        """
        return [s for s in self._steps if s.success == success]
