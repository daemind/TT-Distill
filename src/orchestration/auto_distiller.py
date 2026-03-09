"""TT-Distill Auto-Distillation Loop (Feedback & Crystallize).

This module implements the neuro-endocrine feedback loop for weight optimization.

Workflow:
1. Solver attempts task with current weights
2. If failure:
   - Analyze trajectory buffer
   - Identify problematic adapters
   - Adjust weights (e.g., increase Translation to 0.85)
   - Retry with new weights
3. If success:
   - Call crystallize_weights()
   - Save fused adapter permanently
   - Register in MCP manifold
"""

from __future__ import annotations

import logging
import pickle
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np

from src.orchestration.dora_blender import DoraBlender
from src.orchestration.latent_trajectory import LatentTrajectoryBuffer
from src.persistence.trajectory_store import TrajectoryStore

logger = logging.getLogger(__name__)


@dataclass
class DistillationResult:
    """Result of auto-distillation attempt."""

    success: bool
    strategy: str
    final_weights: dict[str, float]
    trajectory_analysis: dict[str, Any]
    crystallized: bool = False
    crystallized_path: str | None = None
    attempts: int = 0
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))


class AutoDistiller:
    """
    Neuro-endocrine feedback loop for weight optimization.

    Workflow:
    1. Solver attempts task with current weights
    2. If failure:
       - Analyze trajectory buffer
       - Identify problematic adapters
       - Adjust weights (e.g., increase Translation to 0.85)
       - Retry with new weights
    3. If success:
       - Call crystallize_weights()
       - Save fused adapter permanently
       - Register in MCP manifold
    """

    def __init__(
        self,
        blender: DoraBlender,
        trajectory_buffer: LatentTrajectoryBuffer,
        trajectory_store: TrajectoryStore,
        max_attempts: int = 5,
        adapter_paths: Optional[Dict[str, str]] = None,
    ) -> None:
        """Initialize auto distiller.

        Args:
            blender: DoraBlender instance for adapter fusion.
            trajectory_buffer: LatentTrajectoryBuffer for tracking steps.
            trajectory_store: TrajectoryStore for persistence.
            max_attempts: Maximum distillation attempts before giving up.
            adapter_paths: Dict of adapter_name -> file path.
        """
        self.blender = blender
        self.trajectory_buffer = trajectory_buffer
        self.trajectory_store = trajectory_store
        self.max_attempts = max_attempts
        self._adapter_paths = adapter_paths or {}
        self._session_id = str(uuid.uuid4())

        logger.info(
            "AutoDistiller: initialized with max_attempts=%d",
            max_attempts,
        )

    def distill(
        self,
        task_data: Dict[str, Any],
        initial_weights: Dict[str, float],
        solver_fn: Callable[
            [Dict[str, Any], Dict[str, float]],
            tuple[np.ndarray | None, str, np.ndarray | None],
        ],
    ) -> DistillationResult:
        """
        Execute auto-distillation loop.

        Args:
            task_data: ARC task data.
            initial_weights: Starting expert weights.
            solver_fn: Solver function (task_data, weights) ->
                      (prediction, strategy, latent_vector).
                      latent_vector is optional (can be None).

        Returns:
            DistillationResult with final state.
        """
        current_weights = self._normalize_weights(initial_weights.copy())

        for attempt in range(1, self.max_attempts + 1):
            logger.info(
                "AutoDistiller: distillation attempt %d/%d",
                attempt,
                self.max_attempts,
            )

            # Clear trajectory for this attempt
            self.trajectory_buffer.clear()

            # Attempt to solve
            prediction, strategy, latent_vector = solver_fn(
                task_data, current_weights
            )

            # Check success
            if self._is_success(prediction, task_data):
                # Success! Crystallize
                return self._crystallize(
                    current_weights, strategy, task_data, latent_vector, attempt
                )

            # Failure: Analyze and adjust
            self._analyze_and_adjust(attempt, current_weights, task_data, latent_vector)
            current_weights = self._adjust_weights(current_weights, task_data)

        # Max attempts reached without success
        logger.warning(
            "AutoDistiller: max attempts (%d) reached without success",
            self.max_attempts,
        )
        return DistillationResult(
            success=False,
            strategy=strategy,
            final_weights=current_weights,
            trajectory_analysis=self.trajectory_buffer.analyze_trajectory(),
            crystallized=False,
            crystallized_path=None,
            attempts=self.max_attempts,
            session_id=self._session_id,
        )

    def _analyze_and_adjust(
        self,
        attempt: int,
        weights: Dict[str, float],
        task_data: Dict[str, Any],
        latent_vector: Optional[np.ndarray] = None,
    ) -> None:
        """Analyze trajectory and log adjustment.

        Args:
            attempt: Current attempt number.
            weights: Current adapter weights.
            task_data: ARC task data.
            latent_vector: Optional latent vector from solver. If None,
                          no trajectory step is recorded (no placeholders).
        """
        analysis = self.trajectory_buffer.analyze_trajectory()

        logger.info(
            "AutoDistiller: Attempt %d failed. Analysis: %s",
            attempt,
            analysis,
        )

        # Record failure in trajectory only if we have actual latent data
        # NO PLACEHOLDERS - per AGENT.md strict metal protocol
        if latent_vector is not None:
            self.trajectory_buffer.record_step(
                latent_vector=latent_vector,
                adapter_used="failure_analysis",
                adapter_weight=0.0,
                success=False,
            )

    def _adjust_weights(
        self,
        weights: Dict[str, float],
        task_data: Dict[str, Any],
    ) -> Dict[str, float]:
        """
        Adjust weights based on trajectory analysis.

        Strategy:
        - If success_rate is low, reduce weights of underperforming adapters
        - If certain adapters are frequently used but unsuccessful, decrease their weight
        - Increase weight of adapters that correlate with progress
        """
        analysis = self.trajectory_buffer.analyze_trajectory()
        adjusted = weights.copy()

        # Simple heuristic: boost the most-used adapter (only if it's in current weights)
        adapter_frequency = analysis.get("adapter_frequency", {})
        if adapter_frequency:
            most_used = max(
                adapter_frequency.items(),
                key=lambda x: x[1],
            )[0]
            # Only adjust if adapter is in current weights
            if most_used in adjusted:
                adjusted[most_used] = min(1.0, adjusted[most_used] + 0.05)

        # Penalize adapters with low success rate
        for adapter_name, frequency in adapter_frequency.items():
            adapter_steps = self.trajectory_buffer.get_steps_by_adapter(adapter_name)
            if adapter_steps and frequency > 0:
                success_rate = sum(1 for s in adapter_steps if s.success) / len(adapter_steps)
                if success_rate < 0.3 and adapter_name in adjusted:
                    adjusted[adapter_name] = max(0.0, adjusted[adapter_name] - 0.05)

        # Renormalize
        adjusted = self._normalize_weights(adjusted)

        logger.debug(
            "AutoDistiller: adjusted weights from %s to %s",
            weights,
            adjusted,
        )

        return adjusted

    def _normalize_weights(self, weights: dict[str, float]) -> dict[str, float]:
        """Normalize weights to sum to 1.0."""
        total = sum(weights.values())
        if total == 0:
            # Distribute evenly if all weights are zero
            n = len(weights)
            return dict.fromkeys(weights, 1.0 / n) if n > 0 else {}
        return {k: v / total for k, v in weights.items()}

    def _is_success(
        self,
        prediction: Optional[np.ndarray],
        task_data: Dict[str, Any],
    ) -> bool:
        """Check if prediction matches ground truth.

        Args:
            prediction: Model prediction.
            task_data: Task data containing ground truth.

        Returns:
            True if prediction matches ground truth.
        """
        if prediction is None:
            return False

        # Get ground truth from task data
        if "test" in task_data and len(task_data["test"]) > 0:
            test_pair = task_data["test"][0]
            if "output" in test_pair:
                ground_truth = np.array(test_pair["output"])
                return np.array_equal(prediction, ground_truth)

        return False

    def _crystallize(
        self,
        weights: Dict[str, float],
        strategy: str,
        task_data: Dict[str, Any],
        latent_vector: Optional[np.ndarray] = None,
        attempt: int = 1,
    ) -> DistillationResult:
        """
        Crystallize successful configuration.

        Actions:
        1. Save fused adapter as .bin file
        2. Register in MCP manifold
        3. Update trajectory with success

        Args:
            weights: Final adapter weights.
            strategy: Strategy name that succeeded.
            task_data: ARC task data.
            latent_vector: Optional latent vector from successful solve.
                           If None, no trajectory step is recorded (no placeholders).
            attempt: The attempt number that succeeded.
        """
        # Generate crystallized path
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        crystallized_path = (
            Path(__file__).parent.parent / "adapters" / f"crystallized_{timestamp}.bin"
        )
        crystallized_path.parent.mkdir(parents=True, exist_ok=True)

        # Save fused adapter
        self._save_fused_adapter(weights, crystallized_path)

        # Record success in trajectory only if we have actual latent data
        # NO PLACEHOLDERS - per AGENT.md strict metal protocol
        if latent_vector is not None:
            self.trajectory_buffer.record_step(
                latent_vector=latent_vector,
                adapter_used=strategy,
                adapter_weight=1.0,
                success=True,
            )

        # Save trajectory
        self.trajectory_store.save_trajectory(
            self.trajectory_buffer.get_trajectory(),
            self._session_id,
        )

        logger.info(
            "AutoDistiller: crystallized successful configuration to %s",
            crystallized_path,
        )

        return DistillationResult(
            success=True,
            strategy=strategy,
            final_weights=weights,
            trajectory_analysis=self.trajectory_buffer.analyze_trajectory(),
            crystallized=True,
            crystallized_path=str(crystallized_path),
            attempts=attempt,
            session_id=self._session_id,
        )

    def _save_fused_adapter(
        self,
        weights: dict[str, float],
        path: Path,
    ) -> None:
        """Save fused adapter to .bin file."""
        # Load and blend adapters
        adapters = []
        weight_list = []
        for name, weight in weights.items():
            adapter_path = self._get_expert_path(name)
            if adapter_path.exists():
                adapters.append(str(adapter_path))
                weight_list.append(weight)

        if adapters:
            fused = self.blender.blend_adapters(adapters, weight_list)
            with path.open("wb") as f:
                pickle.dump(fused, f)
        else:
            logger.warning(
                "AutoDistiller: no adapters found to crystallize for weights %s",
                weights,
            )

    def _get_expert_path(self, name: str) -> Path:
        """Get path to expert adapter file."""
        # Check explicit adapter paths first
        if name in self._adapter_paths:
            return Path(self._adapter_paths[name])

        # Default path
        return Path(__file__).parent.parent / "adapters" / f"{name}.bin"

    def crystallize_weights(
        self,
        weights: dict[str, float],
        strategy: str,
        task_id: str | None = None,
        latent_vector: np.ndarray | None = None,
    ) -> DistillationResult:
        """
        Manually crystallize a successful weight configuration.

        Args:
            weights: Expert weights that led to success.
            strategy: Strategy name that worked.
            task_id: Optional task ID for tracking.
            latent_vector: Optional latent vector from successful solve.
                          If None, no trajectory step is recorded (no placeholders).

        Returns:
            DistillationResult with crystallization status.
        """
        # Generate crystallized path
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        crystallized_path = (
            Path(__file__).parent.parent / "adapters" / f"crystallized_{timestamp}.bin"
        )
        crystallized_path.parent.mkdir(parents=True, exist_ok=True)

        # Save fused adapter
        self._save_fused_adapter(weights, crystallized_path)

        # Record success in trajectory only if we have actual latent data
        # NO PLACEHOLDERS - per AGENT.md strict metal protocol
        if latent_vector is not None:
            self.trajectory_buffer.record_step(
                latent_vector=latent_vector,
                adapter_used=strategy,
                adapter_weight=1.0,
                success=True,
            )

        # Save trajectory
        self.trajectory_store.save_trajectory(
            self.trajectory_buffer.get_trajectory(),
            self._session_id,
        )

        logger.info(
            "AutoDistiller: manually crystallized weights to %s",
            crystallized_path,
        )

        return DistillationResult(
            success=True,
            strategy=strategy,
            final_weights=weights,
            trajectory_analysis=self.trajectory_buffer.analyze_trajectory(),
            crystallized=True,
            crystallized_path=str(crystallized_path),
            attempts=1,
            session_id=self._session_id,
        )
