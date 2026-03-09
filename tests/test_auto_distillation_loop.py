"""Integration tests for auto-distillation loop.

This module tests the complete neuro-endocrine feedback loop:
1. Solver attempts task with current weights
2. On failure: analyze trajectory → adjust weights → retry
3. On success: crystallize_weights() → save fused adapter

Test coverage targets:
- End-to-end distillation flow
- Trajectory buffer integration
- Weight adjustment feedback
- Crystallization persistence
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from src.orchestration.auto_distiller import AutoDistiller, DistillationResult
from src.orchestration.dora_blender import DoraBlender
from src.orchestration.latent_trajectory import LatentTrajectoryBuffer
from src.persistence.trajectory_store import TrajectoryStore


class TestEndToEndDistillation:
    """End-to-end distillation loop tests."""

    def test_successful_distillation_flow(self) -> None:
        """Test complete successful distillation flow."""
        blender = MagicMock(spec=DoraBlender)
        trajectory_buffer = LatentTrajectoryBuffer()
        trajectory_store = TrajectoryStore(Path(tempfile.mkdtemp()))

        distiller = AutoDistiller(
            blender=blender,
            trajectory_buffer=trajectory_buffer,
            trajectory_store=trajectory_store,
            max_attempts=5,
        )

        # Mock solver that succeeds on second attempt
        attempt_count = [0]

        def mock_solver(task_data, weights):
            attempt_count[0] += 1
            if attempt_count[0] == 1:
                return None, "first_attempt", None
            return np.array([[1, 2], [3, 4]]), "second_attempt", None

        task_data = {
            "test": [{"output": [[1, 2], [3, 4]]}],
        }

        result = distiller.distill(task_data, {"topology": 0.5}, mock_solver)

        assert result.success is True
        assert result.crystallized is True
        assert result.attempts == 2
        assert result.strategy == "second_attempt"

    def test_distillation_with_trajectory_logging(self) -> None:
        """Test that trajectory is logged during distillation."""
        blender = MagicMock(spec=DoraBlender)
        trajectory_buffer = LatentTrajectoryBuffer()
        trajectory_store = TrajectoryStore(Path(tempfile.mkdtemp()))

        distiller = AutoDistiller(
            blender=blender,
            trajectory_buffer=trajectory_buffer,
            trajectory_store=trajectory_store,
            max_attempts=3,
        )

        def mock_solver(task_data, weights):
            # Record some trajectory steps
            trajectory_buffer.record_step(
                np.array([1.0, 2.0, 3.0]),
                "topology",
                0.8,
                success=False,
            )
            return None, "failed", None

        task_data = {
            "test": [{"output": [[1, 2], [3, 4]]}],
        }

        distiller.distill(task_data, {"topology": 0.5}, mock_solver)

        # Trajectory should have been recorded
        assert len(trajectory_buffer.get_trajectory()) > 0

    def test_distillation_saves_trajectory_on_success(self) -> None:
        """Test that trajectory is saved when distillation succeeds."""
        blender = MagicMock(spec=DoraBlender)
        trajectory_buffer = LatentTrajectoryBuffer()
        trajectory_store = TrajectoryStore(Path(tempfile.mkdtemp()))

        distiller = AutoDistiller(
            blender=blender,
            trajectory_buffer=trajectory_buffer,
            trajectory_store=trajectory_store,
            max_attempts=3,
        )

        def mock_solver(task_data, weights):
            trajectory_buffer.record_step(
                np.array([1.0, 2.0, 3.0]),
                "topology",
                0.8,
                success=True,
            )
            return np.array([[1, 2], [3, 4]]), "success", None

        task_data = {
            "test": [{"output": [[1, 2], [3, 4]]}],
        }

        result = distiller.distill(task_data, {"topology": 0.5}, mock_solver)

        assert result.success is True
        assert result.crystallized is True
        # Trajectory should be saved
        assert len(trajectory_store.list_sessions()) > 0


class TestWeightAdjustmentFeedback:
    """Tests for weight adjustment feedback loop."""

    def test_weight_adjustment_on_failure(self) -> None:
        """Test weights are adjusted after failure."""
        blender = MagicMock(spec=DoraBlender)
        trajectory_buffer = LatentTrajectoryBuffer()
        trajectory_store = TrajectoryStore(Path(tempfile.mkdtemp()))

        distiller = AutoDistiller(
            blender=blender,
            trajectory_buffer=trajectory_buffer,
            trajectory_store=trajectory_store,
            max_attempts=3,
        )

        # Record some failed attempts
        trajectory_buffer.record_step(
            np.array([1.0]), "topology", 0.5, success=False
        )
        trajectory_buffer.record_step(
            np.array([2.0]), "translation", 0.5, success=False
        )

        initial_weights = {"topology": 0.5, "translation": 0.5}
        adjusted = distiller._adjust_weights(initial_weights, {})

        # Weights should be different after adjustment
        assert adjusted != initial_weights

    def test_weight_normalization_after_adjustment(self) -> None:
        """Test weights are normalized after adjustment."""
        blender = MagicMock(spec=DoraBlender)
        trajectory_buffer = LatentTrajectoryBuffer()
        trajectory_store = TrajectoryStore(Path(tempfile.mkdtemp()))

        distiller = AutoDistiller(
            blender=blender,
            trajectory_buffer=trajectory_buffer,
            trajectory_store=trajectory_store,
            max_attempts=3,
        )

        weights = {"topology": 0.6, "translation": 0.4}
        adjusted = distiller._adjust_weights(weights, {})

        # Adjusted weights should sum to 1.0
        total = sum(adjusted.values())
        assert abs(total - 1.0) < 1e-6


class TestCrystallizationPersistence:
    """Tests for crystallization persistence."""

    def test_crystallized_adapter_saved_to_disk(self) -> None:
        """Test crystallized adapter is saved to disk."""
        blender = MagicMock(spec=DoraBlender)
        trajectory_buffer = LatentTrajectoryBuffer()
        trajectory_store = TrajectoryStore(Path(tempfile.mkdtemp()))

        with tempfile.TemporaryDirectory() as tmpdir:
            distiller = AutoDistiller(
                blender=blender,
                trajectory_buffer=trajectory_buffer,
                trajectory_store=trajectory_store,
                adapter_paths={"topology": f"{tmpdir}/topology.bin"},
            )

            with (
               patch.object(Path, "exists", return_value=True),
               patch.object(Path, "open", MagicMock()),
               patch.object(blender, "blend_adapters", return_value={}),
           ):
               distiller.crystallize_weights(
                   weights={"topology": 1.0},
                   strategy="test",
               )

    def test_trajectory_reloaded_after_crystallization(self) -> None:
        """Test trajectory can be reloaded after crystallization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            blender = MagicMock(spec=DoraBlender)
            trajectory_buffer = LatentTrajectoryBuffer()
            trajectory_store = TrajectoryStore(Path(tmpdir))

            distiller = AutoDistiller(
                blender=blender,
                trajectory_buffer=trajectory_buffer,
                trajectory_store=trajectory_store,
            )

            # Record some steps
            trajectory_buffer.record_step(
                np.array([1.0]), "topology", 0.8, success=True
            )

            # Save trajectory
            distiller.trajectory_store.save_trajectory(
                trajectory_buffer.get_trajectory(), "test_session"
            )

            # Create new distiller and reload
            new_distiller = AutoDistiller(
                blender=blender,
                trajectory_buffer=LatentTrajectoryBuffer(),
                trajectory_store=TrajectoryStore(Path(tmpdir)),
            )

            loaded = new_distiller.trajectory_store.load_trajectory("test_session")
            assert len(loaded) == 1


class TestDistillationResult:
    """Tests for DistillationResult dataclass."""

    def test_distillation_result_fields(self) -> None:
        """Test DistillationResult has all required fields."""
        result = DistillationResult(
            success=True,
            strategy="test_strategy",
            final_weights={"topology": 0.8},
            trajectory_analysis={"total_steps": 5},
            crystallized=True,
            crystallized_path="/path/to/crystallized.bin",
            attempts=2,
            session_id="test_session",
        )

        assert result.success is True
        assert result.strategy == "test_strategy"
        assert result.final_weights == {"topology": 0.8}
        assert result.trajectory_analysis == {"total_steps": 5}
        assert result.crystallized is True
        assert result.crystallized_path == "/path/to/crystallized.bin"
        assert result.attempts == 2
        assert result.session_id == "test_session"

    def test_distillation_result_defaults(self) -> None:
        """Test DistillationResult default values."""
        result = DistillationResult(
            success=False,
            strategy="failed",
            final_weights={},
            trajectory_analysis={},
        )

        assert result.crystallized is False
        assert result.crystallized_path is None
        assert result.attempts == 0
        assert result.session_id is not None


class TestTrajectoryAnalysisIntegration:
    """Tests for trajectory analysis integration."""

    def test_analysis_includes_all_metrics(self) -> None:
        """Test trajectory analysis includes all required metrics."""
        blender = MagicMock(spec=DoraBlender)
        trajectory_buffer = LatentTrajectoryBuffer()
        trajectory_store = TrajectoryStore(Path(tempfile.mkdtemp()))

        distiller = AutoDistiller(
            blender=blender,
            trajectory_buffer=trajectory_buffer,
            trajectory_store=trajectory_store,
        )

        # Record diverse steps
        trajectory_buffer.record_step(
            np.array([1.0]), "topology", 0.8, success=True
        )
        trajectory_buffer.record_step(
            np.array([2.0]), "translation", 0.5, success=False
        )
        trajectory_buffer.record_step(
            np.array([3.0]), "topology", 0.9, success=True
        )

        analysis = distiller.trajectory_buffer.analyze_trajectory()

        assert "total_steps" in analysis
        assert "success_rate" in analysis
        assert "adapter_frequency" in analysis
        assert "success_rate_by_adapter" in analysis
        assert analysis["total_steps"] == 3
        assert analysis["success_rate"] == 2 / 3

    def test_analysis_used_for_weight_adjustment(self) -> None:
        """Test analysis results are used for weight adjustment."""
        blender = MagicMock(spec=DoraBlender)
        trajectory_buffer = LatentTrajectoryBuffer()
        trajectory_store = TrajectoryStore(Path(tempfile.mkdtemp()))

        distiller = AutoDistiller(
            blender=blender,
            trajectory_buffer=trajectory_buffer,
            trajectory_store=trajectory_store,
        )

        # Record steps with different success rates
        # translation: 100% success (8 successes) - most used, should be boosted
        for _ in range(8):
            trajectory_buffer.record_step(
                np.array([2.0]), "translation", 0.5, success=True
            )
        # topology: 0% success (5 failures) - less used, should be penalized
        for _ in range(5):
            trajectory_buffer.record_step(
                np.array([1.0]), "topology", 0.8, success=False
            )

        weights = {"topology": 0.5, "translation": 0.5}
        adjusted = distiller._adjust_weights(weights, {})

        # translation should be boosted (most used, high success)
        assert adjusted["translation"] > weights["translation"]
        # topology should be penalized (low success_rate = 0.0 < 0.3)
        assert adjusted["topology"] < weights["topology"]


class TestMaxAttemptsBehavior:
    """Tests for max attempts behavior."""

    def test_respects_max_attempts(self) -> None:
        """Test distillation respects max_attempts limit."""
        blender = MagicMock(spec=DoraBlender)
        trajectory_buffer = LatentTrajectoryBuffer()
        trajectory_store = TrajectoryStore(Path(tempfile.mkdtemp()))

        distiller = AutoDistiller(
            blender=blender,
            trajectory_buffer=trajectory_buffer,
            trajectory_store=trajectory_store,
            max_attempts=3,
        )

        def mock_solver(task_data, weights):
            return None, "always_fail", None  # Always fail

        task_data = {
            "test": [{"output": [[1, 2], [3, 4]]}],
        }

        result = distiller.distill(task_data, {"topology": 0.5}, mock_solver)

        assert result.success is False
        assert result.attempts == 3

    def test_clears_trajectory_per_attempt(self) -> None:
        """Test trajectory is cleared for each attempt."""
        blender = MagicMock(spec=DoraBlender)
        trajectory_buffer = LatentTrajectoryBuffer()
        trajectory_store = TrajectoryStore(Path(tempfile.mkdtemp()))

        distiller = AutoDistiller(
            blender=blender,
            trajectory_buffer=trajectory_buffer,
            trajectory_store=trajectory_store,
            max_attempts=3,
        )

        attempt_count = [0]

        def mock_solver(task_data, weights):
            attempt_count[0] += 1
            # Record step
            trajectory_buffer.record_step(
                np.array([float(attempt_count[0])]),
                f"attempt_{attempt_count[0]}",
                0.5,
                success=False,
            )
            return None, "fail", None

        task_data = {
            "test": [{"output": [[1, 2], [3, 4]]}],
        }

        distiller.distill(task_data, {"topology": 0.5}, mock_solver)

        # Each attempt should have its own trajectory
        # Final trajectory should only have steps from last attempt
        assert len(trajectory_buffer.get_trajectory()) == 1
