"""Tests for AutoDistiller (Feedback & Crystallize).

This module implements the neuro-endocrine feedback loop for weight optimization.

Test coverage targets:
- Distillation loop with solver function
- Weight adjustment heuristics
- Crystallization of successful configurations
- Trajectory analysis integration
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.orchestration.auto_distiller import AutoDistiller
from src.orchestration.dora_blender import DoraBlender
from src.orchestration.latent_trajectory import LatentTrajectoryBuffer
from src.persistence.trajectory_store import TrajectoryStore


class TestAutoDistillerInit:
    """Tests for AutoDistiller initialization."""

    def test_init_default(self) -> None:
        """Test default initialization."""
        blender = MagicMock(spec=DoraBlender)
        trajectory_buffer = MagicMock(spec=LatentTrajectoryBuffer)
        trajectory_store = MagicMock(spec=TrajectoryStore)

        distiller = AutoDistiller(
            blender=blender,
            trajectory_buffer=trajectory_buffer,
            trajectory_store=trajectory_store,
        )

        assert distiller.blender == blender
        assert distiller.trajectory_buffer == trajectory_buffer
        assert distiller.trajectory_store == trajectory_store
        assert distiller.max_attempts == 5

    def test_init_custom_max_attempts(self) -> None:
        """Test initialization with custom max attempts."""
        blender = MagicMock(spec=DoraBlender)
        trajectory_buffer = MagicMock(spec=LatentTrajectoryBuffer)
        trajectory_store = MagicMock(spec=TrajectoryStore)

        distiller = AutoDistiller(
            blender=blender,
            trajectory_buffer=trajectory_buffer,
            trajectory_store=trajectory_store,
            max_attempts=10,
        )

        assert distiller.max_attempts == 10

    def test_init_custom_adapter_paths(self) -> None:
        """Test initialization with custom adapter paths."""
        blender = MagicMock(spec=DoraBlender)
        trajectory_buffer = MagicMock(spec=LatentTrajectoryBuffer)
        trajectory_store = MagicMock(spec=TrajectoryStore)

        adapter_paths = {"topology": "/path/to/topology.bin"}
        distiller = AutoDistiller(
            blender=blender,
            trajectory_buffer=trajectory_buffer,
            trajectory_store=trajectory_store,
            adapter_paths=adapter_paths,
        )

        assert distiller._adapter_paths == adapter_paths


class TestNormalizeWeights:
    """Tests for weight normalization."""

    def test_normalize_weights_sum_one(self) -> None:
        """Test normalization produces weights summing to 1.0."""
        blender = MagicMock(spec=DoraBlender)
        trajectory_buffer = MagicMock(spec=LatentTrajectoryBuffer)
        trajectory_store = MagicMock(spec=TrajectoryStore)

        distiller = AutoDistiller(
            blender=blender,
            trajectory_buffer=trajectory_buffer,
            trajectory_store=trajectory_store,
        )

        weights = {"a": 2.0, "b": 3.0, "c": 5.0}
        normalized = distiller._normalize_weights(weights)

        total = sum(normalized.values())
        assert abs(total - 1.0) < 1e-6
        assert normalized["a"] == pytest.approx(0.2)
        assert normalized["b"] == pytest.approx(0.3)
        assert normalized["c"] == pytest.approx(0.5)

    def test_normalize_weights_all_zero(self) -> None:
        """Test normalization distributes evenly when all zero."""
        blender = MagicMock(spec=DoraBlender)
        trajectory_buffer = MagicMock(spec=LatentTrajectoryBuffer)
        trajectory_store = MagicMock(spec=TrajectoryStore)

        distiller = AutoDistiller(
            blender=blender,
            trajectory_buffer=trajectory_buffer,
            trajectory_store=trajectory_store,
        )

        weights = {"a": 0.0, "b": 0.0, "c": 0.0}
        normalized = distiller._normalize_weights(weights)

        assert normalized == {"a": 1 / 3, "b": 1 / 3, "c": 1 / 3}

    def test_normalize_weights_empty(self) -> None:
        """Test normalization handles empty weights."""
        blender = MagicMock(spec=DoraBlender)
        trajectory_buffer = MagicMock(spec=LatentTrajectoryBuffer)
        trajectory_store = MagicMock(spec=TrajectoryStore)

        distiller = AutoDistiller(
            blender=blender,
            trajectory_buffer=trajectory_buffer,
            trajectory_store=trajectory_store,
        )

        normalized = distiller._normalize_weights({})

        assert normalized == {}


class TestIsSuccess:
    """Tests for success checking."""

    def test_is_success_none_prediction(self) -> None:
        """Test None prediction is failure."""
        blender = MagicMock(spec=DoraBlender)
        trajectory_buffer = MagicMock(spec=LatentTrajectoryBuffer)
        trajectory_store = MagicMock(spec=TrajectoryStore)

        distiller = AutoDistiller(
            blender=blender,
            trajectory_buffer=trajectory_buffer,
            trajectory_store=trajectory_store,
        )

        assert distiller._is_success(None, {"test": []}) is False

    def test_is_success_empty_prediction(self) -> None:
        """Test empty prediction is failure."""
        blender = MagicMock(spec=DoraBlender)
        trajectory_buffer = MagicMock(spec=LatentTrajectoryBuffer)
        trajectory_store = MagicMock(spec=TrajectoryStore)

        distiller = AutoDistiller(
            blender=blender,
            trajectory_buffer=trajectory_buffer,
            trajectory_store=trajectory_store,
        )

        assert distiller._is_success(np.array([]), {"test": []}) is False

    def test_is_success_match(self) -> None:
        """Test matching prediction is success."""
        blender = MagicMock(spec=DoraBlender)
        trajectory_buffer = MagicMock(spec=LatentTrajectoryBuffer)
        trajectory_store = MagicMock(spec=TrajectoryStore)

        distiller = AutoDistiller(
            blender=blender,
            trajectory_buffer=trajectory_buffer,
            trajectory_store=trajectory_store,
        )

        prediction = np.array([[1, 2], [3, 4]])
        task_data = {
            "test": [{"output": [[1, 2], [3, 4]]}],
        }

        assert distiller._is_success(prediction, task_data) is True

    def test_is_success_mismatch(self) -> None:
        """Test mismatched prediction is failure."""
        blender = MagicMock(spec=DoraBlender)
        trajectory_buffer = MagicMock(spec=LatentTrajectoryBuffer)
        trajectory_store = MagicMock(spec=TrajectoryStore)

        distiller = AutoDistiller(
            blender=blender,
            trajectory_buffer=trajectory_buffer,
            trajectory_store=trajectory_store,
        )

        prediction = np.array([[1, 2], [3, 4]])
        task_data = {
            "test": [{"output": [[5, 6], [7, 8]]}],
        }

        assert distiller._is_success(prediction, task_data) is False


class TestAdjustWeights:
    """Tests for weight adjustment heuristics."""

    def test_adjust_weights_boost_most_used(self) -> None:
        """Test boosting most-used adapter."""
        blender = MagicMock(spec=DoraBlender)
        trajectory_buffer = MagicMock(spec=LatentTrajectoryBuffer)
        trajectory_store = MagicMock(spec=TrajectoryStore)

        # Mock analyze_trajectory to return adapter frequency
        trajectory_buffer.analyze_trajectory.return_value = {
            "adapter_frequency": {"topology": 5, "translation": 2},
            "success_rate_by_adapter": {"topology": 0.8, "translation": 0.5},
        }
        trajectory_buffer.get_steps_by_adapter.return_value = []

        distiller = AutoDistiller(
            blender=blender,
            trajectory_buffer=trajectory_buffer,
            trajectory_store=trajectory_store,
        )

        weights = {"topology": 0.5, "translation": 0.5}
        adjusted = distiller._adjust_weights(weights, {})

        # topology should be boosted
        assert adjusted["topology"] > weights["topology"]

    def test_adjust_weights_penalize_low_success(self) -> None:
        """Test penalizing adapters with low success rate."""
        blender = MagicMock(spec=DoraBlender)
        trajectory_buffer = MagicMock(spec=LatentTrajectoryBuffer)
        trajectory_store = MagicMock(spec=TrajectoryStore)

        # Mock analyze_trajectory
        # translation is most-used (5 steps), topology is less-used (3 steps)
        # topology has low success rate (0%), translation has high success rate (100%)
        trajectory_buffer.analyze_trajectory.return_value = {
            "adapter_frequency": {"topology": 3, "translation": 5},
            "success_rate_by_adapter": {"topology": 0.0, "translation": 1.0},
        }

        # Mock get_steps_by_adapter to return steps with low success for topology
        topology_steps = [
            MagicMock(success=False),
            MagicMock(success=False),
            MagicMock(success=False),
        ]
        translation_steps = [
            MagicMock(success=True),
            MagicMock(success=True),
            MagicMock(success=True),
            MagicMock(success=True),
            MagicMock(success=True),
        ]
        trajectory_buffer.get_steps_by_adapter.side_effect = (
            lambda name: topology_steps if name == "topology" else translation_steps
        )

        distiller = AutoDistiller(
            blender=blender,
            trajectory_buffer=trajectory_buffer,
            trajectory_store=trajectory_store,
        )

        weights = {"topology": 0.5, "translation": 0.5}
        adjusted = distiller._adjust_weights(weights, {})

        # translation should be boosted (most-used, high success)
        assert adjusted["translation"] > weights["translation"]
        # topology should be penalized (low success rate < 0.3)
        assert adjusted["topology"] < weights["topology"]


class TestCrystallize:
    """Tests for crystallization."""

    def test_crystallize_success(self) -> None:
        """Test successful crystallization."""
        blender = MagicMock(spec=DoraBlender)
        trajectory_buffer = MagicMock(spec=LatentTrajectoryBuffer)
        trajectory_store = MagicMock(spec=TrajectoryStore)

        distiller = AutoDistiller(
            blender=blender,
            trajectory_buffer=trajectory_buffer,
            trajectory_store=trajectory_store,
        )

        result = distiller._crystallize(
            weights={"topology": 0.8, "translation": 0.2},
            strategy="topology_heavy",
            task_data={},
        )

        assert result.success is True
        assert result.crystallized is True
        assert result.crystallized_path is not None
        assert result.strategy == "topology_heavy"
        assert result.session_id is not None

    def test_crystallize_saves_adapter(self) -> None:
        """Test crystallization saves fused adapter to file."""
        blender = MagicMock(spec=DoraBlender)
        trajectory_buffer = MagicMock(spec=LatentTrajectoryBuffer)
        trajectory_store = MagicMock(spec=TrajectoryStore)

        # Mock adapter paths
        distiller = AutoDistiller(
            blender=blender,
            trajectory_buffer=trajectory_buffer,
            trajectory_store=trajectory_store,
            adapter_paths={"topology": "tests/data/mock_adapter.bin"},
        )

        with (
            patch.object(Path, "exists", return_value=True),
            patch.object(Path, "open", MagicMock()),
            patch.object(blender, "blend_adapters", return_value={}),
        ):
            result = distiller._crystallize(
                weights={"topology": 1.0},
                strategy="test",
                task_data={},
            )

            assert result.crystallized is True


class TestDistill:
    """Tests for distillation loop."""

    def test_distill_success_first_attempt(self) -> None:
        """Test distillation succeeds on first attempt."""
        blender = MagicMock(spec=DoraBlender)
        trajectory_buffer = MagicMock(spec=LatentTrajectoryBuffer)
        trajectory_store = MagicMock(spec=TrajectoryStore)

        distiller = AutoDistiller(
            blender=blender,
            trajectory_buffer=trajectory_buffer,
            trajectory_store=trajectory_store,
        )

        # Mock solver to succeed immediately
        def mock_solver(task_data, weights):
            return np.array([[1, 2], [3, 4]]), "direct", None

        task_data = {
            "test": [{"output": [[1, 2], [3, 4]]}],
        }

        result = distiller.distill(task_data, {"topology": 0.5}, mock_solver)

        assert result.success is True
        assert result.crystallized is True
        assert result.attempts == 1

    def test_distill_failure_then_success(self) -> None:
        """Test distillation succeeds after adjustment."""
        blender = MagicMock(spec=DoraBlender)
        trajectory_buffer = MagicMock(spec=LatentTrajectoryBuffer)
        trajectory_store = MagicMock(spec=TrajectoryStore)

        # Mock analyze_trajectory to return non-empty adapter_frequency
        trajectory_buffer.analyze_trajectory.return_value = {
            "adapter_frequency": {"topology": 1, "translation": 0},
            "success_rate_by_adapter": {"topology": 0.0, "translation": 1.0},
        }
        trajectory_buffer.get_steps_by_adapter.return_value = []

        distiller = AutoDistiller(
            blender=blender,
            trajectory_buffer=trajectory_buffer,
            trajectory_store=trajectory_store,
            max_attempts=3,
        )

        attempt_count = [0]

        def mock_solver(task_data, weights):
            attempt_count[0] += 1
            if attempt_count[0] == 1:
                return None, "attempt1", None  # Fail
            return np.array([[1, 2], [3, 4]]), "attempt2", None  # Success

        task_data = {
            "test": [{"output": [[1, 2], [3, 4]]}],
        }

        result = distiller.distill(task_data, {"topology": 0.5}, mock_solver)

        assert result.success is True
        assert result.attempts == 2

    def test_distill_max_attempts_reached(self) -> None:
        """Test distillation fails after max attempts."""
        blender = MagicMock(spec=DoraBlender)
        trajectory_buffer = MagicMock(spec=LatentTrajectoryBuffer)
        trajectory_store = MagicMock(spec=TrajectoryStore)

        # Mock analyze_trajectory to return non-empty adapter_frequency
        trajectory_buffer.analyze_trajectory.return_value = {
            "adapter_frequency": {"topology": 1, "translation": 0},
            "success_rate_by_adapter": {"topology": 0.0, "translation": 1.0},
        }
        trajectory_buffer.get_steps_by_adapter.return_value = []

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
        assert result.crystallized is False
        assert result.attempts == 3


class TestCrystallizeWeights:
    """Tests for manual crystallization."""

    def test_crystallize_weights_success(self) -> None:
        """Test manual crystallization succeeds."""
        blender = MagicMock(spec=DoraBlender)
        trajectory_buffer = MagicMock(spec=LatentTrajectoryBuffer)
        trajectory_store = MagicMock(spec=TrajectoryStore)

        distiller = AutoDistiller(
            blender=blender,
            trajectory_buffer=trajectory_buffer,
            trajectory_store=trajectory_store,
        )

        result = distiller.crystallize_weights(
            weights={"topology": 0.8, "translation": 0.2},
            strategy="manual",
            task_id="test_task",
        )

        assert result.success is True
        assert result.crystallized is True
        assert result.strategy == "manual"


class TestGetExpertPath:
    """Tests for getting expert adapter paths."""

    def test_get_expert_path_from_adapter_paths(self) -> None:
        """Test getting path from custom adapter paths."""
        blender = MagicMock(spec=DoraBlender)
        trajectory_buffer = MagicMock(spec=LatentTrajectoryBuffer)
        trajectory_store = MagicMock(spec=TrajectoryStore)

        distiller = AutoDistiller(
            blender=blender,
            trajectory_buffer=trajectory_buffer,
            trajectory_store=trajectory_store,
            adapter_paths={"topology": "/custom/path/topology.bin"},
        )

        path = distiller._get_expert_path("topology")

        assert str(path) == "/custom/path/topology.bin"

    def test_get_expert_path_default(self) -> None:
        """Test getting default path."""
        blender = MagicMock(spec=DoraBlender)
        trajectory_buffer = MagicMock(spec=LatentTrajectoryBuffer)
        trajectory_store = MagicMock(spec=TrajectoryStore)

        distiller = AutoDistiller(
            blender=blender,
            trajectory_buffer=trajectory_buffer,
            trajectory_store=trajectory_store,
        )

        path = distiller._get_expert_path("topology")

        assert "adapters" in str(path)
        assert path.name == "topology.bin"
