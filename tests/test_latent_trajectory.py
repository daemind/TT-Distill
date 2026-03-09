"""Tests for LatentTrajectoryBuffer.

This module implements the "Tampon de Trajectoire Latente" - short-term memory
for tracking intermediate latent vectors between adapter swaps.

Test coverage targets:
- Step recording with tensor detachment
- FIFO eviction (max 100 steps)
- Trajectory analysis
- Adapter usage logging
"""

from __future__ import annotations

import numpy as np
import pytest

from src.orchestration.latent_trajectory import (
    LatentTrajectoryBuffer,
    TrajectoryStep,
)


class TestTrajectoryStep:
    """Tests for TrajectoryStep dataclass."""

    def test_create_step(self) -> None:
        """Test creating a trajectory step."""
        step = TrajectoryStep(
            step_id=1,
            latent_vector=np.array([1.0, 2.0, 3.0]),
            adapter_used="topology",
            adapter_weight=0.8,
            success=True,
        )

        assert step.step_id == 1
        assert step.adapter_used == "topology"
        assert step.adapter_weight == 0.8
        assert step.success is True

    def test_step_to_dict(self) -> None:
        """Test converting step to dictionary."""
        step = TrajectoryStep(
            step_id=1,
            latent_vector=np.array([1.0, 2.0, 3.0]),
            adapter_used="topology",
            adapter_weight=0.8,
            success=True,
        )

        d = step.to_dict()

        assert d["step_id"] == 1
        assert d["adapter_used"] == "topology"
        assert d["adapter_weight"] == 0.8
        assert d["success"] is True
        assert "latent_vector" in d

    def test_step_from_dict(self) -> None:
        """Test creating step from dictionary."""
        d = {
            "step_id": 1,
            "latent_vector": [1.0, 2.0, 3.0],
            "adapter_used": "topology",
            "adapter_weight": 0.8,
            "timestamp": "2024-01-01T00:00:00",
            "success": True,
        }

        step = TrajectoryStep.from_dict(d)

        assert step.step_id == 1
        assert step.adapter_used == "topology"
        assert step.adapter_weight == 0.8
        assert step.success is True
        np.testing.assert_array_equal(step.latent_vector, d["latent_vector"])

class TestLatentTrajectoryBufferInit:
    """Tests for LatentTrajectoryBuffer initialization."""

    def test_init_default(self) -> None:
        """Test default initialization."""
        buffer = LatentTrajectoryBuffer()

        assert buffer.max_steps == 100
        assert len(buffer.get_trajectory()) == 0

    def test_init_custom_max_steps(self) -> None:
        """Test initialization with custom max steps."""
        buffer = LatentTrajectoryBuffer(max_steps=50)

        assert buffer.max_steps == 50


class TestRecordStep:
    """Tests for recording trajectory steps."""

    def test_record_step_numpy(self) -> None:
        """Test recording step with numpy array."""
        buffer = LatentTrajectoryBuffer()
        latent = np.array([1.0, 2.0, 3.0])

        buffer.record_step(latent, "topology", 0.8, success=True)

        assert len(buffer.get_trajectory()) == 1
        step = buffer.get_trajectory()[0]
        assert step.adapter_used == "topology"
        assert step.adapter_weight == 0.8
        assert step.success is True

    def test_record_step_torch_tensor(self) -> None:
        """Test recording step with PyTorch tensor (detachment)."""
        try:
            import torch

            buffer = LatentTrajectoryBuffer()
            latent = torch.tensor([1.0, 2.0, 3.0])

            buffer.record_step(latent, "topology", 0.8, success=True)

            assert len(buffer.get_trajectory()) == 1
            step = buffer.get_trajectory()[0]
            # Should be numpy array after detachment
            assert isinstance(step.latent_vector, np.ndarray)
        except ImportError:
            pytest.skip("PyTorch not installed")
        except (RuntimeError, SystemExit) as e:
            # Skip if PyTorch/Metal OpenMP conflict causes segfault
            error_msg = str(e).lower()
            if "omp" in error_msg or "thread" in error_msg or "openmp" in error_msg:
                pytest.skip("PyTorch/Metal OpenMP conflict detected")
            raise

    def test_record_step_jax_array(self) -> None:
        """Test recording step with JAX array (numpy conversion)."""
        try:
            import jax.numpy as jnp

            buffer = LatentTrajectoryBuffer()
            latent = jnp.array([1.0, 2.0, 3.0])

            buffer.record_step(latent, "topology", 0.8, success=True)

            assert len(buffer.get_trajectory()) == 1
            step = buffer.get_trajectory()[0]
            # Should be numpy array after conversion
            assert isinstance(step.latent_vector, np.ndarray)
        except ImportError:
            pytest.skip("JAX not installed")

    def test_record_step_updates_step_id(self) -> None:
        """Test that step id increments correctly."""
        buffer = LatentTrajectoryBuffer()

        buffer.record_step(np.array([1.0]), "a", 0.5)
        buffer.record_step(np.array([2.0]), "b", 0.5)
        buffer.record_step(np.array([3.0]), "c", 0.5)

        trajectory = buffer.get_trajectory()
        assert trajectory[0].step_id == 1
        assert trajectory[1].step_id == 2
        assert trajectory[2].step_id == 3


class TestFifoEviction:
    """Tests for FIFO eviction policy."""

    def test_fifo_eviction(self) -> None:
        """Test that oldest steps are evicted when max reached."""
        buffer = LatentTrajectoryBuffer(max_steps=3)

        buffer.record_step(np.array([1.0]), "a", 0.5)
        buffer.record_step(np.array([2.0]), "b", 0.5)
        buffer.record_step(np.array([3.0]), "c", 0.5)
        assert len(buffer.get_trajectory()) == 3

        buffer.record_step(np.array([4.0]), "d", 0.5)
        assert len(buffer.get_trajectory()) == 3
        # Oldest step should be evicted
        assert buffer.get_trajectory()[0].adapter_used == "b"

    def test_fifo_eviction_preserves_order(self) -> None:
        """Test that FIFO eviction preserves chronological order."""
        buffer = LatentTrajectoryBuffer(max_steps=3)

        for i in range(5):
            buffer.record_step(np.array([float(i)]), f"adapter_{i}", 0.5)

        # Should have last 3 steps
        trajectory = buffer.get_trajectory()
        assert len(trajectory) == 3
        assert trajectory[0].adapter_used == "adapter_2"
        assert trajectory[1].adapter_used == "adapter_3"
        assert trajectory[2].adapter_used == "adapter_4"


class TestClear:
    """Tests for clearing trajectory buffer."""

    def test_clear(self) -> None:
        """Test clearing trajectory buffer."""
        buffer = LatentTrajectoryBuffer()

        buffer.record_step(np.array([1.0]), "a", 0.5)
        buffer.record_step(np.array([2.0]), "b", 0.5)

        buffer.clear()

        assert len(buffer.get_trajectory()) == 0


class TestGetTrajectory:
    """Tests for getting trajectory."""

    def test_get_trajectory(self) -> None:
        """Test getting trajectory list."""
        buffer = LatentTrajectoryBuffer()

        buffer.record_step(np.array([1.0]), "a", 0.5)
        buffer.record_step(np.array([2.0]), "b", 0.5)

        trajectory = buffer.get_trajectory()

        assert len(trajectory) == 2
        assert trajectory[0].adapter_used == "a"
        assert trajectory[1].adapter_used == "b"


class TestGetAdapterLog:
    """Tests for getting adapter usage log."""

    def test_get_adapter_log(self) -> None:
        """Test getting chronological adapter log."""
        buffer = LatentTrajectoryBuffer()

        buffer.record_step(np.array([1.0]), "topology", 0.8, success=True)
        buffer.record_step(np.array([2.0]), "translation", 1.0, success=False)
        buffer.record_step(np.array([3.0]), "topology", 0.9, success=True)

        log = buffer.get_adapter_log()

        assert len(log) == 3
        assert log[0] == "Step 1: topology(0.8)"
        assert log[1] == "Step 2: translation(1.0)"
        assert log[2] == "Step 3: topology(0.9)"


class TestAnalyzeTrajectory:
    """Tests for trajectory analysis."""

    def test_analyze_trajectory_empty(self) -> None:
        """Test analysis of empty trajectory."""
        buffer = LatentTrajectoryBuffer()
        analysis = buffer.analyze_trajectory()

        assert analysis["total_steps"] == 0
        assert analysis["success_rate"] == 0.0
        assert analysis["adapter_frequency"] == {}

    def test_analyze_trajectory_success_rate(self) -> None:
        """Test success rate calculation."""
        buffer = LatentTrajectoryBuffer()

        buffer.record_step(np.array([1.0]), "a", 0.5, success=True)
        buffer.record_step(np.array([2.0]), "b", 0.5, success=False)
        buffer.record_step(np.array([3.0]), "c", 0.5, success=True)

        analysis = buffer.analyze_trajectory()

        assert analysis["total_steps"] == 3
        assert analysis["success_rate"] == 2 / 3

    def test_analyze_trajectory_adapter_frequency(self) -> None:
        """Test adapter frequency calculation."""
        buffer = LatentTrajectoryBuffer()

        buffer.record_step(np.array([1.0]), "topology", 0.8)
        buffer.record_step(np.array([2.0]), "translation", 1.0)
        buffer.record_step(np.array([3.0]), "topology", 0.9)

        analysis = buffer.analyze_trajectory()

        assert analysis["adapter_frequency"]["topology"] == 2
        assert analysis["adapter_frequency"]["translation"] == 1

    def test_analyze_trajectory_success_by_adapter(self) -> None:
        """Test success rate per adapter."""
        buffer = LatentTrajectoryBuffer()

        buffer.record_step(np.array([1.0]), "topology", 0.8, success=True)
        buffer.record_step(np.array([2.0]), "topology", 0.9, success=False)
        buffer.record_step(np.array([3.0]), "translation", 1.0, success=True)

        analysis = buffer.analyze_trajectory()

        assert analysis["success_rate_by_adapter"]["topology"] == 0.5
        assert analysis["success_rate_by_adapter"]["translation"] == 1.0


class TestGetStepsByAdapter:
    """Tests for getting steps by adapter."""

    def test_get_steps_by_adapter(self) -> None:
        """Test filtering steps by adapter name."""
        buffer = LatentTrajectoryBuffer()

        buffer.record_step(np.array([1.0]), "topology", 0.8, success=True)
        buffer.record_step(np.array([2.0]), "translation", 1.0, success=False)
        buffer.record_step(np.array([3.0]), "topology", 0.9, success=True)

        steps = buffer.get_steps_by_adapter("topology")

        assert len(steps) == 2
        assert all(s.adapter_used == "topology" for s in steps)

    def test_get_steps_by_adapter_not_found(self) -> None:
        """Test getting steps for nonexistent adapter."""
        buffer = LatentTrajectoryBuffer()

        buffer.record_step(np.array([1.0]), "topology", 0.8)

        steps = buffer.get_steps_by_adapter("translation")

        assert len(steps) == 0
