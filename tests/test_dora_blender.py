"""Tests for DoraBlender (Le Moteur de Fusion).

This module implements the "cocktail synaptique" - CPU-based weighted linear
combination of multiple DoRA adapters followed by O(1) Metal swap.

Test coverage targets:
- GEOMETRIC blending mode
- Ring buffer serialization
- Metal integration
- Error handling for structural mismatches
"""

from __future__ import annotations

import pickle
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.orchestration.dora_blender import BlendingMode, DoraBlender
from src.orchestration.metal_swap import MetalDoRASwapper


class TestDoraBlenderInit:
    """Tests for DoraBlender initialization."""

    def test_init_default(self) -> None:
        """Test default initialization creates swapper and ring buffer."""
        with patch.object(DoraBlender, "_validate_structure"):
            blender = DoraBlender()
            assert blender.blending_mode == BlendingMode.GEOMETRIC
            assert blender._ring_buffer_size == 3
            assert len(blender._buffer_ring) == 3

    def test_init_custom_ring_buffer(self) -> None:
        """Test initialization with custom ring buffer size."""
        blender = DoraBlender(ring_buffer_size=5)
        assert blender._ring_buffer_size == 5
        assert len(blender._buffer_ring) == 5

    def test_init_custom_swapper(self) -> None:
        """Test initialization with custom swapper."""
        mock_swapper = MagicMock(spec=MetalDoRASwapper)
        blender = DoraBlender(swapper=mock_swapper)
        assert blender._swapper == mock_swapper

    def test_init_minimum_ring_buffer(self) -> None:
        """Test that ring buffer size is at least 3."""
        blender = DoraBlender(ring_buffer_size=1)
        assert blender._ring_buffer_size == 3


class TestBlendingModes:
    """Tests for blending mode functionality."""

    def test_geometric_blend(self) -> None:
        """Test GEOMETRIC blending mode: W_mix = Sigma(w_i * D_i)."""
        adapter1 = {
            "lora_a": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
            "lora_b": np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32),
            "scales": np.array([1.0, 1.0], dtype=np.float32),
        }
        adapter2 = {
            "lora_a": np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32),
            "lora_b": np.array([[50.0, 60.0], [70.0, 80.0]], dtype=np.float32),
            "scales": np.array([1.0, 1.0], dtype=np.float32),
        }

        adapters = [adapter1, adapter2]
        weights = [0.7, 0.3]

        blender = DoraBlender(blending_mode=BlendingMode.GEOMETRIC)
        fused = blender._blend_geometric(adapters, weights)

        # Expected: 0.7 * adapter1 + 0.3 * adapter2
        expected_lora_a = 0.7 * adapter1["lora_a"] + 0.3 * adapter2["lora_a"]
        expected_lora_b = 0.7 * adapter1["lora_b"] + 0.3 * adapter2["lora_b"]

        np.testing.assert_array_almost_equal(fused["lora_a"], expected_lora_a)
        np.testing.assert_array_almost_equal(fused["lora_b"], expected_lora_b)

    def test_tiling_mode_not_implemented(self) -> None:
        """Test that TILING mode raises NotImplementedError."""
        blender = DoraBlender(blending_mode=BlendingMode.TILING)
        with pytest.raises(NotImplementedError, match="Tiling mode requires"):
            blender.blend_adapters([], [])

    def test_hybrid_mode_not_implemented(self) -> None:
        """Test that HYBRID mode raises NotImplementedError."""
        blender = DoraBlender(blending_mode=BlendingMode.HYBRID)
        with pytest.raises(NotImplementedError, match="Hybrid mode requires"):
            blender.blend_adapters([], [])

    def test_set_blending_mode(self) -> None:
        """Test setting blending mode."""
        blender = DoraBlender()
        blender.blending_mode = BlendingMode.TILING
        assert blender.blending_mode == BlendingMode.TILING


class TestAdapterLoading:
    """Tests for adapter loading functionality."""

    def test_load_dict_adapter(self) -> None:
        """Test loading adapter from dict."""
        adapter_dict = {
            "lora_a": np.array([[1.0, 2.0]], dtype=np.float32),
            "lora_b": np.array([[3.0, 4.0]], dtype=np.float32),
            "scales": np.array([1.0], dtype=np.float32),
        }

        blender = DoraBlender()
        loaded = blender._load_adapter(adapter_dict)

        assert loaded == adapter_dict

    def test_load_file_adapter(self) -> None:
        """Test loading adapter from pickle file."""
        adapter_dict = {
            "lora_a": np.array([[1.0, 2.0]], dtype=np.float32),
            "lora_b": np.array([[3.0, 4.0]], dtype=np.float32),
            "scales": np.array([1.0], dtype=np.float32),
        }

        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
            pickle.dump(adapter_dict, f)
            temp_path = f.name

        try:
            blender = DoraBlender()
            loaded = blender._load_adapter(temp_path)

            np.testing.assert_array_equal(loaded["lora_a"], adapter_dict["lora_a"])
            np.testing.assert_array_equal(loaded["lora_b"], adapter_dict["lora_b"])
        finally:
            Path(temp_path).unlink()

    def test_load_nonexistent_file(self) -> None:
        """Test loading from nonexistent file raises FileNotFoundError."""
        blender = DoraBlender()
        with pytest.raises(FileNotFoundError, match="Adapter not found"):
            blender._load_adapter("/nonexistent/path/adapter.bin")


class TestStructureValidation:
    """Tests for adapter structure validation."""

    def test_validate_empty_adapters(self) -> None:
        """Test validation fails with empty adapter list."""
        blender = DoraBlender()
        with pytest.raises(ValueError, match="No adapters provided"):
            blender._validate_structure([])

    def test_validate_key_mismatch(self) -> None:
        """Test validation fails with different keys."""
        adapter1 = {"lora_a": np.array([1.0]), "lora_b": np.array([2.0])}
        adapter2 = {"lora_a": np.array([3.0]), "lora_c": np.array([4.0])}

        blender = DoraBlender()
        with pytest.raises(ValueError, match="Keys mismatch"):
            blender._validate_structure([adapter1, adapter2])

    def test_validate_shape_mismatch(self) -> None:
        """Test validation fails with different shapes."""
        adapter1 = {"lora_a": np.array([[1.0, 2.0]])}
        adapter2 = {"lora_a": np.array([[1.0, 2.0, 3.0]])}

        blender = DoraBlender()
        with pytest.raises(ValueError, match="Shape mismatch"):
            blender._validate_structure([adapter1, adapter2])

    def test_validate_success(self) -> None:
        """Test validation passes with matching structure."""
        adapter1 = {
            "lora_a": np.array([[1.0, 2.0]]),
            "lora_b": np.array([[3.0, 4.0]]),
        }
        adapter2 = {
            "lora_a": np.array([[5.0, 6.0]]),
            "lora_b": np.array([[7.0, 8.0]]),
        }

        blender = DoraBlender()
        # Should not raise
        blender._validate_structure([adapter1, adapter2])


class TestGatingVectorValidation:
    """Tests for gating vector validation."""

    def test_gating_length_mismatch(self) -> None:
        """Test blending fails with mismatched gating vector length."""
        adapter1 = {"lora_a": np.array([1.0])}
        adapter2 = {"lora_a": np.array([2.0])}

        blender = DoraBlender()
        with pytest.raises(ValueError, match="gating_vector length"):
            blender.blend_adapters([adapter1, adapter2], [0.5])

    def test_gating_vector_sum_not_one(self) -> None:
        """Test blending with weights not summing to 1.0."""
        adapter1 = {"lora_a": np.array([1.0])}
        adapter2 = {"lora_a": np.array([2.0])}

        blender = DoraBlender()
        # Should not raise, blending still works
        fused = blender.blend_adapters([adapter1, adapter2], [0.3, 0.4])
        assert "lora_a" in fused


class TestRingBufferSerialization:
    """Tests for ring buffer serialization."""

    def test_ring_buffer_serialize(self) -> None:
        """Test serialization to contiguous buffer."""
        adapter = {
            "lora_a": np.array([[1.0, 2.0]], dtype=np.float32),
            "lora_b": np.array([[3.0, 4.0]], dtype=np.float32),
            "scales": np.array([1.0, 1.0], dtype=np.float32),
        }

        blender = DoraBlender()
        ptr = blender._ring_buffer_serialize(adapter)

        assert ptr is not None
        assert ptr.value > 0

    def test_ring_buffer_serialize_empty(self) -> None:
        """Test serialization fails with no tensors."""
        adapter = {"empty": "not_array"}

        blender = DoraBlender()
        with pytest.raises(ValueError, match="No tensors found"):
            blender._ring_buffer_serialize(adapter)

    def test_ring_buffer_circular_indexing(self) -> None:
        """Test ring buffer wraps around correctly."""
        adapter = {
            "lora_a": np.array([[1.0]], dtype=np.float32),
            "scales": np.array([1.0], dtype=np.float32),
        }

        blender = DoraBlender(ring_buffer_size=3)

        # Write 4 times to test circular wrapping
        _ = blender._ring_buffer_serialize(adapter)
        _ = blender._ring_buffer_serialize(adapter)
        _ = blender._ring_buffer_serialize(adapter)
        _ = blender._ring_buffer_serialize(adapter)

        # ptr4 should be in same slot as ptr1 (index 0)
        assert blender._write_index == 4
        assert blender._buffer_ring[0] is not None


class TestMetalIntegration:
    """Tests for Metal backend integration."""

    def test_blend_and_swap(self) -> None:
        """Test full blend_and_swap pipeline."""
        adapter1 = {
            "lora_a": np.zeros((2560, 16), dtype=np.float32),
            "lora_b": np.zeros((16, 2560), dtype=np.float32),
            "scales": np.ones((2560,), dtype=np.float32),
        }
        adapter2 = {
            "lora_a": np.zeros((2560, 16), dtype=np.float32),
            "lora_b": np.zeros((16, 2560), dtype=np.float32),
            "scales": np.ones((2560,), dtype=np.float32),
        }

        blender = DoraBlender()
        result = blender.blend_and_swap([adapter1, adapter2], [0.5, 0.5])

        assert "merge_ms" in result
        assert "serialize_ms" in result
        assert "preload_ms" in result
        assert "swap_ms" in result
        assert "total_ms" in result
        assert result["total_ms"] > 0

    def test_preload_only(self) -> None:
        """Test preload_only method."""
        adapter = {
            "lora_a": np.zeros((2560, 16), dtype=np.float32),
            "scales": np.ones((2560,), dtype=np.float32),
        }

        blender = DoraBlender()
        ptr, preload_ms = blender.preload_only(adapter)

        assert ptr is not None
        assert preload_ms >= 0

    def test_swap_only(self) -> None:
        """Test swap_only method."""
        blender = DoraBlender()
        swap_ms = blender.swap_only()

        assert swap_ms >= 0


class TestFusedAdapterSaving:
    """Tests for saving fused adapters."""

    def test_save_fused_adapter(self) -> None:
        """Test saving fused adapter to file."""
        adapter = {
            "lora_a": np.array([[1.0, 2.0]], dtype=np.float32),
            "lora_b": np.array([[3.0, 4.0]], dtype=np.float32),
            "scales": np.array([1.0, 1.0], dtype=np.float32),
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "fused.bin"
            blender = DoraBlender()
            blender.save_fused_adapter(adapter, save_path)

            assert save_path.exists()
            with save_path.open("rb") as f:
                loaded = pickle.load(f)  # noqa: S301

            np.testing.assert_array_equal(loaded["lora_a"], adapter["lora_a"])


class TestBenchmark:
    """Tests for benchmark functionality."""

    def test_benchmark_blend_and_swap(self) -> None:
        """Test benchmarking blend_and_swap pipeline."""
        adapter1 = {
            "lora_a": np.zeros((2560, 16), dtype=np.float32),
            "lora_b": np.zeros((16, 2560), dtype=np.float32),
            "scales": np.ones((2560,), dtype=np.float32),
        }
        adapter2 = {
            "lora_a": np.zeros((2560, 16), dtype=np.float32),
            "lora_b": np.zeros((16, 2560), dtype=np.float32),
            "scales": np.ones((2560,), dtype=np.float32),
        }

        blender = DoraBlender()
        result = blender.benchmark_blend_and_swap(
            [adapter1, adapter2], [0.5, 0.5], iterations=5
        )

        assert result["iterations"] == 5
        assert "total_ms" in result
        assert "merge_ms" in result
        assert "swap_ms" in result
        assert "min" in result["total_ms"]
        assert "max" in result["total_ms"]
        assert "mean" in result["total_ms"]
        assert "median" in result["total_ms"]


class TestMetalAvailableProperty:
    """Tests for metal_available property."""

    def test_metal_available(self) -> None:
        """Test metal_available returns True."""
        blender = DoraBlender()
        assert blender.metal_available is True
