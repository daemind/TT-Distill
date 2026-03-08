"""Test for Continuous MoA Gating (Adapter Combinator). Hardened version."""

import pickle
from pathlib import Path

import numpy as np
import pytest

from src.orchestration.moa_gating import MoAGater


@pytest.fixture
def tmp_adapters_dir(tmp_path: Path) -> Path:
    """Fixture to create temporary adapters."""
    d = tmp_path / "adapters"
    d.mkdir()

    # Adapter 1
    a1 = {
        "blk.0.attn_k.weight.lora_a": np.ones((16, 128), dtype=np.float32),
        "blk.0.attn_k.weight.lora_b": np.ones((128, 16), dtype=np.float32),
        "dora_scale": np.array([1.0], dtype=np.float32),
    }
    with (d / "adapter_1.bin").open("wb") as f:
        pickle.dump(a1, f)

    # Adapter 2
    a2 = {
        "blk.0.attn_k.weight.lora_a": np.ones((16, 128), dtype=np.float32) * 2.0,
        "blk.0.attn_k.weight.lora_b": np.ones((128, 16), dtype=np.float32) * 2.0,
        "dora_scale": np.array([2.0], dtype=np.float32),
    }
    with (d / "adapter_2.bin").open("wb") as f:
        pickle.dump(a2, f)

    # Adapter Mismatched Shape
    a3 = {
        "blk.0.attn_k.weight.lora_a": np.ones(
            (32, 128), dtype=np.float32
        ),  # Wrong rank
        "blk.0.attn_k.weight.lora_b": np.ones((128, 32), dtype=np.float32),
        "dora_scale": np.array([1.0], dtype=np.float32),
    }
    with (d / "adapter_mismatch_shape.bin").open("wb") as f:
        pickle.dump(a3, f)

    # Adapter Mismatched Keys
    a4 = {
        "blk.0.attn_k.weight.lora_a": np.ones((16, 128), dtype=np.float32),
    }
    with (d / "adapter_mismatch_keys.bin").open("wb") as f:
        pickle.dump(a4, f)

    return d


def test_moa_gater_interpolation(tmp_adapters_dir: Path) -> None:
    """Test 50/50 blending of two adapters."""
    adapters = [
        str(tmp_adapters_dir / "adapter_1.bin"),
        str(tmp_adapters_dir / "adapter_2.bin"),
    ]
    gating_vector = [0.5, 0.5]

    merged = MoAGater.merge_adapters(adapters, gating_vector)

    # Expected output: 0.5 * 1.0 + 0.5 * 2.0 = 1.5
    assert np.allclose(merged["blk.0.attn_k.weight.lora_a"], 1.5)
    assert np.allclose(merged["blk.0.attn_k.weight.lora_b"], 1.5)
    assert np.allclose(merged["dora_scale"], 1.5)


def test_moa_gater_sparse_routing(tmp_adapters_dir: Path) -> None:
    """Test sparse routing (only 1 active adapter)."""
    adapters = [
        str(tmp_adapters_dir / "adapter_1.bin"),
        str(tmp_adapters_dir / "adapter_2.bin"),
    ]
    gating_vector = [0.0, 1.0]

    merged = MoAGater.merge_adapters(adapters, gating_vector)

    # Expected output: adapter 2 exact values
    assert np.allclose(merged["blk.0.attn_k.weight.lora_a"], 2.0)
    assert np.allclose(merged["blk.0.attn_k.weight.lora_b"], 2.0)
    assert np.allclose(merged["dora_scale"], 2.0)


def test_moa_gater_shape_mismatch(tmp_adapters_dir: Path) -> None:
    """Test rejection if adapters have different architectural shapes."""
    adapters = [
        str(tmp_adapters_dir / "adapter_1.bin"),
        str(tmp_adapters_dir / "adapter_mismatch_shape.bin"),
    ]
    gating_vector = [0.5, 0.5]

    with pytest.raises(ValueError, match="Shape mismatch"):
        MoAGater.merge_adapters(adapters, gating_vector)


def test_moa_gater_length_mismatch(tmp_adapters_dir: Path) -> None:
    """Test rejection if gating vector length doesn't match adapters count."""
    adapters = [
        str(tmp_adapters_dir / "adapter_1.bin"),
        str(tmp_adapters_dir / "adapter_2.bin"),
    ]
    gating_vector = [0.5, 0.3, 0.2]  # 3 gates, 2 adapters

    with pytest.raises(ValueError, match="Length of gating_vector"):
        MoAGater.merge_adapters(adapters, gating_vector)


def test_moa_gater_keys_mismatch(tmp_adapters_dir: Path) -> None:
    """Test rejection if adapters have different keys."""
    adapters = [
        str(tmp_adapters_dir / "adapter_1.bin"),
        str(tmp_adapters_dir / "adapter_mismatch_keys.bin"),
    ]
    gating_vector = [0.5, 0.5]

    with pytest.raises(ValueError, match="Keys mismatch"):
        MoAGater.merge_adapters(adapters, gating_vector)


def test_moa_gater_metal_integration(tmp_adapters_dir: Path) -> None:
    """Verify Metal preload and swap integration in MoAGater."""
    gater = MoAGater()
    adapters = [
        str(tmp_adapters_dir / "adapter_1.bin"),
        str(tmp_adapters_dir / "adapter_2.bin"),
    ]
    gating_vector = [0.5, 0.5]

    # 1. Merge
    merged = gater.merge_adapters(adapters, gating_vector)

    # 2. Preload
    preload_ms = gater.preload_to_metal(merged)
    assert preload_ms >= 0
    assert gater._last_preloaded_buffer is not None

    # 3. Swap (Strictly required)
    swap_ms = gater.swap_active_adapter()
    assert swap_ms >= 0


def test_moa_merge_and_swap_pipeline(tmp_adapters_dir: Path) -> None:
    """Verify the full merge_and_swap pipeline."""
    gater = MoAGater(enable_metal=True)

    adapters = [
        str(tmp_adapters_dir / "adapter_1.bin"),
        str(tmp_adapters_dir / "adapter_2.bin"),
    ]
    gating_vector = [0.7, 0.3]

    timings = gater.merge_and_swap(adapters, gating_vector)

    assert "merge_ms" in timings
    assert "preload_ms" in timings
    assert "swap_ms" in timings
    assert "total_ms" in timings
    assert timings["merge_ms"] > 0
    assert timings["preload_ms"] > 0
    assert timings["swap_ms"] >= 0
