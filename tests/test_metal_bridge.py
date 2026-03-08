import ctypes
import os
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

from src.orchestration.metal_swap import MetalDoRASwapper


def test_metal_swapper_initialization() -> None:
    """Test if the swapper can be initialized or fails with a proper error."""
    try:
        swapper = MetalDoRASwapper()
        assert swapper.dylib_path.exists()
    except FileNotFoundError as e:
        assert "Could not load libggml-metal.dylib" in str(e)
        assert "cmake" in str(e)
    except Exception as e:
        pytest.fail(f"Unexpected error during initialization: {e}")


def test_metal_swapper_candidates() -> None:
    """Test that the swapper checks the correct candidate paths."""
    from src.orchestration.metal_swap import _DEFAULT_DYLIB_PATH

    assert isinstance(_DEFAULT_DYLIB_PATH, Path)
    assert _DEFAULT_DYLIB_PATH.suffix == ".dylib"


def test_metal_preload_and_swap_logic() -> None:
    """Verify the Preload and Swap sequence logic."""
    swapper: MetalDoRASwapper | None = None
    try:
        swapper = MetalDoRASwapper()
    except FileNotFoundError:
        pytest.skip("Metal dylib not found, skipping hardware-native logic test.")

    if swapper is not None:
        # Create dummy data for preload
        data = np.random.randn(100).astype(np.float32)
        ptr = data.ctypes.data_as(ctypes.c_void_p)

        # Test Preload (standalone)
        try:
            swapper.preload(ptr.value if ptr.value else 0, data.nbytes)
        except Exception as e:
            pytest.fail(f"Preload failed: {e}")

        # Test Swap (standalone)
        try:
            latency = swapper.benchmark_swap_overhead(iterations=10)
            assert "median_ms" in latency
            assert latency["median_ms"] >= 0
        except Exception as e:
            pytest.fail(f"Swap benchmark failed: {e}")


def test_metal_dylib_path_env_override(tmp_path: Path) -> None:
    """Test that GGML_METAL_DYLIB environment variable is respected."""
    original_env = os.environ.get("GGML_METAL_DYLIB")
    fake_path = str(tmp_path / "non_existent_lib.dylib")
    os.environ["GGML_METAL_DYLIB"] = fake_path

    with mock.patch(
        "src.orchestration.metal_swap._DEFAULT_DYLIB_PATH", tmp_path / "invalid_default"
    ):
        try:
            with pytest.raises(FileNotFoundError) as excinfo:
                MetalDoRASwapper()
            assert fake_path in str(excinfo.value)
        finally:
            if original_env:
                os.environ["GGML_METAL_DYLIB"] = original_env
            else:
                del os.environ["GGML_METAL_DYLIB"]
