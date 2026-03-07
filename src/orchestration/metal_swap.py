"""TT-Distill Metal O(1) DoRA Hot-Swap Interface.

This module provides a Python CTypes bridge to the custom `ggml_backend_metal_swap_dora`
function injected into Apple's Metal backend of llama.cpp.

The function performs an O(1) pointer swap between `active_dora_buffer` and
`preload_dora_buffer` inside the `ggml_metal_context` struct, bypassing the
~215ms Metal graph re-creation penalty entirely.

Architecture:
    Python (CTypes) -> libggml-metal.dylib -> ggml_backend_metal_swap_dora()
                                           -> ggml_metal_swap_dora()
                                           -> pointer swap (< 0.1ms)
"""

from __future__ import annotations

import ctypes
import logging
import time
from pathlib import Path

logger = logging.getLogger(__name__)

# Default path to the custom-compiled libggml-metal.dylib
_DEFAULT_DYLIB_PATH = (
    Path(__file__).resolve().parents[2]
    / "llama.cpp"
    / "build"
    / "src"
    / "ggml-metal"
    / "libggml-metal.dylib"
)


class MetalDoRASwapper:
    """O(1) DoRA adapter hot-swap via the custom Metal backend.

    This class loads `libggml-metal.dylib` at construction time and
    exposes `swap()` which calls `ggml_backend_metal_swap_dora` on
    a given backend pointer.

    For standalone benchmarking (without a live llama.cpp session), use
    `benchmark_swap_overhead()` which measures the raw CTypes call latency.
    """

    def __init__(self, dylib_path: Path | str | None = None) -> None:
        """Load the custom Metal dynamic library.

        Args:
            dylib_path: Absolute path to `libggml-metal.dylib`.
                       Defaults to `llama.cpp/build/src/ggml-metal/libggml-metal.dylib`.
                       Can also be set via the `GGML_METAL_DYLIB` env variable.
        """
        # Strategy 1: Explicit argument
        import os  # noqa: PLC0415

        # Strategy 1: Explicit argument
        # Strategy 2: Environment variable
        # Strategy 3: Default project path
        candidates = []
        if dylib_path:
            candidates.append(Path(dylib_path))

        env_path = os.environ.get("GGML_METAL_DYLIB")
        if env_path:
            candidates.append(Path(env_path))

        candidates.append(_DEFAULT_DYLIB_PATH)

        # Try loading directly via ctypes (bypasses os.stat sandbox issues)
        self._lib = None
        loaded_path = None
        for candidate in candidates:
            try:
                self._lib = ctypes.cdll.LoadLibrary(str(candidate))
                loaded_path = candidate
                break
            except OSError:
                continue

        if self._lib is None:
            msg = (
                "Could not load libggml-metal.dylib from any of: "
                + ", ".join(str(c) for c in candidates)
                + "\nPlease compile llama.cpp with the TT-Distill patches first:\n"
                "  cd llama.cpp && cmake -S ggml -B build -DGGML_METAL=ON "
                "-DBUILD_SHARED_LIBS=ON && cmake --build build -j\n"
                "Or set GGML_METAL_DYLIB=/path/to/libggml-metal.dylib"
            )
            raise FileNotFoundError(msg)

        self.dylib_path = loaded_path or _DEFAULT_DYLIB_PATH

        # Bind ggml_backend_metal_swap_dora(ggml_backend_t backend)
        # ggml_backend_t is an opaque pointer (void*)
        self._swap_fn = self._lib.ggml_backend_metal_swap_dora
        self._swap_fn.argtypes = [ctypes.c_void_p]
        self._swap_fn.restype = None

        # Also bind the internal ggml_metal_swap_dora(ggml_metal_t ctx)
        # This one is NULL-safe (checks ctx != NULL before proceeding)
        # Used for standalone benchmarking without a live Metal backend.
        self._swap_internal_fn = self._lib.ggml_metal_swap_dora
        self._swap_internal_fn.argtypes = [ctypes.c_void_p]
        self._swap_internal_fn.restype = None

        logger.info(
            "MetalDoRASwapper: loaded from %s",
            self.dylib_path,
        )

    def swap(self, backend_ptr: int) -> float:
        """Execute the O(1) DoRA pointer swap.

        Args:
            backend_ptr: The raw `ggml_backend_t` pointer (integer address).

        Returns:
            Elapsed time in milliseconds.
        """
        t0 = time.perf_counter_ns()
        self._swap_fn(ctypes.c_void_p(backend_ptr))
        elapsed_ns = time.perf_counter_ns() - t0
        elapsed_ms = elapsed_ns / 1_000_000
        logger.info("DoRA swap executed in %.4f ms", elapsed_ms)
        return elapsed_ms

    def benchmark_swap_overhead(self, iterations: int = 1000) -> dict[str, float]:
        """Benchmark the raw CTypes call overhead without a live backend.

        This calls the *internal* ggml_metal_swap_dora(NULL) which safely
        returns immediately (the function checks `ctx != NULL`).
        The public ggml_backend_metal_swap_dora() cannot be used here
        because it has a fatal GGML_ASSERT on NULL.

        Args:
            iterations: Number of swap calls to average over.

        Returns:
            Dictionary with min, max, mean, and median latencies in ms.
        """
        latencies: list[float] = []
        for _ in range(iterations):
            t0 = time.perf_counter_ns()
            self._swap_internal_fn(ctypes.c_void_p(0))  # NULL -> safe no-op
            elapsed_ns = time.perf_counter_ns() - t0
            latencies.append(elapsed_ns / 1_000_000)

        latencies.sort()
        n = len(latencies)
        median = latencies[n // 2] if n % 2 else (latencies[n // 2 - 1] + latencies[n // 2]) / 2

        return {
            "iterations": iterations,
            "min_ms": latencies[0],
            "max_ms": latencies[-1],
            "mean_ms": sum(latencies) / n,
            "median_ms": median,
            "p99_ms": latencies[int(n * 0.99)],
        }
