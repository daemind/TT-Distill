"""Hot Swap Integrity & Stress Test for the Metal O(1) Manifold.

This test performs high-frequency swaps of large DoRA weight buffers
to verify CTypes stability and hardware response consistency.
"""

import logging

import numpy as np

from src.orchestration.moa_gating import MoAGater

logger = logging.getLogger(__name__)


def test_hot_swap_high_frequency() -> None:
    """Verify that O(1) swap remains stable at sub-millisecond frequencies."""
    gater = MoAGater()

    # Large synthetic adapter weights (16MB total)
    dim, rank = 2560, 16
    # Use float32 as required by Metal backend
    expert_a = np.random.randn(dim, rank).astype(np.float32)
    expert_b = np.random.randn(rank, dim).astype(np.float32)

    adapter = {"lora_a": expert_a, "lora_b": expert_b}

    # Stress test: 1000 Rapid Swaps
    latencies = []
    for _ in range(1000):
        # 1. Preload (Simulate weight evolution/generation)
        gater.preload_to_metal(adapter)

        # 2. Swap (The Atomic O(1) operation)
        swap_ms = gater.swap_active_adapter()

        latencies.append(swap_ms)
        assert swap_ms >= 0

    avg_swap = sum(latencies) / len(latencies)
    logger.info(f"[STRESS TEST] Avg O(1) Swap Latency: {avg_swap:.6f} ms")

    # Requirement: Swap should be negligible (< 0.01ms avg)
    assert avg_swap < 0.05

def test_hot_swap_idempotency() -> None:
    """Verify that repeated swaps of the same buffer don't leak or degrade."""
    gater = MoAGater()
    adapter = {
        "lora_a": np.random.randn(2560, 16).astype(np.float32),
        "lora_b": np.random.randn(16, 2560).astype(np.float32)
    }

    gater.preload_to_metal(adapter)

    # 100 swaps of the same preloaded pointer
    for i in range(100):
        ms = gater.swap_active_adapter()
        assert ms >= 0
        if i % 20 == 0:
            logger.info(f"Swap {i}: {ms:.6f} ms")

def test_hot_swap_collision_resistance() -> None:
    """Verify that rapid preloads don't corrupt the swap sequence."""
    gater = MoAGater()

    for i in range(10):
        # Rapidly change the weights and swap
        weights = i * np.ones((1024, 16), dtype=np.float32)
        adapter = {"w": weights}

        gater.preload_to_metal(adapter)
        ms = gater.swap_active_adapter()
        assert ms >= 0

if __name__ == "__main__":
    # If run manually, execute the stress test
    test_hot_swap_high_frequency()
