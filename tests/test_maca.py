"""Hardened tests for MACA (Multi-Agent Consensus Alignment). Ported from Core."""

import time
from typing import Any, Dict, List

import numpy as np
import pytest

from src.orchestration.maca import (
    ConsensusResult,
    LatentTrajectory,
    MACAEngine,
    SinkhornBarycenter,
    TTDistillBridge,
)


@pytest.fixture
def latent_dim() -> int:
    return 512


@pytest.fixture
def seq_len() -> int:
    return 32


def test_latent_trajectory_properties(latent_dim: int, seq_len: int) -> None:
    """Verify that trajectories maintain structural integrity."""
    traj = np.random.randn(seq_len, latent_dim).astype(np.float32)
    lt = LatentTrajectory(agent_id="A", hidden_states=traj, confidence=0.9)
    assert lt.hidden_states.shape == (seq_len, latent_dim)
    assert lt.hidden_states.dtype == np.float32


def test_sinkhorn_barycenter_convergence(latent_dim: int, seq_len: int) -> None:
    """Verify that Sinkhorn barycenter converges to a valid consensus."""
    trajectories = []
    for i in range(4):
        np.random.seed(i)
        traj = np.random.randn(seq_len, latent_dim).astype(np.float32)
        trajectories.append(LatentTrajectory(f"agent_{i}", traj))

    barycenter = SinkhornBarycenter.compute_barycenter(trajectories)
    assert barycenter.barycentre.shape == (seq_len, latent_dim)
    assert 0 <= barycenter.consensus_score <= 1.0
    # Scientific check: Score should be stable
    assert barycenter.consensus_score > 0.0


@pytest.mark.asyncio
async def test_maca_consensus_basic() -> None:
    """Verify basic consensus between two simple trajectories."""
    latent_dim = 128
    seq_len = 32
    engine = MACAEngine(latent_dim=latent_dim, seq_len=seq_len)

    intentions: List[Dict[str, Any]] = [
        {
            "agent_id": "agent_a",
            "initial_state": np.random.randn(latent_dim).astype(np.float32),
        },
        {
            "agent_id": "agent_b",
            "initial_state": np.random.randn(latent_dim).astype(np.float32),
        },
    ]

    result = await engine.run_consensus(intentions)
    assert result.barycentre.shape == (seq_len, latent_dim)
    assert result.consensus_score > 0


@pytest.mark.asyncio
async def test_maca_sinkhorn_convergence() -> None:
    """Verify that Sinkhorn barycenter converges for aligned trajectories."""
    latent_dim = 64
    seq_len = 16
    engine = MACAEngine(latent_dim=latent_dim, seq_len=seq_len)

    # Highly correlated trajectories
    base = np.random.randn(latent_dim).astype(np.float32)
    intentions: List[Dict[str, Any]] = [
        {
            "agent_id": "a1",
            "initial_state": base
            + np.random.randn(latent_dim).astype(np.float32) * 0.01,
        },
        {
            "agent_id": "a2",
            "initial_state": base
            + np.random.randn(latent_dim).astype(np.float32) * 0.01,
        },
    ]

    result = await engine.run_consensus(intentions)
    assert result.consensus_score > 0.95  # Alta convergence


@pytest.mark.asyncio
async def test_tt_distill_bridge_svd() -> None:
    """Verify that DoRA distillation produces correct adapter shapes."""
    latent_dim = 256
    seq_len = 32
    rank = 16

    bridge = TTDistillBridge(rank=rank, hidden_size=latent_dim)
    barycentre = np.random.randn(seq_len, latent_dim).astype(np.float32)

    adapter = bridge.distill_barycenter(
        barycentre, use_svd=True, preserve_temporal=True
    )

    # LoRA_A should be (rank, seq_len) if we preserve temporal
    assert adapter["lora_a"].shape[0] == rank
    assert adapter["lora_a"].shape[1] == seq_len

    # LoRA_B should be (latent_dim, rank)
    assert adapter["lora_b"].shape[0] == latent_dim
    assert adapter["lora_b"].shape[1] == rank


@pytest.mark.asyncio
async def test_maca_agent_scaling() -> None:
    """Verify engine performance/stability with a large number of agents."""
    latent_dim = 64
    seq_len = 16
    engine = MACAEngine(latent_dim=latent_dim, seq_len=seq_len)
    n_agents = 16

    agent_intentions: List[Dict[str, Any]] = [
        {
            "agent_id": f"agent_{i}",
            "initial_state": np.random.randn(latent_dim).astype(np.float32),
            "confidence": 1.0,
        }
        for i in range(n_agents)
    ]

    consensus = await engine.run_consensus(agent_intentions)
    assert len(consensus.participating_agents) == n_agents
    assert consensus.barycentre.shape == (seq_len, latent_dim)


@pytest.mark.asyncio
async def test_maca_engine_full_run(latent_dim: int, seq_len: int) -> None:
    """Verify full MACAEngine loop with multiple agent intents."""
    engine = MACAEngine(latent_dim=latent_dim, seq_len=seq_len)

    agent_intentions: List[Dict[str, Any]] = []
    for i in range(4):
        np.random.seed(i)
        initial_state = np.random.randn(latent_dim).astype(np.float32)
        agent_intentions.append(
            {
                "agent_id": f"agent_{i}",
                "initial_state": initial_state,
                "confidence": 1.0,
            }
        )

    start_time = time.perf_counter()
    consensus = await engine.run_consensus(agent_intentions)
    latency_ms = (time.perf_counter() - start_time) * 1000

    assert isinstance(consensus, ConsensusResult)
    assert consensus.barycentre.shape == (seq_len, latent_dim)
    assert len(consensus.participating_agents) == 4
    # Performance requirement: Consensus should be fast
    assert latency_ms < 500.0  # Strict target
