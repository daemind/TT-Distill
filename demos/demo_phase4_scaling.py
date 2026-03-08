# ruff: noqa
"""
demo_phase4_scaling.py
Demonstration of TT-Distill AGI Scaling Architecture (Phase 4).
Focuses on Latent Topology Filtering and Continuous MoA Gating.
"""

import pickle
from pathlib import Path

import numpy as np

from src.orchestration.maca import LatentTrajectory, SinkhornBarycenter
from src.orchestration.moa_gating import MoAGater


def run_topology_demo():  # type: ignore[no-untyped-def]

    # Simulate a deep continuous thought trajectory (e.g. solving ARC-AGI 673ef223)
    traj_len, h_dim = 15, 768
    np.random.seed(42)

    # 3 core agents (tight arithmetic cluster representing logic/physics invariant)
    base_logic = np.random.randn(traj_len, h_dim)
    t1 = base_logic + np.random.randn(traj_len, h_dim) * 0.05
    t2 = base_logic + np.random.randn(traj_len, h_dim) * 0.05
    t3 = base_logic + np.random.randn(traj_len, h_dim) * 0.05

    # 1 severely hallucinating agent (disconnected from reality filter)
    t4_hallucinator = np.random.randn(traj_len, h_dim) * 5.0

    trajectories = [
        LatentTrajectory(agent_id="agent_1_logic", hidden_states=t1, confidence=1.0),
        LatentTrajectory(agent_id="agent_2_physics", hidden_states=t2, confidence=1.0),
        LatentTrajectory(agent_id="agent_3_geometry", hidden_states=t3, confidence=1.0),
        LatentTrajectory(
            agent_id="agent_4_hallucinator",
            hidden_states=t4_hallucinator,
            confidence=1.0,
        ),
    ]

    # Compute the robust barycenter (filtering outliers)
    barycenter_engine = SinkhornBarycenter()
    barycenter_engine.compute_barycenter(trajectories, rejection_threshold=0.85)

    rejected_agents = [
        t.agent_id for t in trajectories if t.metadata.get("rejected", False)
    ]

    assert "agent_4_hallucinator" in rejected_agents, (
        "ERROR: Filter failed to catch hallucinator."
    )
    assert "agent_1_logic" not in rejected_agents, "ERROR: Filter rejected valid agent."


def run_gating_demo():  # type: ignore[no-untyped-def]

    tmp_dir = Path("demo_adapters")
    tmp_dir.mkdir(exist_ok=True)

    # Generate 2 simulated DoRA adapters mathematically equivalent to .bin models
    # DoRA involves LoRA weights and a scaling magnitude vector
    a1 = {
        "lora.weight": np.random.randn(128, 128).astype(np.float32),
        "dora_scale": np.ones((128, 1)).astype(np.float32) * 1.5,
    }
    a2 = {
        "lora.weight": np.random.randn(128, 128).astype(np.float32),
        "dora_scale": np.ones((128, 1)).astype(np.float32) * 0.5,
    }

    p1 = tmp_dir / "physics_invariant_v1.bin"
    p2 = tmp_dir / "logic_invariant_v1.bin"

    with open(p1, "wb") as f:
        pickle.dump(a1, f)
    with open(p2, "wb") as f:
        pickle.dump(a2, f)

    # The System 2 Router issues a sparse distribution array for the current task
    gating_vector = [0.8, 0.2]

    # Fuse them dynamically
    MoAGater.merge_adapters([str(p1), str(p2)], gating_vector)

    # Cleanup
    p1.unlink()
    p2.unlink()
    tmp_dir.rmdir()


if __name__ == "__main__":
    try:
        run_topology_demo()  # type: ignore[no-untyped-call]
        run_gating_demo()  # type: ignore[no-untyped-call]
    except Exception:
        pass
