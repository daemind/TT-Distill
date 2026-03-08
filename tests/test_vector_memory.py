"""Hardened tests for VectorMemory: Persistence, Search, and Topic Clustering."""

import hashlib
from typing import AsyncGenerator
from unittest.mock import patch

import pytest
from anyio import Path

from src.vector_memory import VectorMemory


def stable_embed(text: str, dim: int = 128) -> list[float]:
    """Generates a deterministic embedding with topic separation for testing."""
    result = [0.0] * dim
    text_lower = text.lower()
    # Topic 1: Physics/Sim/Robotics
    if "mujoco" in text_lower or "physics" in text_lower:
        result[0] = 1.0
        result[2] = 0.8  # Related to physics
    elif "tt-distill" in text_lower or "swap" in text_lower:
        result[0] = 0.3  # General robotics
        result[3] = 1.0  # Swap specific
    elif any(w in text_lower for w in ["robot", "control", "engine"]):
        result[0] = 0.6

    # Topic 2: Culinary/Nature/Food
    elif "apple" in text_lower or "fruit" in text_lower:
        result[1] = 1.0
        result[4] = 0.9  # Fruit specific
    elif any(w in text_lower for w in ["bread", "sweet", "juicy", "yeast"]):
        result[1] = 0.8
    else:
        # Default fallback to hash
        h = hashlib.sha256(text.encode()).digest()
        base = [float(b) / 255.0 for b in h]
        result = (base * (dim // 32 + 1))[:dim]
    return result


@pytest.fixture
async def vmem_temp(tmp_path: Path) -> AsyncGenerator[VectorMemory, None]:
    """Fixture to provide an isolated VectorMemory instance."""
    db_path = Path(tmp_path) / "test_vmem_exhaustive.db"
    if await db_path.exists():
        await db_path.unlink()

    config = VectorMemory.Config(db_path=str(db_path), embedding_dim=128)
    vm = VectorMemory(config)
    # Mock embedding to avoid fallback to TF-IDF
    with patch.object(
        vm, "_compute_embedding", side_effect=lambda text: stable_embed(text, 128)
    ):
        yield vm
    await vm.close()
    if await db_path.exists():
        await db_path.unlink()


@pytest.mark.asyncio
async def test_vector_memory_topic_retrieval(vmem_temp: VectorMemory) -> None:
    """Verify that retrieval accurately distinguishes between different topics."""
    vmem = vmem_temp

    # Topic 1: Physics/Robotics
    await vmem.add_document_async(
        1, "MuJoCo is a physics engine for model-based control."
    )
    await vmem.add_document_async(
        2, "TT-Distill uses O(1) context swapping for low latency."
    )

    # Topic 2: Culinary/Nature
    await vmem.add_document_async(3, "The red apple is sweet and juicy.")
    await vmem.add_document_async(4, "Baking bread requires yeast and warm water.")

    # Search for robotics
    if not vmem.use_sqlite_vec:
        pytest.skip("sqlite-vec extension not loaded, skipping semantic search tests")

    results_robot = await vmem.search_async("robotic control engine")
    assert results_robot[0][0]["id"] == 1

    # Search for food
    results_food = await vmem.search_async("sweet fruit")
    assert results_food[0][0]["id"] == 3


@pytest.mark.asyncio
async def test_vector_memory_persistence(tmp_path: Path) -> None:
    """Verify that documents persist after closing and reopening the DB."""
    db_path = Path(tmp_path) / "test_vmem_persistence.db"

    config = VectorMemory.Config(db_path=str(db_path), embedding_dim=128)

    # Session 1: Persist
    vm1 = VectorMemory(config)
    with patch.object(
        vm1, "_compute_embedding", side_effect=lambda text: stable_embed(text, 128)
    ):
        await vm1.add_document_async(100, "Persistent Memory Test")
    await vm1.close()

    # Session 2: Retrieve
    vm2 = VectorMemory(config)
    if not vm2.use_sqlite_vec:
        await vm2.close()
        pytest.skip("sqlite-vec extension not loaded, skipping persistence search test")

    with patch.object(
        vm2, "_compute_embedding", side_effect=lambda text: stable_embed(text, 128)
    ):
        results = await vm2.search_async("Memory Test")
        assert len(results) == 1
        assert results[0][0]["id"] == 100
    await vm2.close()

    if await db_path.exists():
        await db_path.unlink()


@pytest.mark.asyncio
async def test_vector_memory_empty_query(vmem_temp: VectorMemory) -> None:
    """Verify that empty queries don't crash the system."""
    if not vmem_temp.use_sqlite_vec:
        pytest.skip("sqlite-vec extension not loaded")

    # The fixture already mocks _compute_embedding
    results = await vmem_temp.search_async("")
    assert isinstance(results, list)
