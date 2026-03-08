import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

# Mocking external heavyweight dependencies not present in TT-Distill source
# This allows the integration logic itself to be rigorously tested.
sys.modules["reflex_engine"] = MagicMock()
sys.modules["src.orchestration.post_silicon"] = MagicMock()
sys.modules["src.orchestration.maca_salon_bridge"] = MagicMock()

from typing import Any, Generator, cast  # noqa: E402

from src.orchestration.tt_distill_integration import (  # noqa: E402
    TTDistillConfig,
    TTDistillIntegration,
)


@pytest.fixture
def mock_integration(tmp_path: Path) -> Generator[TTDistillIntegration, None, None]:
    """Fixture for TTDistillIntegration with mocked external agents."""
    model_file = tmp_path / "mock_s1.gguf"
    # Create a physical mock file to satisfy Path(model_path).exists()
    model_file.write_text("mock")

    config = TTDistillConfig(
        model_path=str(model_file),
        lora_path=None,
        vector_db_path=str(tmp_path / "test_integration_vmem.db"),
    )

    integration = TTDistillIntegration(config)

    # Mock specific methods that would call missing external logic
    setattr(
        integration.reflex_engine,
        "query_reflex",
        MagicMock(return_value=(10.0, "Mocked Response")),
    )
    setattr(integration.post_silicon, "run_optimization_loop", AsyncMock())

    yield integration

    if model_file.exists():
        model_file.unlink()


@pytest.mark.asyncio
async def test_full_pipeline_flow(mock_integration: TTDistillIntegration) -> None:
    """Verify the E2E flow: Consensus -> Distillation -> Reflex -> Post-Silicon."""
    it = mock_integration
    await it.initialize()

    # 1. Consensus
    intents = ["Balance on ice", "Avoid collision"]
    consensus = await it.run_consensus(intents)
    assert len(consensus.participating_agents) == 2

    # 2. Distillation
    dora_adapter = it.distill_to_dora(consensus)
    assert dora_adapter.shape[1] == it.config.vector_embedding_dim

    # 3. Reflex Execution (with RAG)
    latency, response = await it.execute_reflex("How to stabilize?", use_rag=True)
    assert latency > 0
    assert "Mocked" in response
    assert len(it._reflex_metrics) == 1

    # 4. Metrics & Shutdown
    metrics = it.get_reflex_metrics()
    assert metrics["count"] == 1
    assert "mean_latency_ms" in metrics

    await it.shutdown()


@pytest.mark.asyncio
async def test_multimodal_unsupported(mock_integration: TTDistillIntegration) -> None:
    """Verify that multimodal calls fallback gracefully when files are missing."""
    it = mock_integration
    # Should fallback to text query if image path doesn't exist
    _latency, response = await it.execute_multimodal_reflex(
        "See this", image_path="non_existent.jpg"
    )
    assert "Mocked" in response
    # Cast to Any to satisfy mypy for assert_called
    cast(Any, it.reflex_engine.query_reflex).assert_called()


def test_integration_config_immutability() -> None:
    """Verify that config defaults are respected."""
    config = TTDistillConfig()
    assert config.target_latency_ms == 12.0
    assert config.maca_rank == 16
