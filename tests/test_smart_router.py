"""Exhaustive tests for SmartRouter: Complexity Scoring and Routing Accuracy."""

import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Provide stubs for missing dependencies
sys.modules["src.db_manager"] = MagicMock()
sys.modules["src.model_registry"] = MagicMock()
mock_hw = MagicMock()
mock_hw.discover_hardware = AsyncMock(return_value=MagicMock(total_ram_gb=32.0))
sys.modules["src.hardware_discovery"] = mock_hw

from src.orchestration.router import SmartRouter  # noqa: E402


@pytest.mark.asyncio
async def test_smart_router_model_selection() -> None:
    """Verify that SmartRouter selects the correct model based on complexity and RAM."""
    # We use patch to ensure discover_hardware is awaitable
    with patch(
        "src.orchestration.router.discover_hardware", new_callable=AsyncMock
    ) as mock_disc:
        mock_disc.return_value = MagicMock(total_ram_gb=32.0)
        router = SmartRouter()

        # Simple task -> Small model
        simple_metrics = await router.calculate_complexity("test", 1, 0, 1, 10)
        model_id, _ = await router.select_model_for_task(simple_metrics)
        assert "8b" in model_id.lower()

        # Complex task -> Large model (since RAM is 32GB)
        complex_metrics = await router.calculate_complexity("complex", 15, 10, 20, 1000)
        model_id, _ = await router.select_model_for_task(complex_metrics)
        assert "70b" in model_id.lower()


@pytest.mark.asyncio
async def test_smart_router_low_ram_adaptation() -> None:
    """Verify that SmartRouter adapts to low RAM by selecting smaller models."""
    with patch(
        "src.orchestration.router.discover_hardware", new_callable=AsyncMock
    ) as mock_disc:
        mock_disc.return_value = MagicMock(total_ram_gb=8.0)  # Low RAM
        router = SmartRouter()

        # Even complex task should use small model on low RAM
        complex_metrics = await router.calculate_complexity("complex", 15, 10, 20, 1000)
        model_id, _ = await router.select_model_for_task(complex_metrics)
        assert "8b" in model_id.lower()
