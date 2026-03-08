"""Hardened tests for SmartRouter: Complexity and Routing. Adapted from Core."""

import sys
from typing import Any, ClassVar, Generator, cast
from unittest.mock import MagicMock

import pytest


# Mocking missing core dependencies to allow standalone testing in TT-Distill context
# This addresses the 'no module named src.db_manager' issues.
class MockDB:
    def _get_connection(self) -> None:
        pass


class MockHardware:
    total_ram_gb: float = 32.0
    cpu_cores: int = 16
    gpu_info: ClassVar[list[dict[str, Any]]] = [
        {"name": "M2 Ultra", "vram_mb": 128 * 1024}
    ]
    system_info: ClassVar[dict[str, str]] = {"platform": "darwin"}


# Inject mocks into sys.modules before importing SmartRouter
sys.modules["src.db_manager"] = MagicMock()
sys.modules["src.hardware_discovery"] = MagicMock()
sys.modules["src.model_registry"] = MagicMock()

from src.orchestration.router import (  # noqa: E402
    ExecutionPlan,
    SmartRouter,
    TaskComplexityMetrics,
)


@pytest.fixture
def smart_router() -> Generator[SmartRouter, None, None]:
    """Create a SmartRouter instance with mocked hardware discovery."""
    registry = MagicMock()
    # Mocking select_model_for_task dependencies
    router = SmartRouter(model_registry=registry)
    router._hardware_info = cast(Any, MockHardware())
    yield router


def test_complexity_score_rigor() -> None:
    """Verify complexity math precisely."""
    metrics = TaskComplexityMetrics(
        scope_factor=1.0,
        dependency_depth=2,
        resource_intensity=1.0,
        file_count=5,
        lines_changed=100,
    )
    # Score = 1.0*1.0 + 2*0.5 + 1.0*2.0 + 5*0.1 + 100*0.01
    # = 1.0 + 1.0 + 2.0 + 0.5 + 1.0 = 5.5
    assert metrics.complexity_score == 5.5


@pytest.mark.asyncio
async def test_routing_logic_standard(smart_router: SmartRouter) -> None:
    """Verify model selection logic for a standard task."""
    complexity = await smart_router.calculate_complexity(
        task_description="Simple bug fix",
        feature_count=1,
        dependency_count=1,
        file_count=1,
        estimated_lines=50,
    )
    model_id, quantization = await smart_router.select_model_for_task(complexity)

    assert "8b" in model_id.lower()
    assert quantization == "Q4_K_M"


@pytest.mark.asyncio
async def test_execution_mode_bottleneck(smart_router: SmartRouter) -> None:
    """Verify that high complexity forces queue execution."""
    metrics = TaskComplexityMetrics(
        scope_factor=4.0,
        dependency_depth=10,  # Bottleneck
        resource_intensity=2.0,
        file_count=10,
        lines_changed=500,
    )
    model_id = "meta-llama/Llama-3-8b-chat-hf"

    mode = await smart_router.determine_execution_mode(metrics, model_id)
    assert mode == "queue"


@pytest.mark.asyncio
async def test_generate_execution_plan_e2e(smart_router: SmartRouter) -> None:
    """Verify that a full execution plan is generated with realistic metadata."""
    plan = await smart_router.generate_execution_plan(
        task_description="Refactor MACA engine",
        feature_count=8,
        dependency_count=4,
        file_count=5,
        estimated_lines=300,
        skill_type="reasoning",
    )

    assert isinstance(plan, ExecutionPlan)
    assert plan.estimated_vram_gb > 0
    assert plan.estimated_time_seconds > 0
    assert "Complexity" in plan.reason
