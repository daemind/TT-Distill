import asyncio
from typing import Generator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.persistence.sync_pipeline import SyncTask, VectorSyncPipeline


@pytest.fixture
def mock_postgres_for_pipeline() -> Generator[MagicMock, None, None]:
    with patch("src.persistence.sync_pipeline.PostgresPersistence") as mock:
        instance = mock.return_value
        instance.initialize = AsyncMock()
        instance.close = AsyncMock()
        instance.add_vector = AsyncMock()
        yield instance

@pytest.mark.anyio
async def test_sync_pipeline_start_stop(mock_postgres_for_pipeline: MagicMock) -> None:
    pipeline = VectorSyncPipeline(worker_count=1)
    await pipeline.start()
    assert pipeline._running is True
    assert len(pipeline._workers) == 1

    await pipeline.stop()
    assert pipeline._running is False
    # Use Any cast or ignore to satisfy mypy's unreachable analysis
    assert len(pipeline._workers) == 0  # type: ignore[unreachable]
    mock_postgres_for_pipeline.close.assert_awaited_once()

@pytest.mark.anyio
async def test_sync_pipeline_process_task(mock_postgres_for_pipeline: MagicMock) -> None:
    pipeline = VectorSyncPipeline(worker_count=1)
    await pipeline.start()

    await pipeline.queue_sync(1, "text", {"m": 1}, [0.1], "key")

    # Wait for queue to be processed
    await pipeline.sync_queue.join()

    mock_postgres_for_pipeline.add_vector.assert_awaited_once_with(
        session_id=1, text="text", metadata={"m": 1}, embedding=[0.1], chunk_key="key"
    )

    await pipeline.stop()

@pytest.mark.anyio
async def test_sync_pipeline_worker_timeout(mock_postgres_for_pipeline: MagicMock) -> None:
    # Test TimeoutError in worker loop
    pipeline = VectorSyncPipeline(worker_count=1)
    await pipeline.start()

    # We just need to let the worker wait and timeout once
    await asyncio.sleep(1.2) # Longer than 1.0 timeout

    await pipeline.stop()

@pytest.mark.anyio
async def test_sync_pipeline_worker_error(mock_postgres_for_pipeline: MagicMock) -> None:
    pipeline = VectorSyncPipeline(worker_count=1)
    mock_postgres_for_pipeline.add_vector.side_effect = Exception("Sync failed")

    await pipeline.start()
    await pipeline.queue_sync(1, "text", {}, [0.1])

    await pipeline.sync_queue.join()
    # The error should be logged but worker continues

    await pipeline.stop()

@pytest.mark.anyio
async def test_sync_pipeline_generate_key() -> None:
    pipeline = VectorSyncPipeline()
    key = await pipeline._generate_chunk_key("file", 0, 100)
    assert len(key) == 16

@pytest.mark.anyio
async def test_sync_pipeline_redundant_start(mock_postgres_for_pipeline: MagicMock) -> None:
    pipeline = VectorSyncPipeline(worker_count=1)
    await pipeline.start()
    await pipeline.start() # Should return immediately (cov line 66)
    assert len(pipeline._workers) == 1
    await pipeline.stop()

@pytest.mark.anyio
async def test_sync_pipeline_worker_process_error(mock_postgres_for_pipeline: MagicMock) -> None:
    pipeline = VectorSyncPipeline(worker_count=1)
    await pipeline.start()

    # Mock _process_task to raise directly to hit line 141-143
    with patch.object(pipeline, "_process_task", side_effect=Exception("Major error")):
        await pipeline.queue_sync(1, "text", {}, [0.1])
        # We need to wait a bit for the worker to catch it
        await asyncio.sleep(0.5)

    await pipeline.stop()

@pytest.mark.anyio
async def test_sync_pipeline_uninitialized_postgres(mock_postgres_for_pipeline: MagicMock) -> None:
    pipeline = VectorSyncPipeline()
    # Don't call start()
    await pipeline._process_task(SyncTask(1, "t", {}, [0.1]))
    # Should log error and return
