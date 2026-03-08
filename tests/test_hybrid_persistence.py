from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.persistence.hybrid_persistence import HybridPersistence


@pytest.fixture
def mock_postgres():
    with patch("src.persistence.hybrid_persistence.PostgresPersistence") as mock:
        instance = mock.return_value
        instance.initialize = AsyncMock()
        instance.close = AsyncMock()
        instance.session_repo = MagicMock()
        instance.vector_repo = MagicMock()
        yield instance

@pytest.fixture
def mock_sqlite():
    with patch("src.persistence.hybrid_persistence.SQLitePersistence") as mock:
        instance = mock.return_value
        instance.add_memory = AsyncMock(return_value=123)
        yield instance

@pytest.fixture
def mock_pipeline():
    with patch("src.persistence.hybrid_persistence.VectorSyncPipeline") as mock:
        instance = mock.return_value
        instance.start = AsyncMock()
        instance.stop = AsyncMock()
        instance.queue_sync = AsyncMock()
        yield instance

@pytest.mark.anyio
async def test_hybrid_persistence_init_close(mock_postgres, mock_sqlite, mock_pipeline):
    hp = HybridPersistence()
    await hp.initialize()
    assert hp._initialized is True
    mock_postgres.initialize.assert_awaited_once()
    mock_pipeline.start.assert_awaited_once()

    await hp.close()
    assert hp._initialized is False
    mock_pipeline.stop.assert_awaited_once()
    mock_postgres.close.assert_awaited_once()

@pytest.mark.anyio
async def test_hybrid_persistence_repos(mock_postgres, mock_sqlite, mock_pipeline):
    hp = HybridPersistence()
    assert hp.session_repo == mock_postgres.session_repo
    assert hp.memory_repo == mock_sqlite
    assert hp.vector_repo == mock_postgres.vector_repo

@pytest.mark.anyio
async def test_hybrid_persistence_add_memory_sync(mock_postgres, mock_sqlite, mock_pipeline):
    hp = HybridPersistence()

    # Test with embedding (triggers sync)
    mid = await hp.add_memory_with_sync(1, "text", embedding=[0.1])
    assert mid == 123
    mock_sqlite.add_memory.assert_awaited_once_with(
        session_id=1, text="text", metadata=None, embedding=[0.1], chunk_key=None
    )
    mock_pipeline.queue_sync.assert_awaited_once_with(
        session_id=1, text="text", metadata={}, embedding=[0.1], chunk_key=None
    )

    mock_pipeline.queue_sync.reset_mock()
    # Test without embedding (no sync)
    await hp.add_memory_with_sync(1, "text2")
    mock_pipeline.queue_sync.assert_not_called()
