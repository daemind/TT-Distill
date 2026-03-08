import asyncio
import contextlib
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.persistence.cocoindex_ingestion import CocoIndexIngestion, FileChunk


@pytest.fixture
def mock_deps():
    emb = MagicMock()
    emb.compute_embedding.return_value = [0.1]*1024
    sync = MagicMock()
    sync.queue_sync = AsyncMock()
    return emb, sync

@pytest.mark.anyio
async def test_cocoindex_ingestion_chunking(mock_deps):
    emb, sync = mock_deps
    ci = CocoIndexIngestion(
        watch_paths=[Path("/temp_test_path")],
        embedding_service=emb,
        sync_pipeline=sync,
        chunk_size=10,
        chunk_overlap=2
    )

    text = "This is a long text for chunking"
    chunks = ci._chunk_text(text, "test.txt")
    assert len(chunks) > 1

    small_chunks = ci._chunk_text("short", "small.txt")
    assert len(small_chunks) == 1

@pytest.mark.anyio
async def test_cocoindex_ingestion_should_ingest(mock_deps):
    emb, sync = mock_deps
    ci = CocoIndexIngestion(
        watch_paths=[],
        embedding_service=emb,
        sync_pipeline=sync,
        file_patterns=["*.py", "README.md", "config"]
    )
    assert ci._should_ingest(Path("test.py")) is True
    assert ci._should_ingest(Path("README.md")) is True
    assert ci._should_ingest(Path("config")) is True
    assert ci._should_ingest(Path("test.txt")) is False

    ci_all = CocoIndexIngestion([], emb, sync, file_patterns=["*"])
    assert ci_all._should_ingest(Path("any.txt")) is True

@pytest.mark.anyio
async def test_cocoindex_ingestion_ingest_chunks(mock_deps):
    emb, sync = mock_deps
    ci = CocoIndexIngestion([Path("/temp_test_path")], emb, sync, session_id=123)

    assert (await ci._ingest_chunks([])).chunks_ingested == 0

    chunks = [FileChunk("p1", 0, 5, "hello")]
    res = await ci._ingest_chunks(chunks)
    assert res.chunks_ingested == 1

    sync.queue_sync.side_effect = Exception("error")
    res2 = await ci._ingest_chunks(chunks)
    assert res2.chunks_skipped == 1

@pytest.mark.anyio
async def test_cocoindex_ingestion_file_ops(tmp_path, mock_deps):
    emb, sync = mock_deps
    ci = CocoIndexIngestion([tmp_path], emb, sync, session_id=1)
    # Use pattern to force hit the loop
    ci.file_patterns = ["*.txt"]

    test_file = tmp_path / "test.txt"
    test_file.write_text("content")

    # Ingest file
    await ci._ingest_file(test_file)
    assert sync.queue_sync.called

    # Cached content skip
    sync.queue_sync.reset_mock()
    await ci._ingest_file(test_file)
    assert not sync.queue_sync.called

    # Error in read
    with patch("pathlib.Path.read_text", side_effect=Exception("oops")):
        await ci._ingest_file(test_file)

    # Scan directory
    await ci.clear_file_cache()
    sync.queue_sync.reset_mock()
    await ci._scan_directory(tmp_path)
    assert sync.queue_sync.called

    # Permission error and Exception in scan
    with patch("pathlib.Path.rglob", side_effect=PermissionError):
        await ci._scan_directory(tmp_path)

    with patch("pathlib.Path.rglob", side_effect=Exception("oops")):
        await ci._scan_directory(tmp_path)

@pytest.mark.anyio
async def test_cocoindex_ingestion_lifecycle(mock_deps):
    emb, sync = mock_deps
    # Test valid watch path to hit task creation
    ci = CocoIndexIngestion([Path()], emb, sync) # "." exists

    await ci.stop()
    await ci.start()
    assert ci._running is True
    assert len(ci._watchers) == 1
    await ci.start() # Redundant

    # Mock task for await
    async def dummy():
        pass
    task = asyncio.create_task(dummy())
    ci._watchers = [task]
    await ci.stop()
    assert ci._running is False
    assert len(ci._watchers) == 0

@pytest.mark.anyio
async def test_cocoindex_ingestion_watch_loop(mock_deps):
    emb, sync = mock_deps
    ci = CocoIndexIngestion([Path("/temp_test")], emb, sync)
    ci._running = True

    with (
        patch.object(ci, "_scan_directory", side_effect=[None, Exception("poll error"), Exception("stop")]) as mock_scan,
        patch("asyncio.sleep", side_effect=[None, None, asyncio.CancelledError]),
        contextlib.suppress(asyncio.CancelledError),
    ):
        await ci._watch_path(Path("/temp_test"))
        assert mock_scan.called

def test_cocoindex_ingestion_getset(mock_deps):
    emb, sync = mock_deps
    ci = CocoIndexIngestion([Path("/temp_test")], emb, sync)
    ci.set_session_id(456)
    assert ci.session_id == 456
    ci._file_cache["f"] = "c"
    assert ci.get_file_cache() == {"f": "c"}
