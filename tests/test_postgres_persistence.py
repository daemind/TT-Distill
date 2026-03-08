from typing import Any, Generator, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.persistence.postgres_persistence import PostgresPersistence
from src.persistence.strategy import SessionCompleteArgs, SessionCreateArgs


@pytest.fixture
def mock_asyncpg() -> Generator[tuple[MagicMock, MagicMock, MagicMock], None, None]:
    with patch("src.persistence.postgres_persistence.asyncpg") as mock:
        mock_pool = MagicMock()
        mock_conn = MagicMock()

        mock_conn.execute = AsyncMock(return_value="OK")
        mock_conn.fetch = AsyncMock(return_value=[])
        mock_conn.fetchrow = AsyncMock(return_value=None)
        mock_conn.fetchval = AsyncMock(return_value=None)
        mock_conn.close = AsyncMock()

        mock_pool.acquire = MagicMock()
        acm = AsyncMock()
        acm.__aenter__.return_value = mock_conn
        mock_pool.acquire.return_value = acm
        mock_pool.close = AsyncMock()

        mock.create_pool = AsyncMock(return_value=mock_pool)

        yield mock, mock_pool, mock_conn

@pytest.mark.anyio
async def test_postgres_persistence_init_close(mock_asyncpg: tuple[MagicMock, MagicMock, MagicMock]) -> None:
    mock_api, mock_pool, _ = mock_asyncpg
    pp = PostgresPersistence(db_url="postgres://user:pass@localhost/db")

    with patch("src.persistence.postgres_persistence.ASYNCPG_AVAILABLE", True):
        await pp.initialize()
        mock_api.create_pool.assert_awaited_once_with("postgres://user:pass@localhost/db")

        await pp.close()
        mock_pool.close.assert_awaited_once()

@pytest.mark.anyio
async def test_postgres_persistence_create_session(mock_asyncpg: tuple[MagicMock, MagicMock, MagicMock]) -> None:
    _, _, mock_conn = mock_asyncpg
    pp = PostgresPersistence()
    with patch("src.persistence.postgres_persistence.ASYNCPG_AVAILABLE", True):
        await pp.initialize()
        mock_conn.fetchrow.return_value = {"id": 123}
        args = SessionCreateArgs(project_id=1, agent_name="test", role="user", task="desc", prompt="p")
        assert await pp.create_session(args) == 123

@pytest.mark.anyio
async def test_postgres_persistence_complete_session(mock_asyncpg: tuple[MagicMock, MagicMock, MagicMock]) -> None:
    _, _, mock_conn = mock_asyncpg
    pp = PostgresPersistence()
    with patch("src.persistence.postgres_persistence.ASYNCPG_AVAILABLE", True):
        await pp.initialize()
        args = SessionCompleteArgs(session_id=123, result="done", reasoning="none", success=True, tokens_used=10)
        await pp.complete_session(args)
        assert mock_conn.execute.called

@pytest.mark.anyio
async def test_postgres_persistence_get_session(mock_asyncpg: tuple[MagicMock, MagicMock, MagicMock]) -> None:
    _, _, mock_conn = mock_asyncpg
    pp = PostgresPersistence()
    with patch("src.persistence.postgres_persistence.ASYNCPG_AVAILABLE", True):
        await pp.initialize()
        from datetime import datetime
        mock_conn.fetchrow.side_effect = [
            {
                "id": 123, "project_id": 1, "agent_name": "test", "role": "user",
                "status": "active", "task": "desc", "prompt": "p", "result": "r",
                "reasoning_content": "none", "success": True, "tokens_used": 10,
                "parent_session_id": None, "started_at": datetime.now(), "completed_at": datetime.now()
            },
            { # Token used None and parent None
                "id": 124, "project_id": 1, "agent_name": "test", "role": "user",
                "status": "active", "task": "desc", "prompt": "p", "result": "r",
                "reasoning_content": "none", "success": True, "tokens_used": None,
                "parent_session_id": None, "started_at": None, "completed_at": None
            },
            None
        ]
        s1 = await pp.get_session(123)
        assert s1 is not None
        assert s1.id == 123
        assert s1.completed_at is not None

        s2 = await pp.get_session(124)
        assert s2 is not None
        assert s2.tokens_used is None
        assert s2.started_at == ""

        assert await pp.get_session(999) is None

@pytest.mark.anyio
async def test_postgres_persistence_list_sessions(mock_asyncpg: tuple[MagicMock, MagicMock, MagicMock]) -> None:
    _, _, mock_conn = mock_asyncpg
    pp = PostgresPersistence()
    with patch("src.persistence.postgres_persistence.ASYNCPG_AVAILABLE", True):
        await pp.initialize()
        from datetime import datetime
        row = {"id": 1, "project_id": 1, "agent_name": "a1", "role": "user", "status": "active", "task": "d1", "prompt": "p", "result": "r", "reasoning_content": "none", "success": True, "tokens_used": 10, "parent_session_id": None, "started_at": datetime.now(), "completed_at": None}
        mock_conn.fetch.return_value = [row]

        # Test all filter combinations
        await pp.list_sessions(project_id=1, agent_name="a1")
        await pp.list_sessions(project_id=1)
        await pp.list_sessions(agent_name="a1")
        await pp.list_sessions()
        assert mock_conn.fetch.call_count == 4

@pytest.mark.anyio
async def test_postgres_persistence_search(mock_asyncpg: tuple[MagicMock, MagicMock, MagicMock]) -> None:
    _, _, mock_conn = mock_asyncpg
    pp = PostgresPersistence()
    with patch("src.persistence.postgres_persistence.ASYNCPG_AVAILABLE", True):
        await pp.initialize()

        mock_conn.fetch.return_value = [
            {"id": 10, "session_id": 123, "text": "found", "metadata": '{"key":1}', "embedding": [0.1]*1024, "similarity": 0.9},
            {"id": 11, "session_id": 123, "text": "low sim", "metadata": None, "embedding": None, "similarity": 0.1}
        ]
        # Similarity bug fixed: now it checks float(row["similarity"]) >= min_similarity
        results = await pp.search([0.1]*1024, session_id=123, min_similarity=0.5)
        assert len(results) == 1
        assert results[0].text == "found"
        assert results[0].metadata == {"key": 1}

        await pp.search([0.1]*1024)
        assert mock_conn.fetch.call_count == 2

@pytest.mark.anyio
async def test_postgres_persistence_search_by_text(mock_asyncpg: tuple[MagicMock, MagicMock, MagicMock]) -> None:
    _, _, mock_conn = mock_asyncpg
    pp = PostgresPersistence()
    with patch("src.persistence.postgres_persistence.ASYNCPG_AVAILABLE", True):
        await pp.initialize()
        mock_conn.fetch.return_value = [{"id": 1, "session_id": 123, "text": "t", "metadata": None, "embedding": None}]
        assert len(await pp.search_by_text("q", session_id=123)) == 1
        assert len(await pp.search_by_text("q")) == 1

@pytest.mark.anyio
async def test_postgres_persistence_add_vector(mock_asyncpg: tuple[MagicMock, MagicMock, MagicMock]) -> None:
    _, _, mock_conn = mock_asyncpg
    pp = PostgresPersistence()
    with patch("src.persistence.postgres_persistence.ASYNCPG_AVAILABLE", True):
        await pp.initialize()
        mock_conn.fetchval.return_value = None
        mock_conn.fetchrow.return_value = {"id": 1}
        await pp.add_vector(123, "text", {"m": 1}, [0.1]*1024, chunk_key="key")

        mock_conn.fetchval.return_value = 1
        assert await pp.add_vector(123, "text", {}, [0.1], chunk_key="key") == 1

@pytest.mark.anyio
async def test_postgres_persistence_remove_vector(mock_asyncpg: tuple[MagicMock, MagicMock, MagicMock]) -> None:
    _, _, mock_conn = mock_asyncpg
    pp = PostgresPersistence()
    with patch("src.persistence.postgres_persistence.ASYNCPG_AVAILABLE", True):
        await pp.initialize()
        mock_conn.execute.return_value = "DELETE 1"
        assert await pp.remove_vector(1) is True
        mock_conn.execute.return_value = "DELETE 0"
        assert await pp.remove_vector(2) is False

@pytest.mark.anyio
async def test_postgres_persistence_sync_wrappers():
    pp = PostgresPersistence()
    # Use setattr to bypass mypy method assignment checks
    # MagicMock is used because the sync wrappers expect results after run_until_complete,
    # and mocking the loop to return the coroutine itself (AsyncMock) causes assertion failures.
    setattr(pp, "create_session", MagicMock(return_value=123))
    setattr(pp, "get_session", MagicMock(return_value=123))
    setattr(pp, "complete_session", MagicMock(return_value=None))
    setattr(pp, "list_sessions", MagicMock(return_value=123))
    setattr(pp, "search", MagicMock(return_value=123))
    setattr(pp, "search_by_text", MagicMock(return_value=123))
    setattr(pp, "add_vector", MagicMock(return_value=123))
    setattr(pp, "remove_vector", MagicMock(return_value=123))

    with patch("asyncio.get_event_loop") as mock_get_loop:
        mock_loop = MagicMock()
        mock_get_loop.return_value = mock_loop
        # The sync wrappers just return what run_until_complete returns
        mock_loop.run_until_complete.side_effect = lambda x: x

        # Since run_until_complete is mocked to return the coroutine result,
        # and our AsyncMocks return ints, we need to cast or just check equality
        # with the understanding that the sync wrappers call run_until_complete internally.
        assert cast(Any, pp.session_repo.create_session(MagicMock())) == 123
        assert cast(Any, pp.session_repo.get_session(1)) == 123
        pp.session_repo.complete_session(MagicMock())
        assert cast(Any, pp.session_repo.list_sessions()) == 123
        assert cast(Any, pp.vector_repo.search([0.1])) == 123
        assert cast(Any, pp.vector_repo.search_by_text("q")) == 123
        assert cast(Any, pp.vector_repo.add_vector(1, "t", {}, [0.1])) == 123
        assert cast(Any, pp.vector_repo.remove_vector(1)) == 123

@pytest.mark.anyio
async def test_postgres_persistence_properties(mock_asyncpg: tuple[MagicMock, MagicMock, MagicMock]) -> None:
    pp = PostgresPersistence()
    with patch("src.persistence.postgres_persistence.ASYNCPG_AVAILABLE", True):
        await pp.initialize()
        assert pp.session_repo is not None
        assert pp.vector_repo is not None
        assert pp.memory_repo is not None

@pytest.mark.anyio
async def test_postgres_persistence_errors() -> None:
    pp = PostgresPersistence()
    # Test all uninitialized calls
    with pytest.raises(RuntimeError, match="not initialized"):
        await pp.create_session(MagicMock())
    with pytest.raises(RuntimeError, match="not initialized"):
        await pp.complete_session(MagicMock())
    with pytest.raises(RuntimeError, match="not initialized"):
        await pp.get_session(1)
    with pytest.raises(RuntimeError, match="not initialized"):
        await pp.list_sessions()
    with pytest.raises(RuntimeError, match="not initialized"):
        await pp.search([0.1])
    with pytest.raises(RuntimeError, match="not initialized"):
        await pp.search_by_text("q")
    with pytest.raises(RuntimeError, match="not initialized"):
        await pp.add_vector(1, "t", {}, [0.1])
    with pytest.raises(RuntimeError, match="not initialized"):
        await pp.remove_vector(1)

    with pytest.raises(RuntimeError, match="not initialized"):
        _ = pp.memory_repo

    with (
        patch("src.persistence.postgres_persistence.ASYNCPG_AVAILABLE", False),
        pytest.raises(RuntimeError, match="asyncpg is not installed"),
    ):
        await pp.initialize()
