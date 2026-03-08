import sqlite3

import pytest

from src.db_manager import DBManager
from src.persistence.strategy import (
    SessionCompleteArgs,
    SessionCreateArgs,
)


@pytest.fixture
async def db_temp(tmp_path):
    db_file = tmp_path / "test_pm.db"
    db = DBManager(str(db_file))
    yield db
    await db.close()

@pytest.mark.anyio
async def test_db_manager_init(tmp_path):
    db_file = tmp_path / "init.db"
    db = DBManager(str(db_file))
    assert db_file.exists()
    await db.close()

@pytest.mark.anyio
async def test_db_manager_memory_init():
    db = DBManager(":memory:")
    assert db.db_path == ":memory:"
    await db.close()

@pytest.mark.anyio
async def test_db_manager_project_ops(db_temp):
    project_id = db_temp.create_project("Test Project", "/temp_db_test")
    assert project_id > 0

    project = db_temp.get_project(project_id)
    assert project["name"] == "Test Project"
    assert project["path"] == "/temp_db_test"

    project_by_path = db_temp.get_project_by_path("/temp_db_test")
    assert project_by_path["id"] == project_id

    db_temp.update_project_status(project_id, "active")
    project = db_temp.get_project(project_id)
    assert project["status"] == "active"

    db_temp.update_project(project_id, "Updated Project", "/temp_db_updated")
    project = db_temp.get_project(project_id)
    assert project["name"] == "Updated Project"

    projects = db_temp.get_projects()
    assert len(projects) == 1

    db_temp.delete_project(project_id)
    assert db_temp.get_project(project_id) is None

@pytest.mark.anyio
async def test_db_manager_session_ops(db_temp):
    project_id = db_temp.create_project("Session Project", "/path")

    args = SessionCreateArgs(
        project_id=project_id,
        agent_name="TestAgent",
        role="Tester",
        task="Test Task"
    )
    session_id = await db_temp.create_session(args)
    assert session_id > 0

    session = await db_temp.get_session(session_id)
    assert session["agent_name"] == "TestAgent"
    assert session["role"] == "Tester"

    complete_args = SessionCompleteArgs(
        session_id=session_id,
        result="Success",
        reasoning="Test Reasoning",
        success=True
    )
    await db_temp.complete_session(complete_args)

    session = await db_temp.get_session(session_id)
    assert session["success"] == 1
    assert session["result"] == "Success"

    sessions = await db_temp.list_sessions()
    assert len(sessions) == 1

    sessions_pn = await db_temp.list_sessions(project_id=999)
    assert len(sessions_pn) == 0

    # Test agent_name filter
    sessions_a = await db_temp.list_sessions(agent_name="TestAgent")
    assert len(sessions_a) == 1

    # Test limit
    sessions_l = await db_temp.list_sessions(limit=0)
    assert len(sessions_l) == 0

    # Test list_sessions_by_project (explicit coverage)
    sessions_bp = await db_temp.list_sessions_by_project(project_id)
    assert len(sessions_bp) == 1

@pytest.mark.anyio
async def test_db_manager_migration(tmp_path):
    db_file = tmp_path / "migrate.db"
    # Create DB without parent_session_id column
    import sqlite3
    conn = sqlite3.connect(str(db_file))
    conn.execute("CREATE TABLE agent_sessions (id INTEGER PRIMARY KEY, project_id INTEGER, agent_name TEXT NOT NULL, role TEXT NOT NULL, task TEXT, prompt TEXT, result TEXT, reasoning_content TEXT, success BOOLEAN, tokens_used INTEGER, started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, completed_at TIMESTAMP)")
    conn.close()

    # DBManager should add the missing column
    db = DBManager(str(db_file))
    await db.initialize()

    # Check if column exists
    conn = sqlite3.connect(str(db_file))
    cursor = conn.execute("PRAGMA table_info(agent_sessions)")
    cols = [c[1] for c in cursor.fetchall()]
    assert "parent_session_id" in cols
    conn.close()
    await db.close()

@pytest.mark.anyio
async def test_db_manager_memory_ops(db_temp):
    project_id = db_temp.create_project("Memory Project", "/path")
    session_args = SessionCreateArgs(project_id=project_id, agent_name="A", role="R")
    session_id = await db_temp.create_session(session_args)

    await db_temp.add_memory(session_id, "Test memory text", {"key": "value"}, "chunk1")

    memories = await db_temp.list_memories(session_id)
    assert len(memories) == 1
    memory_id = memories[0].id

    memory = await db_temp.get_memory(memory_id)
    assert memory.text == "Test memory text"
    assert memory.metadata["key"] == "value"
    assert memory.created_at is not None

    # Corrupt metadata JSON to test exception handler
    with db_temp._get_connection() as conn:
        conn.execute("UPDATE memories SET metadata = 'invalid json' WHERE id = ?", (memory_id,))
    memory = await db_temp.get_memory(memory_id)
    assert memory.metadata == {}

    # Corrupt metadata JSON to test exception handler in list_memories
    with db_temp._get_connection() as conn:
        conn.execute("UPDATE memories SET metadata = 'invalid json'")

    # Trigger search/search_by_text JSONDecodeError
    results_err = await db_temp.search([0.1]*128)
    assert results_err[0].metadata == {}

    # Session-scoped JSON error
    results_err_s = await db_temp.search([0.1]*128, session_id=session_id)
    assert results_err_s[0].metadata == {}

    results_text_err = await db_temp.search_by_text("test")
    assert results_text_err[0].metadata == {}

    # Session-scoped text JSON error
    results_text_err_s = await db_temp.search_by_text("test", session_id=session_id)
    assert results_text_err_s[0].metadata == {}

    # Trigger list_memories JSONDecodeError
    list_err = await db_temp.list_memories()
    assert list_err[0].metadata == {}

    # Test list_memories with session_id=None (already done by list_memories() but being explicit)
    list_all = await db_temp.list_memories(session_id=None)
    assert len(list_all) >= 1

    # Call add_vector (covered pass)
    await db_temp.add_vector(list_err[0])

    # Test remove vector by chunk_key (cov 293-296)
    await db_temp.add_memory(session_id, "text2", chunk_key="key2")
    assert await db_temp.remove_vector("key2") is True
    assert await db_temp.remove_vector("nonexistent") is False

    # Test remove_memory False
    assert await db_temp.remove_memory(999999) is False

    # Test get_memory not found (cov line 154)
    assert await db_temp.get_memory(999999) is None

@pytest.mark.anyio
async def test_db_manager_legacy_methods(db_temp):
    project_id = db_temp.create_project("Legacy Project", "/path")

    session_id = db_temp.create_session_sync(project_id, "LegacyAgent", "L", "Task")
    assert session_id > 0

    session = db_temp.get_session_sync(session_id)
    assert session["agent_name"] == "LegacyAgent"

    db_temp.complete_session_sync(session_id, "Legacy Result", True, tokens=100)
    session = db_temp.get_session_sync(session_id)
    assert session["tokens_used"] == 100

    assert len(db_temp.list_sessions_sync()) == 1
    assert len(db_temp.list_sessions_by_project_sync(project_id)) == 1

@pytest.mark.anyio
async def test_db_manager_error_handling(tmp_path):
    # Test read-only directory
    ro_dir = tmp_path / "readonly"
    ro_dir.mkdir()
    db_file = ro_dir / "db.sqlite"

    # Make dir read-only
    ro_dir.chmod(0o555)

    # DBManager should fallback to :memory: if it can't initialize disk DB
    # or handle it according to its implementation.
    # In src/db_manager.py:304, it logs error and falls back to :memory:
    db = DBManager(str(db_file))
    assert db.db_path == ":memory:"
    await db.close()

@pytest.mark.anyio
async def test_db_manager_concurrency_retry(tmp_path):
    db_file = tmp_path / "retry.db"
    db = DBManager(str(db_file))
    await db.initialize()
    await db.close()

@pytest.mark.anyio
async def test_db_manager_mocked_failures(tmp_path):
    from unittest.mock import MagicMock, patch
    db_file = tmp_path / "mock.db"

    # Mock PRAGMA failure
    db = DBManager(str(db_file))
    with patch("sqlite3.connect") as mock_connect:
        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn
        # Trigger OperationalError only on the first execute call
        mock_conn.execute.side_effect = [sqlite3.OperationalError("WAL failed"), MagicMock()]

        # This should trigger the warning and fallback
        conn = db._get_connection()
        assert conn == mock_conn
        assert mock_conn.execute.call_count == 2

    # Mock Retry loop (disk i/o)
    with (
        patch("sqlite3.connect") as mock_connect,
        patch("time.sleep") as mock_sleep,
    ):
        mock_connect.side_effect = sqlite3.OperationalError("disk i/o error")
        with pytest.raises(RuntimeError, match="Failed to establish database connection"):
            db._get_connection()
        assert mock_connect.call_count == 3
        assert mock_sleep.call_count == 2

    # Test connection reuse (cov line 324)
    db_mem = DBManager(":memory:")
    c1 = db_mem._get_connection()
    c2 = db_mem._get_connection()
    assert c1 == c2
    await db_mem.close()

    # Mock Non-retryable error
    with patch("sqlite3.connect") as mock_connect:
        mock_connect.side_effect = sqlite3.OperationalError("syntax error")
        with pytest.raises(sqlite3.OperationalError):
            db._get_connection()
        assert mock_connect.call_count == 1

    await db.close()

@pytest.mark.anyio
async def test_db_manager_properties(db_temp):
    assert db_temp.session_repo == db_temp
    assert db_temp.memory_repo == db_temp
    assert db_temp.vector_repo == db_temp
