
import pytest

from src.persistence.sqlite_persistence import SQLitePersistence


@pytest.fixture
async def sqlite_temp(tmp_path):
    db_file = tmp_path / "test_sqlite.db"
    # Create agent_sessions table first as memories depends on it
    import sqlite3
    conn = sqlite3.connect(str(db_file))
    conn.execute("CREATE TABLE agent_sessions (id INTEGER PRIMARY KEY)")
    conn.execute("INSERT INTO agent_sessions (id) VALUES (1)")
    conn.commit()
    conn.close()

    p = SQLitePersistence(str(db_file))
    yield p
    p.close()

@pytest.mark.anyio
async def test_sqlite_persistence_add_get(sqlite_temp):
    mem_id = await sqlite_temp.add_memory(
        session_id=1,
        text="Hello world",
        metadata={"source": "test"},
        chunk_key="unique_key"
    )
    assert mem_id > 0

    # Test duplicate chunk_key
    dup_id = await sqlite_temp.add_memory(1, "other", chunk_key="unique_key")
    assert dup_id == mem_id

    memory = await sqlite_temp.get_memory(mem_id)
    assert memory.text == "Hello world"
    assert memory.metadata["source"] == "test"
    assert memory.chunk_key == "unique_key"

@pytest.mark.anyio
async def test_sqlite_persistence_list_remove(sqlite_temp):
    await sqlite_temp.add_memory(1, "m1")
    await sqlite_temp.add_memory(1, "m2")

    memories = await sqlite_temp.list_memories(session_id=1)
    assert len(memories) == 2

    all_memories = await sqlite_temp.list_memories(session_id=None)
    assert len(all_memories) == 2

    await sqlite_temp.remove_memory(memories[0].id)
    assert len(await sqlite_temp.list_memories(1)) == 1

@pytest.mark.anyio
async def test_sqlite_persistence_errors(tmp_path):
    p = SQLitePersistence(str(tmp_path / "err.db"))
    p.close()

    with pytest.raises(RuntimeError, match="Database connection not initialized"):
        await p.add_memory(1, "text")

    with pytest.raises(RuntimeError, match="Database connection not initialized"):
        await p.get_memory(1)

    with pytest.raises(RuntimeError, match="Database connection not initialized"):
        await p.list_memories()

    with pytest.raises(RuntimeError, match="Database connection not initialized"):
        await p.remove_memory(1)

@pytest.mark.anyio
async def test_sqlite_persistence_corrupt_metadata(sqlite_temp):
    import sqlite3
    mem_id = await sqlite_temp.add_memory(1, "corrupt")
    # Manually corrupt metadata
    conn = sqlite3.connect(sqlite_temp.db_path)
    conn.execute("UPDATE memories SET metadata = 'invalid' WHERE id = ?", (mem_id,))
    conn.commit()
    conn.close()

    memory = await sqlite_temp.get_memory(mem_id)
    assert memory.metadata == {}

@pytest.mark.anyio
async def test_sqlite_persistence_embedding(sqlite_temp):
    embed = [0.1, 0.2, 0.3]
    mem_id = await sqlite_temp.add_memory(1, "embed text", embedding=embed)

    memory = await sqlite_temp.get_memory(mem_id)
    assert memory.embedding == embed

    memories = await sqlite_temp.list_memories(1)
    assert memories[0].embedding == embed

@pytest.mark.anyio
async def test_sqlite_persistence_get_not_found(sqlite_temp):
    assert await sqlite_temp.get_memory(999) is None
