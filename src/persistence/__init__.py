"""Persistence module for the Cybernetic Production Studio.

This module provides a hybrid persistence layer that separates the
orchestration strategy from the memory storage and retrieval.

It implements:
- PersistenceStrategy: Abstract interface for persistence operations
- SessionRepository: Repository for agent session management
- MemoryRepository: Repository for local agent memory (SQLite/svec)
- VectorSearchRepository: Repository for global vector search (Postgres/CocoIndex)
- PostgresPersistence: Postgres implementation with pgvector
- SQLitePersistence: SQLite implementation with sqlite-vec
- HybridPersistence: Combination with Event-Driven synchronization
- VectorSyncPipeline: Event-driven synchronization pipeline
- EmbeddingService: Service for computing embeddings at source
"""

from src.persistence.embedding_service import EmbeddingService
from src.persistence.hybrid_persistence import HybridPersistence
from src.persistence.postgres_persistence import PostgresPersistence
from src.persistence.sqlite_persistence import SQLitePersistence
from src.persistence.strategy import (
    MemoryRepository,
    PersistenceStrategy,
    SessionRepository,
    VectorSearchRepository,
)
from src.persistence.sync_pipeline import VectorSyncPipeline

__all__ = [
    "EmbeddingService",
    "HybridPersistence",
    "MemoryRepository",
    "PersistenceStrategy",
    "PostgresPersistence",
    "SQLitePersistence",
    "SessionRepository",
    "VectorSearchRepository",
    "VectorSyncPipeline",
]
