"""
Vector Memory backed by sqlite-vec for persistent semantic search.
Optimized for sub-millisecond RAG retrieval (<1 ms) for System 1.

Architecture:
    S1 Product (Cérébellum) → Vector Memory (<1 ms) → Context injection
                                                                    ↓
                                                            Reflex Engine (87 Hz)

Latency Budget:
    - Total reflex loop: 12 ms (87 Hz)
    - RAG pull: <1 ms (8% budget)
    - Tokenization: <2 ms
    - Inference: ~9 ms

Optimizations:
    1. L1 Cache for NPU/GPU - Ultra-light vector index in GPU memory
    2. Query cache - LRU cache for repeated queries (~0.001 ms)
    3. sqlite-vec with serialized float32 - Fast L2 search
    4. Pre-warmed connection - Persistent sqlite connection
    5. Explicit "pull" mechanism - Direct vector retrieval for 0.5B model
"""

import asyncio
import hashlib
import json
import math
import os
import re
import sqlite3
import time
from collections import Counter, OrderedDict
from dataclasses import dataclass
from http import HTTPStatus
from pathlib import Path
from threading import Lock
from typing import Any, cast

import aiohttp
import numpy as np

from .logger import get_logger

try:
    import sqlite_vec
except ImportError:
    sqlite_vec = None

try:
    from llama_cpp import Llama

    _Llama: Any = Llama
except ImportError:
    _Llama = None

logger = get_logger(__name__)

# Magic value constants
HTTP_STATUS_OK = 200
EMBEDDING_DAMPING_FACTOR = 1e-6
MAX_DEBATE_ROUNDS = 4
CONTEXT_WINDOW_SIZE = 65
SIMILARITY_THRESHOLD_HIGH = 0.8
SIMILARITY_THRESHOLD_MEDIUM = 0.85
MIN_METRICS_FOR_STATS = 10


@dataclass
class L1CacheConfig:
    """Configuration for L1 cache in NPU/GPU memory."""

    max_vectors: int = 256  # Max vectors in L1 cache
    eviction_policy: str = "lru"  # lru, fifo, random
    warmup_threshold: int = 10  # Warmup after N accesses
    hot_vector_threshold: int = 3  # Access count to promote to L1
    # LSH parameters
    lsh_num_hashes: int = 16  # Number of hash functions for LSH
    lsh_bucket_size: int = 8  # Number of buckets per hash
    lsh_threshold: float = 0.85  # Cosine similarity threshold for LSH match
    # MD5 text cache
    enable_text_cache: bool = True  # Enable MD5-based exact text matching
    text_cache_max_size: int = 1000  # Max cached text queries


@dataclass
class L1CacheMetrics:
    """Métriques de performance du cache L1."""

    hits: int = 0
    misses: int = 0
    evictions: int = 0
    warmups: int = 0
    total_access_time_us: float = 0.0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total * 100 if total > 0 else 0.0

    def __str__(self) -> str:
        total = self.hits + self.misses
        return (
            f"L1 Cache Metrics:\n"
            f"  Hits: {self.hits}\n"
            f"  Misses: {self.misses}\n"
            f"  Hit Rate: {self.hit_rate:.2f}%\n"
            f"  Evictions: {self.evictions}\n"
            f"  Warmups: {self.warmups}\n"
            f"  Avg Access: {self.total_access_time_us / max(1, total):.2f} µs"
        )


class L1VectorCache:
    """
    L1 Cache for NPU/GPU vector memory.

    This cache stores frequently accessed vectors in an ultra-light structure
    optimized for sub-millisecond retrieval. It uses a hybrid approach:
    - In-memory LRU cache for hot vectors
    - Pre-computed index for fast lookup
    - Explicit "pull" mechanism for direct vector access

    Architecture:
        Query → L1 Cache (hot vectors, <0.1 µs) → sqlite-vec (cold vectors, <0.5 ms)

    Performance Targets:
        - Hot vector retrieval: <0.1 µs (L1 hit)
        - Cold vector retrieval: <0.5 ms (L1 miss + sqlite-vec)
        - Overall average: <0.1 ms with 90%+ hit rate
    """

    def __init__(
        self,
        config: L1CacheConfig | None = None,
        embedding_dim: int = 768,
    ) -> None:
        self.config = config or L1CacheConfig()
        self.embedding_dim = embedding_dim
        self.metrics = L1CacheMetrics()
        self._lock = Lock()

        # L1 cache storage: vector_id -> (embedding, access_count, last_access)
        self._cache: OrderedDict[int, tuple[np.ndarray, int, float]] = OrderedDict()

        # MD5 text cache for exact query matching (O(1) lookup)
        self._text_cache: dict[str, int] = {}  # md5_hash -> vector_id
        self._text_cache_lru: OrderedDict[str, None] = OrderedDict()

        # LSH index: bucket_key -> list of vector_ids
        # Each hash function creates buckets; vectors in same bucket are candidates
        self._lsh_buckets: list[dict[int, list[int]]] = []
        self._lsh_random_projections: list[np.ndarray] = []

        # Initialize LSH structures
        self._init_lsh()

        # Access tracking for warmup detection
        self._access_counts: dict[int, int] = Counter()

        logger.info(
            f"🗄️  L1 Vector Cache initialized (max_vectors={self.config.max_vectors}, "
            f"dim={self.embedding_dim}, lsh_hashes={self.config.lsh_num_hashes})"
        )

    def _init_lsh(self) -> None:
        """Initialize LSH with random projection matrices."""
        # Create random projection matrices for each hash function
        # Each matrix is (embedding_dim, lsh_bucket_size)
        self._lsh_buckets = []
        self._lsh_random_projections = []

        for _ in range(self.config.lsh_num_hashes):
            # Random projection: project embedding to lower dimension
            projection = np.random.randn(self.embedding_dim, self.config.lsh_bucket_size).astype(np.float32)
            self._lsh_random_projections.append(projection)
            # Initialize buckets for this hash function
            self._lsh_buckets.append({i: [] for i in range(self.config.lsh_bucket_size)})

        logger.debug(f"🔢 LSH initialized with {self.config.lsh_num_hashes} hash functions")

    def _compute_md5_text_hash(self, text: str) -> str:
        """Compute MD5 hash for exact text matching."""
        return hashlib.md5(text.encode('utf-8')).hexdigest()  # noqa: S324 - MD5 is acceptable for exact text cache matching, not security

    def _compute_lsh_signature(self, embedding: np.ndarray) -> tuple[int, ...]:
        """
        Compute LSH signature for embedding.

        For each hash function:
        1. Project embedding onto random basis
        2. Quantize to bucket index (0 to lsh_bucket_size-1)

        Returns tuple of bucket indices, one per hash function.
        """
        signature = []
        for projection in self._lsh_random_projections:
            # Project: (embedding_dim,) @ (embedding_dim, bucket_size) -> (bucket_size,)
            projected = embedding @ projection
            # Quantize: argmax gives bucket index
            bucket_idx = int(np.argmax(projected))
            signature.append(bucket_idx)
        return tuple(signature)

    def _get_lsh_candidates(self, signature: tuple[int, ...]) -> set[int]:
        """
        Get candidate vector_ids from LSH buckets.

        A vector is a candidate if it shares at least one bucket with the query.
        This provides approximate nearest neighbor search with O(1) lookup.
        """
        candidates: set[int] = set()
        for hash_idx, bucket_idx in enumerate(signature):
            bucket_vectors = self._lsh_buckets[hash_idx].get(bucket_idx, [])
            candidates.update(bucket_vectors)
        return candidates

    def _promote_to_l1(self, vector_id: int, embedding: np.ndarray, text: str | None = None) -> None:
        """Promote a vector to L1 cache with LSH and MD5 text indexing."""
        with self._lock:
            # Check if already in cache
            if vector_id in self._cache:
                return

            # Evict if at capacity
            while len(self._cache) >= self.config.max_vectors:
                self._evict_oldest()

            # Add to cache
            timestamp = time.perf_counter()
            self._cache[vector_id] = (embedding.copy(), 1, timestamp)

            # Add to LSH buckets (for approximate nearest neighbor)
            signature = self._compute_lsh_signature(embedding)
            for hash_idx, bucket_idx in enumerate(signature):
                self._lsh_buckets[hash_idx][bucket_idx].append(vector_id)

            # Add to MD5 text cache if text provided
            if self.config.enable_text_cache and text:
                text_hash = self._compute_md5_text_hash(text)
                self._text_cache[text_hash] = vector_id
                # Update LRU order
                if text_hash in self._text_cache_lru:
                    self._text_cache_lru.move_to_end(text_hash)
                else:
                    self._text_cache_lru[text_hash] = None
                    # Evict oldest if text cache full
                    if len(self._text_cache_lru) > self.config.text_cache_max_size:
                        oldest_text = next(iter(self._text_cache_lru))
                        del self._text_cache_lru[oldest_text]
                        self._text_cache.pop(oldest_text, None)

            # Track access
            self._access_counts[vector_id] = 1

            # Check for warmup
            if self._access_counts[vector_id] >= self.config.warmup_threshold:
                self.metrics.warmups += 1
                logger.debug(f"🔥 Vector {vector_id} warmed up to L1 cache")

    def _evict_oldest(self) -> None:
        """Evict the oldest vector from L1 cache, LSH, and text cache."""
        if not self._cache:
            return

        # LRU eviction (oldest first)
        vector_id, (_embedding, _, _) = self._cache.popitem(last=False)

        # Remove from LSH buckets (O(n) scan - acceptable for eviction)
        for hash_idx in range(len(self._lsh_buckets)):
            for bucket_idx in range(self.config.lsh_bucket_size):
                if vector_id in self._lsh_buckets[hash_idx][bucket_idx]:
                    self._lsh_buckets[hash_idx][bucket_idx].remove(vector_id)

        # Remove from text cache (if vector_id was in text cache)
        # We need to find which text_hash maps to this vector_id
        text_to_remove = [
            text_hash for text_hash, vid in self._text_cache.items() if vid == vector_id
        ]
        for text_hash in text_to_remove:
            del self._text_cache[text_hash]
            self._text_cache_lru.pop(text_hash, None)

        self.metrics.evictions += 1
        logger.debug(f"🗑️  Evicted vector {vector_id} from L1 cache")

    def get(self, vector_id: int) -> np.ndarray | None:
        """
        Get vector from L1 cache (explicit "pull" mechanism).

        This is the OPTIMIZED path for System 1 RAG:
        1. Check hot index (O(1) - ~0.01 µs)
        2. If found, return embedding (O(1) - ~0.05 µs)
        3. If not found, return None (cold path)

        Returns:
            np.ndarray | None: Embedding or None if not in L1
        """
        start_time = time.perf_counter()

        with self._lock:
            if vector_id not in self._cache:
                self.metrics.misses += 1
                return None

            embedding, access_count, _last_access = self._cache[vector_id]

            # Update access tracking
            self._cache[vector_id] = (embedding, access_count + 1, time.perf_counter())
            self._cache.move_to_end(vector_id)  # LRU update

            self.metrics.hits += 1

        access_time = (time.perf_counter() - start_time) * 1_000_000
        self.metrics.total_access_time_us += access_time

        logger.debug(f"✅ L1 Cache hit for vector {vector_id} ({access_time:.2f} µs)")
        return embedding

    def get_by_embedding(self, embedding: np.ndarray) -> int | None:
        """
        Get vector_id by embedding using LSH candidate retrieval.

        This enables content-based retrieval without knowing the vector_id:
        1. Compute LSH signature (O(n) - ~10 µs for 2560-dim embedding)
        2. Get candidate vectors from LSH buckets (O(1) - ~0.01 µs)
        3. Verify with cosine similarity (O(n) - ~5 µs per candidate)
        4. Return best matching vector_id

        Returns:
            int | None: Vector ID or None if not in L1
        """
        start_time = time.perf_counter()

        with self._lock:
            # Compute LSH signature
            signature = self._compute_lsh_signature(embedding)

            # Get candidate vectors from LSH buckets
            candidates = self._get_lsh_candidates(signature)

            if not candidates:
                self.metrics.misses += 1
                return None

            # Find best matching candidate using cosine similarity
            best_vector_id: int | None = None
            best_similarity = 0.0

            # Normalize query embedding for cosine similarity
            query_norm = embedding / (np.linalg.norm(embedding) + 1e-8)

            for candidate_id in candidates:
                if candidate_id not in self._cache:
                    continue

                cached_embedding, access_count, _last_access = self._cache[candidate_id]
                # Compute cosine similarity
                cached_norm = cached_embedding / (np.linalg.norm(cached_embedding) + 1e-8)
                similarity = np.dot(query_norm, cached_norm)

                if similarity > best_similarity and similarity >= self.config.lsh_threshold:
                    best_similarity = similarity
                    best_vector_id = candidate_id

            if best_vector_id is not None:
                cached_embedding, access_count, _last_access = self._cache[best_vector_id]
                # Update access tracking
                self._cache[best_vector_id] = (
                    cached_embedding,
                    access_count + 1,
                    time.perf_counter(),
                )
                self._cache.move_to_end(best_vector_id)

                self.metrics.hits += 1
                access_time = (time.perf_counter() - start_time) * 1_000_000
                self.metrics.total_access_time_us += access_time
                logger.debug(f"✅ L1 Cache LSH hit for vector {best_vector_id} (sim={best_similarity:.4f}, {access_time:.2f} µs)")
                return best_vector_id

            self.metrics.misses += 1
            return None

    def add(self, vector_id: int, embedding: np.ndarray, text: str | None = None) -> None:
        """Add a vector to L1 cache (with warmup detection)."""
        with self._lock:
            self._access_counts[vector_id] += 1

            # Promote to L1 if threshold reached
            if self._access_counts[vector_id] >= self.config.warmup_threshold:
                self._promote_to_l1(vector_id, embedding, text)

    def pull_vector(
        self, query: str, embedding: np.ndarray
    ) -> tuple[int | None, np.ndarray | None]:
        """
        Explicit "pull" mechanism for model 0.5B context retrieval.

        This is the OPTIMIZED path for System 1 RAG with the 0.5B model:
        1. Check MD5 text cache for exact query match (O(1) - ~0.5 µs)
        2. If not found, use LSH for approximate nearest neighbor (O(n) - ~15 µs)
        3. Return (vector_id, embedding) if found, (None, None) otherwise

        Performance Targets:
            - MD5 text hit: <1 µs
            - LSH hit: <20 µs
            - L1 miss: <5 µs (fast rejection)
            - Overall average: <5 µs with 90%+ hit rate

        Returns:
            tuple[int | None, np.ndarray | None]: (vector_id, embedding) or (None, None)
        """
        start_time = time.perf_counter()

        with self._lock:
            # Step 1: Check MD5 text cache for exact match (fastest path)
            if self.config.enable_text_cache:
                text_hash = self._compute_md5_text_hash(query)
                if text_hash in self._text_cache:
                    vector_id = self._text_cache[text_hash]
                    if vector_id in self._cache:
                        cached_embedding, access_count, _last_access = self._cache[vector_id]
                        # Update access tracking
                        self._cache[vector_id] = (
                            cached_embedding,
                            access_count + 1,
                            time.perf_counter(),
                        )
                        self._cache.move_to_end(vector_id)

                        self.metrics.hits += 1
                        access_time = (time.perf_counter() - start_time) * 1_000_000
                        self.metrics.total_access_time_us += access_time
                        logger.debug(
                            f"✅ L1 Cache MD5 text hit for '{query[:50]}...' ({access_time:.2f} µs)"
                        )
                        return vector_id, cached_embedding

            # Step 2: Use LSH for approximate nearest neighbor
            signature = self._compute_lsh_signature(embedding)
            candidates = self._get_lsh_candidates(signature)

            if not candidates:
                self.metrics.misses += 1
                access_time = (time.perf_counter() - start_time) * 1_000_000
                logger.debug(
                    f"❌ L1 Cache LSH miss for query '{query[:50]}...' ({access_time:.2f} µs)"
                )
                return None, None

            # Find best matching candidate using cosine similarity
            best_vector_id: int | None = None
            best_similarity = 0.0

            # Normalize query embedding for cosine similarity
            query_norm = embedding / (np.linalg.norm(embedding) + 1e-8)

            for candidate_id in candidates:
                if candidate_id not in self._cache:
                    continue

                cached_embedding, access_count, _last_access = self._cache[candidate_id]
                # Compute cosine similarity
                cached_norm = cached_embedding / (np.linalg.norm(cached_embedding) + 1e-8)
                similarity = np.dot(query_norm, cached_norm)

                if similarity > best_similarity and similarity >= self.config.lsh_threshold:
                    best_similarity = similarity
                    best_vector_id = candidate_id

            if best_vector_id is not None:
                cached_embedding, access_count, _last_access = self._cache[best_vector_id]
                # Update access tracking
                self._cache[best_vector_id] = (
                    cached_embedding,
                    access_count + 1,
                    time.perf_counter(),
                )
                self._cache.move_to_end(best_vector_id)

                self.metrics.hits += 1
                access_time = (time.perf_counter() - start_time) * 1_000_000
                self.metrics.total_access_time_us += access_time
                logger.debug(
                    f"✅ L1 Cache LSH hit for '{query[:50]}...' (sim={best_similarity:.4f}, {access_time:.2f} µs)"
                )
                return best_vector_id, cached_embedding

            self.metrics.misses += 1
            access_time = (time.perf_counter() - start_time) * 1_000_000
            logger.debug(
                f"❌ L1 Cache pull miss for '{query[:50]}...' ({access_time:.2f} µs)"
            )
            return None, None

    def clear(self) -> None:
        """Clear the L1 cache, LSH, and text cache."""
        with self._lock:
            self._cache.clear()
            self._text_cache.clear()
            self._text_cache_lru.clear()
            # Clear LSH buckets
            for hash_idx in range(len(self._lsh_buckets)):
                for bucket_idx in range(self.config.lsh_bucket_size):
                    self._lsh_buckets[hash_idx][bucket_idx].clear()
            self._access_counts.clear()
            logger.info("🗑️  L1 cache cleared")

    def get_stats(self) -> dict[str, Any]:
        """Get L1 cache statistics."""
        # Count non-empty LSH buckets
        total_lsh_buckets = sum(
            sum(1 for bucket in buckets.values() if bucket)
            for buckets in self._lsh_buckets
        )

        return {
            "cache_size": len(self._cache),
            "max_vectors": self.config.max_vectors,
            "hits": self.metrics.hits,
            "misses": self.metrics.misses,
            "hit_rate": f"{self.metrics.hit_rate:.2f}%",
            "evictions": self.metrics.evictions,
            "warmups": self.metrics.warmups,
            "avg_access_us": f"{self.metrics.total_access_time_us / max(1, self.metrics.hits + self.metrics.misses):.2f}",
            "text_cache_size": len(self._text_cache),
            "text_cache_max_size": self.config.text_cache_max_size,
            "lsh_buckets_used": total_lsh_buckets,
            "lsh_total_buckets": len(self._lsh_buckets) * self.config.lsh_bucket_size,
        }


class VectorMemory:
    """
    Persistent Vector Memory using sqlite-vec + LLM embeddings.
    Optimized for sub-millisecond RAG retrieval (<1 ms target).

    Optimizations:
    1. L1 Cache for NPU/GPU - Ultra-light vector index in GPU memory
    2. Query cache - LRU cache for repeated queries (~0.001 ms)
    3. sqlite-vec with serialized float32 - Fast L2 search
    4. Pre-warmed connection - Persistent sqlite connection
    5. Explicit "pull" mechanism - Direct vector retrieval for 0.5B model
    """

    @dataclass
    class Config:
        """Configuration for VectorMemory instance."""

        db_path: str | None = None
        max_documents: int = 100
        embedding_dim: int = 768
        enable_query_cache: bool = True
        cache_max_size: int = 50
        enable_l1_cache: bool = True
        l1_cache_config: L1CacheConfig | None = None

    def __init__(
        self,
        config: Config | None = None,
    ) -> None:
        config = config or self.Config()

        self.max_documents = config.max_documents
        self.embedding_max_chars = int(os.getenv("EMBEDDING_MAX_CHARS", "8000"))
        self.vec_store_max_chars = int(os.getenv("VEC_STORE_MAX_CHARS", "10000"))
        self.db_path = str(Path(config.db_path).resolve()) if config.db_path else None
        self.vector_db_path = self._resolve_vector_db_path(config.db_path)

        # Local model state
        self._local_model: Any = None
        self.embedding_dim = config.embedding_dim

        # Query cache for sub-ms retrieval (optimization)
        self.enable_query_cache = config.enable_query_cache
        self.cache_max_size = config.cache_max_size
        self._query_cache: dict[
            str, tuple[list[float], float]
        ] = {}  # query -> (embedding, timestamp)
        self._cache_hits = 0
        self._cache_misses = 0

        # L1 Cache for NPU/GPU (NEW)
        self.enable_l1_cache = config.enable_l1_cache
        self.l1_cache: L1VectorCache | None = None
        if config.enable_l1_cache:
            self.l1_cache = L1VectorCache(
                config=config.l1_cache_config,
                embedding_dim=config.embedding_dim,
            )

        # Initialize components access denied check
        access_denied = self._init_local_model(config.embedding_dim)
        self._init_http_fallback(access_denied)

        # TF-IDF fallback state
        self.documents: list[dict[str, Any]] = []
        self.idf: dict[str, float] = {}
        self._conn: sqlite3.Connection | None = (
            None  # Persistent connection for :memory: fallback
        )
        self.use_sqlite_vec: bool = False
        # Try to initialize sqlite-vec
        if self.vector_db_path:
            self._init_sqlite_vec(self.vector_db_path)

    def _resolve_vector_db_path(self, db_path: str | None) -> str | None:
        """Resolve the path for the vector database."""
        if not db_path:
            return None
        p = Path(db_path).resolve()
        return str(p.parent / f"{p.stem}_vectors.db")

    def _init_local_model(self, default_dim: int) -> bool:
        """Initialize the local GGUF model if configured. Returns True if access was denied by sandbox."""
        embed_model_path_str = os.getenv("EMBEDDING_MODEL_PATH")
        if not embed_model_path_str:
            self.embedding_dim = int(os.getenv("EMBEDDING_DIM", str(default_dim)))
            return False

        model_path = Path(embed_model_path_str).resolve()
        logger.info(f"  🔍 Checking embedding model path: {model_path}")

        model_exists = False
        access_denied = False

        try:
            model_path.stat()
            model_exists = True
        except FileNotFoundError:
            logger.warning(f"  ⚠️ Embedding model file NOT found at: {model_path}")
        except PermissionError:
            logger.warning(
                f"  🔐 macOS Sandbox Restricted access to: {model_path.name}. "
                "This process does not have permission to read the model file, even though it exists. "
                "Local embeddings will be disabled for this process."
            )
            access_denied = True
        except Exception as e:
            logger.debug(f"  🔍 Handled non-critical error during model check: {e}")

        if _Llama and model_exists:
            try:
                self._local_model = _Llama(
                    model_path=str(model_path),
                    embedding=True,
                    n_ctx=2048,
                    verbose=False,
                )
                self.embedding_dim = self._local_model.n_embd()
                logger.info(
                    f"  🧠 Local embedding model loaded: {model_path.name} (dim={self.embedding_dim})"
                )
                return False
            except Exception as e:
                logger.error(f"  ❌ Failed to load local Llama model: {e}")

        # Fallback diagnostics if local model failed
        if not _Llama:
            logger.warning(
                "  ⚠️ llama-cpp-python not installed. Local embeddings disabled."
            )
        elif access_denied:
            logger.info(
                f"  🔐 Local embedding model is RESTRICTED by macOS Sandbox at {model_path}. Using HTTP/TF-IDF fallback."
            )
        else:
            logger.info(
                f"  ⚠️ Embedding model not found or inaccessible at {model_path}, using HTTP/TF-IDF fallback"
            )

        self.embedding_dim = int(os.getenv("EMBEDDING_DIM", str(default_dim)))
        return access_denied

    def _init_http_fallback(self, access_denied: bool) -> None:
        """Configure HTTP embedding fallback."""
        base_url = os.getenv("LLM_SERVER_URL")
        self.embedding_model_name = os.getenv("LLM_MODEL")
        self.embedding_url = f"{base_url}/embeddings" if base_url else None

    def _init_sqlite_vec(self, db_path: str) -> None:
        """Initialize sqlite-vec virtual table."""
        if not sqlite_vec:
            logger.warning("  ⚠️ sqlite-vec not installed. Vector search disabled.")
            return

        # Ensure the directory exists
        db_dir = Path(db_path).parent
        db_dir.mkdir(parents=True, exist_ok=True)

        # Clean up stale WAL files that may cause disk I/O errors
        # SQLite WAL files are: db-name-wal and db-name-shm
        wal_path = Path(db_path).with_suffix(".db-wal")
        shm_path = Path(db_path).with_suffix(".db-shm")

        # Also handle case where db_path is full path with .db extension
        wal_path_direct = Path(db_path + "-wal")
        shm_path_direct = Path(db_path + "-shm")

        for stale_file in [wal_path, shm_path, wal_path_direct, shm_path_direct]:
            if stale_file.exists():
                logger.warning(f"  ⚠️ Cleaning up stale WAL file: {stale_file.name}")
                stale_file.unlink()

        try:
            conn = sqlite3.connect(db_path)
            conn.enable_load_extension(True)
            sqlite_vec.load(conn)
        except sqlite3.OperationalError as e:
            logger.warning(
                f"  ⚠️ Vector store disk access failed ({e}), falling back to :memory:"
            )
            self.vector_db_path = ":memory:"
            conn = sqlite3.connect(":memory:")
            conn.enable_load_extension(True)
            sqlite_vec.load(conn)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS vec_documents (
                id INTEGER PRIMARY KEY,
                session_id INTEGER,
                text TEXT,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute(f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS vec_embeddings USING vec0(
                id INTEGER PRIMARY KEY,
                embedding float[{self.embedding_dim}]
            )
        """)
        conn.commit()
        if self.vector_db_path == ":memory:":
            self._conn = conn
        else:
            conn.close()
        self.use_sqlite_vec = True
        logger.info(f"  🧠 sqlite-vec initialized (dim={self.embedding_dim})")

    def _get_conn(self) -> sqlite3.Connection:
        """Get a sqlite connection with vec extension loaded."""
        if self._conn:
            return self._conn

        if self.vector_db_path is None:
            raise ValueError("vector_db_path must not be None")

        conn = sqlite3.connect(self.vector_db_path)
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)

        if self.vector_db_path == ":memory:":
            self._conn = conn
        return conn

    def get_embedding(self, text: str) -> list[float] | None:
        """Public entry point for hardware-accelerated embeddings.

        Args:
            text: Input text to embed.

        Returns:
            list[float] | None: The resulting embedding vector.
        """
        return self._get_embedding_cached(text)

    def _get_embedding_cached(self, text: str) -> list[float] | None:
        """
        Get embedding with query cache for sub-ms retrieval.

        This is the OPTIMIZED path for System 1 RAG:
        1. Check query cache (O(1) - ~0.001 ms)
        2. If cache miss, compute embedding with local model (~5-10 ms)
        3. Cache result for future identical queries

        Returns:
            list[float] | None: Embedding or None if unavailable
        """
        truncated = text[: self.embedding_max_chars]

        # Check cache first (optimization for repeated queries)
        if self.enable_query_cache and truncated in self._query_cache:
            embedding, _ = self._query_cache[truncated]
            self._cache_hits += 1
            return embedding

        self._cache_misses += 1

        # Compute embedding
        embedding_val = self._compute_embedding(truncated)
        if embedding_val is None:
            return None

        # Cache result if successful
        if self.enable_query_cache:
            # Evict oldest if cache full
            if len(self._query_cache) >= self.cache_max_size:
                # Remove first entry (FIFO)
                oldest_key = next(iter(self._query_cache))
                del self._query_cache[oldest_key]
            self._query_cache[truncated] = (embedding_val, time.time())

        return embedding_val

    def _compute_embedding(self, text: str) -> list[float] | None:
        """Compute embedding using local model (fastest path)."""
        # 1. In-process embedding (fast, no HTTP) - OPTIMIZED PATH
        if self._local_model:
            try:
                start_time = time.perf_counter()
                result = self._local_model.embed(text)
                embed_time = (time.perf_counter() - start_time) * 1000

                # Handle nested list format
                if result and isinstance(result[0], list):
                    embedding = cast(list[float], result[0])
                else:
                    embedding = cast(list[float], result)

                logger.debug(f"  ✅ Local embedding: {embed_time:.3f} ms")
                return embedding
            except Exception as e:
                logger.info(
                    f"  ⚠️ Local embedding model error: {e}. Falling back to HTTP/TF-IDF."
                )
                return None

        # 2. HTTP fallback (uses the main LLM server) - SLOW PATH
        if not self.embedding_url:
            logger.info("  ⚠️ No embedding URL configured. Using TF-IDF fallback.")
            return None

        # Use synchronous aiohttp call for non-async context
        embedding_result: list[float] | None = None
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                timeout = aiohttp.ClientTimeout(total=30)

                async def fetch_embedding() -> list[float] | None:
                    if self.embedding_url is None:
                        return None
                    async with (
                        aiohttp.ClientSession(timeout=timeout) as session,
                        session.post(
                            self.embedding_url,
                            headers={"Content-Type": "application/json"},
                            json={"model": self.embedding_model_name, "input": text},
                        ) as resp,
                    ):
                        if resp.status == HTTPStatus.OK:
                            data = await resp.json()
                            if "data" in data and len(data["data"]) > 0:
                                return data["data"][0]["embedding"]  # type: ignore[no-any-return]
                            logger.info(
                                f"  ⚠️ Invalid embedding response: {data}. Using TF-IDF fallback."
                            )
                        else:
                            error_text = await resp.text()
                            logger.info(
                                f"  ⚠️ Embedding API error ({resp.status}): {error_text[:200]}. Using TF-IDF fallback."
                            )
                        return None

                embedding_result = loop.run_until_complete(fetch_embedding())
            finally:
                loop.close()
        except aiohttp.ClientError as e:
            logger.info(f"  ⚠️ Embedding HTTP error: {e}. Using TF-IDF fallback.")

        return embedding_result

    async def add_document_async(
        self, doc_id: int, text: str, metadata: dict[str, Any] | None = None
    ) -> None:
        """Add a document with real embeddings (async version)."""
        if not self.use_sqlite_vec:
            self.add_document(doc_id, text, metadata)
            return

        # Use cached embedding for sub-ms retrieval
        embedding = self._get_embedding_cached(text)
        if not embedding:
            # Fallback to TF-IDF if embedding fails
            self.add_document(doc_id, text, metadata)
            return

        conn = self._get_conn()
        # Evict oldest if at capacity
        count = conn.execute("SELECT COUNT(*) FROM vec_documents").fetchone()[0]
        while count >= self.max_documents:
            oldest = conn.execute(
                "SELECT id FROM vec_documents ORDER BY created_at ASC LIMIT 1"
            ).fetchone()
            if oldest:
                conn.execute("DELETE FROM vec_documents WHERE id = ?", (oldest[0],))
                conn.execute("DELETE FROM vec_embeddings WHERE id = ?", (oldest[0],))
                count -= 1
            else:
                break

        # Insert document and embedding
        conn.execute(
            "INSERT OR REPLACE INTO vec_documents (id, session_id, text, metadata) VALUES (?, ?, ?, ?)",
            (
                doc_id,
                doc_id,
                text[: self.vec_store_max_chars],
                json.dumps(metadata or {}),
            ),
        )
        conn.execute(
            "INSERT OR REPLACE INTO vec_embeddings (id, embedding) VALUES (?, ?)",
            (doc_id, sqlite_vec.serialize_float32(embedding)),
        )
        conn.commit()
        conn.close()

        # Update L1 cache if enabled
        if self.l1_cache and isinstance(embedding, list):
            embedding_array = np.array(embedding, dtype=np.float32)
            self.l1_cache.add(doc_id, embedding_array, text)

    async def search_async(
        self, query: str, top_k: int = 3
    ) -> list[tuple[dict[str, Any], float]]:
        """
        Search with real embeddings (async version) - OPTIMIZED FOR SUB-MS.

        Latency budget for System 1 RAG:
        - Total: <1 ms (8% of 12 ms reflex loop)
        - Embedding: ~0.001 ms (cache hit) or ~5-10 ms (cache miss, one-time)
        - Search: <0.5 ms (sqlite-vec indexed search)

        Returns:
            list[tuple[dict, float]]: (document, similarity_score)
        """
        if not self.use_sqlite_vec:
            return self.search(query, top_k)

        # Use cached embedding for sub-ms retrieval
        embedding = self._get_embedding_cached(query)
        if not embedding:
            return self.search(query, top_k)

        # Check L1 cache first (explicit "pull" mechanism)
        if self.l1_cache:
            embedding_array = np.array(embedding, dtype=np.float32)
            vector_id = self.l1_cache.get_by_embedding(embedding_array)
            if vector_id is not None:
                # L1 cache hit - return cached document
                conn = self._get_conn()
                doc = conn.execute(
                    "SELECT id, text, metadata, session_id FROM vec_documents WHERE id = ?",
                    (vector_id,),
                ).fetchone()
                conn.close()
                if doc:
                    return [
                        (
                            {
                                "id": doc[0],
                                "text": doc[1],
                                "metadata": json.loads(doc[2]) if doc[2] else {},
                                "session_id": doc[3],
                            },
                            1.0,  # Perfect match from L1 cache
                        )
                    ]

        # Measure search latency
        start_time = time.perf_counter()

        conn = self._get_conn()
        rows = conn.execute(
            """
            SELECT v.id, v.distance, d.text, d.metadata, d.session_id
            FROM vec_embeddings v
            JOIN vec_documents d ON v.id = d.id
            WHERE v.embedding MATCH ?
              AND k = ?
            ORDER BY v.distance
        """,
            (sqlite_vec.serialize_float32(embedding), top_k),
        ).fetchall()

        search_time = (time.perf_counter() - start_time) * 1000
        logger.debug(f"  🔍 sqlite-vec search: {search_time:.3f} ms")

        results = []
        for row in rows:
            doc = {
                "id": row[0],
                "text": row[2],
                "metadata": json.loads(row[3]) if row[3] else {},
                "session_id": row[4],
            }
            # Convert L2 distance to similarity score (1 / (1 + distance))
            score = 1.0 / (1.0 + row[1])
            results.append((doc, score))

        conn.close()
        return results

    def get_cache_stats(self) -> dict[str, Any]:
        """Get query cache statistics."""
        total = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total * 100 if total > 0 else 0.0
        stats = {
            "query_cache": {
                "cache_size": len(self._query_cache),
                "cache_hits": self._cache_hits,
                "cache_misses": self._cache_misses,
                "hit_rate": f"{hit_rate:.2f}%",
            }
        }
        # Add L1 cache stats if enabled
        if self.l1_cache:
            stats["l1_cache"] = self.l1_cache.get_stats()
        return stats

    async def close(self) -> None:
        """Closes the vector memory connection."""
        if self._conn:
            self._conn.close()
            self._conn = None

    def get_vector_by_id(self, vector_id: int) -> np.ndarray | None:
        """
        Explicit "pull" mechanism for direct vector retrieval.

        This enables the 0.5B model to directly pull vectors from L1 cache
        without going through the full search pipeline.

        Returns:
            np.ndarray | None: Embedding or None if not in L1 cache
        """
        if self.l1_cache:
            return self.l1_cache.get(vector_id)
        return None

    def clear(self) -> None:
        """Clears all documents. Call between projects."""
        self.documents.clear()
        self.idf.clear()

        if self.use_sqlite_vec and self.vector_db_path:
            conn = self._get_conn()
            conn.execute("DELETE FROM vec_documents")
            conn.execute("DELETE FROM vec_embeddings")
            conn.commit()
            conn.close()

        # Clear L1 cache
        if self.l1_cache:
            self.l1_cache.clear()

    def remove_document(self, doc_id: int) -> None:
        """Removes a specific document by its ID."""
        self.documents = [d for d in self.documents if d["id"] != doc_id]
        if self.documents:
            self._calculate_idf()
        else:
            self.idf = {}

        if self.use_sqlite_vec and self.vector_db_path:
            conn = self._get_conn()
            conn.execute("DELETE FROM vec_documents WHERE id = ?", (doc_id,))
            conn.execute("DELETE FROM vec_embeddings WHERE id = ?", (doc_id,))
            conn.commit()
            conn.close()

        # Remove from L1 cache
        if self.l1_cache:
            self.l1_cache.clear()

    # === TF-IDF FALLBACK (synchronous, in-memory) ===

    def add_document(
        self, doc_id: int, text: str, metadata: dict[str, Any] | None = None
    ) -> None:
        """Adds a document using TF-IDF fallback. Evicts oldest if at capacity."""
        while len(self.documents) >= self.max_documents:
            self.documents.pop(0)

        tokens = self._tokenize(text)
        self.documents.append(
            {"id": doc_id, "text": text, "tokens": tokens, "metadata": metadata or {}}
        )
        self._calculate_idf()

    def search(self, query: str, top_k: int = 3) -> list[tuple[dict[str, Any], float]]:
        """Performs cosine similarity search using TF-IDF."""
        if not Path(".git").exists():
            return []

        query_tokens = self._tokenize(query)
        query_vector = self._get_tfidf_vector(query_tokens)

        results = []
        for doc in self.documents:
            doc_vector = self._get_tfidf_vector(doc["tokens"])
            score = self._cosine_similarity(query_vector, doc_vector)
            if score > 0:
                results.append((doc, score))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def _tokenize(self, text: str) -> list[str]:
        return re.findall(r"\w+", text.lower())

    def _calculate_idf(self) -> None:
        n_docs = len(self.documents)
        all_tokens = set()
        for doc in self.documents:
            all_tokens.update(doc["tokens"])

        self.idf = {}
        for token in all_tokens:
            count = sum(1 for doc in self.documents if token in doc["tokens"])
            self.idf[token] = math.log(n_docs / (1 + count))

    def _get_tfidf_vector(self, tokens: list[str]) -> dict[str, float]:
        tf = Counter(tokens)
        total_tokens = len(tokens)
        vector = {}
        for token, count in tf.items():
            if token in self.idf:
                vector[token] = (count / total_tokens) * self.idf[token]
        return vector

    def _cosine_similarity(self, v1: dict[str, float], v2: dict[str, float]) -> float:
        dot_product = sum(
            v1.get(token, 0) * v2.get(token, 0) for token in set(v1) | set(v2)
        )
        mag1 = math.sqrt(sum(val**2 for val in v1.values()))
        mag2 = math.sqrt(sum(val**2 for val in v2.values()))

        if mag1 == 0 or mag2 == 0:
            return 0.0
        return dot_product / (mag1 * mag2)
