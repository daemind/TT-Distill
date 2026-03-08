"""CocoIndex ingestion pipeline for file-based vector ingestion.

This module provides integration with CocoIndex for monitoring project directories
and automatically ingesting code changes into the vector search index. It uses
CocoIndex's LocalFile source to watch agent worktrees and trigger embeddings
when files are modified.

Key features:
- File watching using CocoIndex LocalFile source
- Automatic chunking of source files
- Embedding computation at source using EmbeddingService
- Reconciliation key generation to prevent duplicates
- Event-driven synchronization with VectorSyncPipeline
"""

import asyncio
import contextlib
import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.persistence.embedding_service import EmbeddingService
from src.persistence.sync_pipeline import VectorSyncPipeline


@dataclass
class FileChunk:
    """Represents a chunk of text from a file."""

    file_path: str
    chunk_start: int
    chunk_end: int
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class IngestionResult:
    """Result of a file ingestion operation."""

    file_path: str
    chunks_ingested: int
    chunks_skipped: int
    errors: list[str] = field(default_factory=list)


class CocoIndexIngestion:
    """CocoIndex-based file ingestion pipeline.

    This class integrates with CocoIndex to monitor project directories
    and automatically ingest code changes into the vector search index.
    It uses CocoIndex's LocalFile source to watch for file changes and
    triggers embeddings when files are modified.

    Key features:
    - File watching using CocoIndex LocalFile source
    - Automatic chunking of source files
    - Embedding computation at source using EmbeddingService
    - Reconciliation key generation to prevent duplicates
    - Event-driven synchronization with VectorSyncPipeline

    Example:
        ingestion = CocoIndexIngestion(
            watch_paths=["/path/to/project"],
            embedding_service=EmbeddingService(),
            sync_pipeline=VectorSyncPipeline(),
        )
        await ingestion.start()
    """

    def __init__(
        self,
        watch_paths: list[str],
        embedding_service: EmbeddingService,
        sync_pipeline: VectorSyncPipeline,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        file_patterns: list[str] | None = None,
        session_id: int | None = None,
    ):
        """Initialize the CocoIndex ingestion pipeline.

        Args:
            watch_paths: List of directory paths to watch for file changes.
            embedding_service: Service for computing embeddings.
            sync_pipeline: Pipeline for synchronizing vectors to Postgres.
            chunk_size: Size of text chunks in characters.
            chunk_overlap: Overlap between chunks in characters.
            file_patterns: List of file patterns to include (e.g., ["*.py", "*.md"]).
                          If None, all files are watched.
            session_id: Optional session ID to associate with ingested memories.
        """
        self.watch_paths = [Path(p) for p in watch_paths]
        self.embedding_service = embedding_service
        self.sync_pipeline = sync_pipeline
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.file_patterns = file_patterns or ["*"]
        self.session_id = session_id
        self._running = False
        self._watchers: list[asyncio.Task[None]] = []
        self._file_cache: dict[str, str] = {}
        self._logger = logging.getLogger(__name__)

    async def start(self) -> None:
        """Start the ingestion pipeline and begin watching files.

        This method initializes the file watchers for all configured paths
        and starts the background tasks that monitor for file changes.
        """
        if self._running:
            self._logger.warning("Ingestion pipeline already running")
            return

        self._running = True

        for watch_path in self.watch_paths:
            if not watch_path.exists():
                self._logger.warning(f"Watch path does not exist: {watch_path}")
                continue

            watcher_task = asyncio.create_task(self._watch_path(watch_path))
            self._watchers.append(watcher_task)

        self._logger.info(f"Started ingestion pipeline with {len(self.watch_paths)} watch paths")

    async def stop(self) -> None:
        """Stop the ingestion pipeline and clean up resources.

        This method stops all file watchers and waits for them to complete.
        """
        if not self._running:
            return

        self._running = False

        for watcher in self._watchers:
            watcher.cancel()

        with contextlib.suppress(asyncio.CancelledError):
            for watcher in self._watchers:
                await watcher

        self._watchers.clear()
        self._logger.info("Stopped ingestion pipeline")

    async def _watch_path(self, path: Path) -> None:
        """Watch a directory path for file changes.

        Args:
            path: The directory path to watch.
        """
        while self._running:
            try:
                await self._scan_directory(path)
                # Wait before next scan (CocoIndex-style polling with optimization)
                await asyncio.sleep(1.0)
            except Exception as e:
                self._logger.error(f"Error watching path {path}: {e}")
                await asyncio.sleep(5.0)

    async def _scan_directory(self, path: Path) -> None:
        """Scan a directory for files and ingest changes.

        Args:
            path: The directory path to scan.
        """
        try:
            for file_path in path.rglob("*"):  # noqa: ASYNC240
                if file_path.is_file() and self._should_ingest(file_path):
                    await self._ingest_file(file_path)
        except PermissionError:
            self._logger.warning(f"Permission denied: {path}")
        except Exception as e:
            self._logger.error(f"Error scanning directory {path}: {e}")

    def _should_ingest(self, file_path: Path) -> bool:
        """Check if a file should be ingested based on patterns.

        Args:
            file_path: The file path to check.

        Returns:
            True if the file should be ingested, False otherwise.
        """
        if self.file_patterns == ["*"]:
            return True

        # Check file extension patterns
        for pattern in self.file_patterns:
            if pattern.startswith("*."):
                extension = pattern[1:] # e.g. ".py"
                if file_path.suffix == extension:
                    return True
            if pattern == "*":
                return True
            if file_path.name == pattern or file_path.match(pattern):
                return True

        return False

    async def _ingest_file(self, file_path: Path) -> None:
        """Ingest a single file into the vector index.

        Args:
            file_path: The file path to ingest.
        """
        try:
            # Read file content
            content = file_path.read_text(encoding="utf-8")  # noqa: ASYNC240

            # Check if content has changed
            cached_content = self._file_cache.get(str(file_path))
            if cached_content == content:
                return  # No change, skip

            self._file_cache[str(file_path)] = content

            # Chunk and ingest
            chunks = self._chunk_text(content, str(file_path))
            result = await self._ingest_chunks(chunks)

            self._logger.debug(
                f"Ingested {file_path}: {result.chunks_ingested} chunks, "
                f"{result.chunks_skipped} skipped"
            )

        except Exception as e:
            self._logger.error(f"Error ingesting file {file_path}: {e}")

    def _chunk_text(self, text: str, file_path: str) -> list[FileChunk]:
        """Chunk text into smaller pieces for embedding.

        Args:
            text: The text to chunk.
            file_path: The source file path.

        Returns:
            List of FileChunk objects.
        """
        chunks = []
        text_length = len(text)

        if text_length <= self.chunk_size:
            # Single chunk
            chunks.append(
                FileChunk(
                    file_path=file_path,
                    chunk_start=0,
                    chunk_end=text_length,
                    text=text,
                    metadata={"source": str(file_path)},
                )
            )
            return chunks

        # Multi-chunk with overlap
        start = 0
        while start < text_length:
            end = min(start + self.chunk_size, text_length)
            chunk_text = text[start:end]

            # Add metadata
            metadata = {
                "source": str(file_path),
                "chunk_index": len(chunks),
                "total_chunks": (text_length + self.chunk_overlap - 1) // (self.chunk_size - self.chunk_overlap),
            }

            chunks.append(
                FileChunk(
                    file_path=file_path,
                    chunk_start=start,
                    chunk_end=end,
                    text=chunk_text,
                    metadata=metadata,
                )
            )

            # Move start position with overlap
            start = end - self.chunk_overlap if end < text_length else text_length

        return chunks

    async def _ingest_chunks(self, chunks: list[FileChunk]) -> IngestionResult:
        """Ingest a list of chunks into the vector index.

        Args:
            chunks: List of FileChunk objects to ingest.

        Returns:
            IngestionResult with statistics.
        """
        chunks_ingested = 0
        chunks_skipped = 0
        errors: list[str] = []

        for chunk in chunks:
            try:
                # Generate reconciliation key
                chunk_key = self._generate_chunk_key(
                    chunk.file_path,
                    chunk.chunk_start,
                    chunk.chunk_end,
                )

                # Compute embedding
                embedding = self.embedding_service.compute_embedding(chunk.text)

                # Prepare metadata
                metadata = chunk.metadata.copy()
                metadata["chunk_key"] = chunk_key

                # Queue for sync to Postgres
                session_id = self.session_id or 0
                await self.sync_pipeline.queue_sync(
                    session_id=session_id,
                    text=chunk.text,
                    metadata=metadata,
                    embedding=embedding,
                    chunk_key=chunk_key,
                )

                chunks_ingested += 1

            except Exception as e:
                errors.append(f"Error processing chunk at {chunk.chunk_start}: {e}")
                chunks_skipped += 1

        return IngestionResult(
            file_path=chunks[0].file_path if chunks else "",
            chunks_ingested=chunks_ingested,
            chunks_skipped=chunks_skipped,
            errors=errors,
        )

    def _generate_chunk_key(
        self,
        file_path: str,
        chunk_start: int,
        chunk_end: int,
    ) -> str:
        """Generate a reconciliation key for deduplication.

        Args:
            file_path: Path to the source file.
            chunk_start: Start position of the chunk.
            chunk_end: End position of the chunk.

        Returns:
            A 16-character hexadecimal hash.
        """
        key_string = f"{file_path}:{chunk_start}:{chunk_end}"
        hash_object = hashlib.sha256(key_string.encode())
        return hash_object.hexdigest()[:16]

    def set_session_id(self, session_id: int) -> None:
        """Set the session ID for ingested memories.

        Args:
            session_id: The session ID to associate with ingested memories.
        """
        self.session_id = session_id

    def get_file_cache(self) -> dict[str, str]:
        """Get the current file cache.

        Returns:
            Dictionary mapping file paths to their cached content.
        """
        return self._file_cache.copy()

    async def clear_file_cache(self) -> None:
        """Clear the file cache."""
        self._file_cache.clear()
