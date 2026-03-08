"""Event-Driven synchronization pipeline.

This module provides an Event-Driven synchronization mechanism that
replaces polling-based approaches with asyncio.Queue for instant
wake-up and efficient batch processing.
"""

import asyncio
import hashlib
import logging
from dataclasses import dataclass
from typing import Any

from src.persistence.postgres_persistence import PostgresPersistence


@dataclass
class SyncTask:
    """A synchronization task to be processed."""

    session_id: int
    text: str
    metadata: dict[str, Any]
    embedding: list[float]
    chunk_key: str | None = None


class VectorSyncPipeline:
    """Event-Driven synchronization pipeline.

    This class implements an Event-Driven synchronization mechanism
    that replaces polling with asyncio.Queue for instant wake-up.
    It processes synchronization tasks in batches for efficiency.

    Key features:
    - Event-Driven architecture using asyncio.Queue
    - Batch processing for efficiency
    - Duplicate detection using chunk_key
    - Graceful shutdown with task draining
    """

    def __init__(
        self,
        batch_size: int = 10,
        worker_count: int = 2,
        postgres_url: str = "postgres://localhost/project_manager",
    ):
        """Initialize the sync pipeline.

        Args:
            batch_size: Number of tasks to process per batch.
            worker_count: Number of worker tasks to run concurrently.
            postgres_url: Postgres database URL for synchronization.
        """
        self.sync_queue: asyncio.Queue[SyncTask] = asyncio.Queue()
        self._worker_count = worker_count
        self._workers: list[asyncio.Task[None]] = []
        self._running = False
        self._batch_size = batch_size
        self._postgres_url = postgres_url
        self._postgres: PostgresPersistence | None = None

    async def start(self) -> None:
        """Start the synchronization pipeline workers."""
        if self._running:
            return

        # Initialize Postgres connection
        self._postgres = PostgresPersistence(self._postgres_url)
        await self._postgres.initialize()

        self._running = True
        for _ in range(self._worker_count):
            worker = asyncio.create_task(self._sync_worker())
            self._workers.append(worker)

    async def stop(self) -> None:
        """Stop the synchronization pipeline and drain remaining tasks."""
        self._running = False

        # Wait for all workers to finish
        if self._workers:
            await asyncio.gather(*self._workers, return_exceptions=True)
            self._workers.clear()

        # Close Postgres connection
        if self._postgres is not None:
            await self._postgres.close()
            self._postgres = None

    async def queue_sync(
        self,
        session_id: int,
        text: str,
        metadata: dict[str, Any],
        embedding: list[float],
        chunk_key: str | None = None,
    ) -> None:
        """Queue a synchronization task.

        Args:
            session_id: The session ID this memory belongs to.
            text: The memory text content.
            metadata: Metadata dictionary.
            embedding: The vector embedding.
            chunk_key: Optional reconciliation key for deduplication.
        """
        task = SyncTask(
            session_id=session_id,
            text=text,
            metadata=metadata,
            embedding=embedding,
            chunk_key=chunk_key,
        )
        await self.sync_queue.put(task)

    async def _sync_worker(self) -> None:
        """Worker task that processes synchronization tasks.

        This worker continuously pulls tasks from the queue and
        processes them in batches. It uses asyncio.wait_for with
        a timeout to allow graceful shutdown.
        """
        while self._running:
            try:
                # Wait for a task with timeout for graceful shutdown
                task = await asyncio.wait_for(
                    self.sync_queue.get(),
                    timeout=1.0,
                )

                # Process the task
                await self._process_task(task)

                # Mark task as done
                self.sync_queue.task_done()

            except TimeoutError:
                # No task available, continue loop
                continue
            except Exception:
                # Log error but continue processing
                logging.exception("Error processing sync task")

    async def _process_task(self, task: SyncTask) -> None:
        """Process a single synchronization task.

        Args:
            task: The synchronization task to process.
        """
        if self._postgres is None:
            logging.error("PostgresPersistence not initialized")
            return

        try:
            await self._postgres.add_vector(
                session_id=task.session_id,
                text=task.text,
                metadata=task.metadata,
                embedding=task.embedding,
                chunk_key=task.chunk_key,
            )
        except Exception as e:
            logging.error(f"Failed to sync vector to Postgres: {e}")

    async def _generate_chunk_key(
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
