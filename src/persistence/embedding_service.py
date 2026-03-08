"""Embedding service for computing vector embeddings.

This module provides a service for computing embeddings at the source
during memory insertion, ensuring that embeddings are always available
and avoiding the cost of recomputation.
"""

import hashlib
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    SentenceTransformer = None


class EmbeddingService:
    """Service for computing and managing embeddings.

    This class provides methods for computing embeddings using various
    models and for generating reconciliation keys to prevent duplicate
    embeddings.

    Key features:
    - Embedding computation at source during insertion
    - Reconciliation key generation for deduplication
    - Support for multiple embedding models
    - Caching for repeated computations
    """

    def __init__(
        self,
        model_id: str = "BAAI/bge-m3",
        use_local_fallback: bool = True,
    ):
        """Initialize the embedding service.

        Args:
            model_id: Hugging Face model ID for embeddings.
            use_local_fallback: If True, use a simple hash-based fallback
                when sentence_transformers is not available or fails to load.
        """
        self.model_id = model_id
        self._cache: dict[str, list[float]] = {}
        self._model: SentenceTransformer | None = None
        self._embedding_dim = 1024  # Default for BGE-M3
        self._available = SENTENCE_TRANSFORMERS_AVAILABLE
        self._use_local_fallback = use_local_fallback

    def _get_model(self) -> SentenceTransformer:
        """Get or initialize the embedding model.

        Returns:
            The SentenceTransformer model instance.

        Raises:
            RuntimeError: If the model fails to load.
        """
        if not self._available:
            raise RuntimeError(
                "sentence_transformers is not installed. "
                "Install with: pip install sentence-transformers"
            )
        if self._model is None:
            try:
                logging.info(f"Loading embedding model: {self.model_id}")
                self._model = SentenceTransformer(self.model_id)
                self._embedding_dim = self._model.get_sentence_embedding_dimension()
            except Exception as e:
                logging.error(f"Failed to load embedding model: {e}")
                raise RuntimeError(f"Failed to load embedding model: {e}") from e
        return self._model

    def _generate_chunk_key(
        self,
        text: str,
        chunk_start: int = 0,
        chunk_end: int | None = None,
    ) -> str:
        """Generate a reconciliation key for deduplication.

        Args:
            text: The text content.
            chunk_start: Start position of the chunk.
            chunk_end: End position of the chunk.

        Returns:
            A 16-character hexadecimal hash.
        """
        if chunk_end is None:
            chunk_end = len(text)

        chunk_text = text[chunk_start:chunk_end]
        key_string = f"{chunk_text}"
        hash_object = hashlib.sha256(key_string.encode())
        return hash_object.hexdigest()[:16]

    def compute_embedding(self, text: str) -> list[float]:
        """Compute an embedding for the given text.

        This method computes an embedding using the configured model
        and caches the result for repeated computations.

        Args:
            text: The text to compute an embedding for.

        Returns:
            The vector embedding as a list of floats.
        """
        # Check cache first
        if text in self._cache:
            return self._cache[text]

        # Use local fallback if sentence_transformers is not available
        if not self._available and self._use_local_fallback:
            embedding = self._compute_local_embedding(text)
            self._cache[text] = embedding
            return embedding

        try:
            model = self._get_model()
            embedding_array = model.encode(text, convert_to_numpy=True)
            embedding = list(map(float, embedding_array))
            self._cache[text] = embedding
            return embedding
        except Exception as e:
            logging.error(f"Failed to compute embedding for text: {e}")
            # Fallback to local embedding on error
            if self._use_local_fallback:
                embedding = self._compute_local_embedding(text)
                self._cache[text] = embedding
                return embedding
            # Fallback to zero embedding on error
            embedding = [0.0] * self._embedding_dim
            self._cache[text] = embedding
            return embedding

    def _compute_local_embedding(self, text: str) -> list[float]:
        """Compute a simple hash-based embedding locally.

        This is a fallback when sentence_transformers is not available.
        It generates a deterministic embedding based on the text hash.

        Args:
            text: The text to compute an embedding for.

        Returns:
            A deterministic embedding vector.
        """
        # Generate a deterministic embedding based on text hash
        hash_value = int(hashlib.sha256(text.encode()).hexdigest(), 16)
        embedding = []
        for i in range(self._embedding_dim):
            # Use modular arithmetic to generate values in [-1, 1]
            value = ((hash_value >> (i % 64)) & 0xFFFFFFFF) / 0x100000000 - 0.5
            embedding.append(value * 2.0)  # Scale to [-1, 1]
        return embedding

    def compute_batch_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Compute embeddings for multiple texts.

        Args:
            texts: List of texts to compute embeddings for.

        Returns:
            List of vector embeddings.
        """
        return [self.compute_embedding(text) for text in texts]

    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        self._cache.clear()

    def set_embedding_dimension(self, dimension: int) -> None:
        """Set the embedding dimension.

        Args:
            dimension: The embedding dimension.
        """
        self._embedding_dim = dimension
