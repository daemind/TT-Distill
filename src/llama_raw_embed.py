"""Low-level llama.cpp bindings for direct tensor injection.

This module provides direct access to the llama.cpp C API via ctypes,
allowing raw embedding tensor injection without tokenization.

Reference: https://github.com/ggml-org/llama.cpp/blob/master/include/llama.h

The llama_batch structure allows passing:
- token: token IDs (used when embd is NULL)
- embd: token embeddings (float vector of size n_embd) (used when token is NULL)

By setting token=NULL and providing embd, we bypass the tokenizer entirely.

CRITICAL: For causal autoregressive Decoder models (like Qwen), llama_decode
is the correct function to update the KV Cache with injected latent tensors.
llama_encode is designed for encoder-only models (like BERT).

The key insight is that llama_decode with embd != NULL and token == NULL
allows zero-copy injection of raw embedding tensors directly into the model's
latent space, updating the KV Cache for subsequent autoregressive generation.
"""

from __future__ import annotations

import ctypes
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from typing import Any  # noqa: F401

try:
    import llama_cpp
    import llama_cpp.llama_cpp as llama_cpp_module
except ImportError:
    llama_cpp = None  # type: ignore[assignment]
    llama_cpp_module = None  # type: ignore[assignment]


class LlamaRawEmbed:
    """Low-level llama.cpp interface for direct embedding injection.

    This class provides direct access to llama.cpp's C API via ctypes,
    allowing raw tensor injection without tokenization.

    CRITICAL: For causal autoregressive Decoder models, llama_decode is the
    correct function to update the KV Cache with injected latent tensors.
    """

    def __init__(self, model_path: str, n_ctx: int = 128, n_gpu_layers: int = 0):
        """Initialize the raw llama interface.

        Args:
            model_path: Path to the GGUF model file
            n_ctx: Context size for the llama context
            n_gpu_layers: Number of layers to offload to GPU
        """
        if llama_cpp is None:
            raise ImportError(
                "Le module 'llama-cpp-python' est requis pour LlamaRawEmbed. "
                "Veuillez l'installer via 'pip install llama-cpp-python' ou 'uv sync'."
            )

        # Use the high-level Llama class with embedding=True
        # This is required for embeddings mode
        self.llama = llama_cpp.Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            embedding=True,  # CRITICAL: Enable embedding mode
            verbose=False,  # CRITICAL: Disable spammy C++ logs
        )

        # Get model info
        self.n_embd = self.llama.n_embd()
        self.n_vocab = self.llama.n_vocab()
        self.n_ctx = self.llama.n_ctx()

    def embed_tensor(self, embedding_tensor: np.ndarray) -> np.ndarray:
        """Run inference with direct embedding tensor injection.

        This method bypasses tokenization by passing raw embeddings directly
        to the model via the llama_batch.embd pointer.

        CRITICAL: llama_decode expects llama_batch* (pointer to struct)
        and is the correct function for causal autoregressive Decoder models.

        Args:
            embedding_tensor: numpy array of shape (n_tokens, n_embd) containing
                             the raw embedding values to inject

        Returns:
            numpy array of shape (n_embd,) containing the sequence embedding
            (mean of all token embeddings)
        """

        n_tokens = embedding_tensor.shape[0]
        if embedding_tensor.shape[1] != self.n_embd:
            raise ValueError(
                f"Embedding dimension mismatch: {embedding_tensor.shape[1]} != {self.n_embd}"
            )

        # 1. Extraction de la structure C brute (le vrai llama_batch)
        batch_wrapper = self.llama._batch
        # llama_cpp_python stocke la structure sous .batch ou ._batch selon les versions
        raw_batch = getattr(batch_wrapper, "batch", None) or getattr(
            batch_wrapper, "_batch", None
        )

        if raw_batch is None:
            raise RuntimeError(
                "Impossible de trouver la structure C llama_batch interne."
            )

        raw_batch.n_tokens = n_tokens

        # 2. INJECTION ZÉRO-COPIE (Pointeur direct vers Numpy)
        self._flat_tensor = embedding_tensor.astype(np.float32).flatten()

        # On branche l'entrée embd du C++ directement sur la RAM de Python
        # IMPORTANT: llama.cpp requires EITHER token OR embd to be set, NOT both
        # When using embd mode, token must be NULL/0
        raw_batch.embd = ctypes.cast(
            self._flat_tensor.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.POINTER(ctypes.c_float),
        )
        # Set token to NULL pointer to indicate we're using embedding mode
        # Use ctypes.c_void_p(0) for NULL pointer instead of c_int
        raw_batch.token = ctypes.cast(
            ctypes.c_void_p(0), ctypes.POINTER(llama_cpp_module.llama_token)
        )

        # 3. Configuration des métadonnées temporelles
        for i in range(n_tokens):
            raw_batch.pos[i] = i
            raw_batch.n_seq_id[i] = 1
            raw_batch.seq_id[i][0] = 0
            # Mark the token as an output token to prevent "embeddings required but some input tokens were not marked as outputs" warning
            raw_batch.logits[i] = 1

        # 4. DÉCODAGE (Inférence) - llama_decode est la fonction correcte pour Decoder
        # llama_decode met à jour le KV Cache avec les embeddings injectés
        result = llama_cpp_module.llama_decode(self.llama.ctx, raw_batch)
        if result != 0:
            raise RuntimeError(f"llama_decode failed with code {result}")

        # 5. Extraction du tenseur de sortie (Latent State)
        # llama_get_embeddings retourne les embeddings de séquence (poolés)
        embeddings_ptr = llama_cpp_module.llama_get_embeddings(self.llama.ctx)
        if not embeddings_ptr:
            raise RuntimeError(
                "Le pointeur d'embedding est NULL. Le contexte est-il en mode embedding=True ?"
            )

        embeddings = np.ctypeslib.as_array(
            ctypes.cast(embeddings_ptr, ctypes.POINTER(ctypes.c_float)),
            shape=(self.n_embd,),
        )

        output_tensor = embeddings.copy()

        # 6. Nettoyage de sécurité : on débranche le pointeur pour éviter un double-free
        # par llama-cpp-python lors du cleanup
        raw_batch.embd = ctypes.cast(ctypes.c_void_p(0), ctypes.POINTER(ctypes.c_float))
        raw_batch.token = ctypes.cast(
            ctypes.c_void_p(0), ctypes.POINTER(llama_cpp_module.llama_token)
        )

        return output_tensor

    def get_embedding_dimension(self) -> int:
        """Get the embedding dimension of the model."""
        return self.n_embd

    def cleanup(self) -> None:
        """Clean up resources."""
        if hasattr(self, "llama"):
            del self.llama


def embed_with_raw_llama(
    model_path: str,
    embedding_tensor: np.ndarray,
    n_ctx: int = 128,
) -> np.ndarray:
    """Convenience function to run inference with direct embedding injection.

    Args:
        model_path: Path to the GGUF model file
        embedding_tensor: numpy array of shape (n_tokens, n_embd)
        n_ctx: Context size

    Returns:
        numpy array of shape (n_embd,) containing the sequence embedding
    """
    raw_llama = LlamaRawEmbed(model_path, n_ctx=n_ctx)
    try:
        return raw_llama.embed_tensor(embedding_tensor)
    finally:
        raw_llama.cleanup()
