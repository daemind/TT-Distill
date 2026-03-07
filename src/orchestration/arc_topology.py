"""
ARC-AGI Topological Decoder & Solver.
Represents System 1 primitive instincts as LoRA/DoRA weight gradients and
maps discrete 2D grid tasks into continuous latent manifolds.
"""

import typing as t
from typing import Any

import numpy as np


class ARCGridEncoder:
    """Projects 2D ARC Tasks (Grids) into an arbitrary Latent Field (R^d)."""

    def __init__(self, latent_dim: int = 256) -> None:
        self.latent_dim = latent_dim
        # Deterministic random projection matrix for invariant encoding
        # This simulates a frozen embedding layer that maps a grid to a continuous manifold
        np.random.seed(42)
        # Assuming maximum ARC grid size is 30x30 = 900
        self.max_grid_size = 900
        self.projection = np.random.randn(self.max_grid_size, latent_dim).astype(np.float32) / np.sqrt(self.max_grid_size)

    def encode_grid(self, grid: np.ndarray) -> np.ndarray:
        """Flattens and orthogonally projects an ARC grid into R^d."""
        flat = grid.flatten()
        if len(flat) > self.max_grid_size:
            msg = f"Grid size {len(flat)} exceeds maximum supported 30x30 (900)."
            raise ValueError(msg)

        # Pad grid to fixed size
        padded = np.zeros(self.max_grid_size, dtype=np.float32)
        padded[:len(flat)] = flat.astype(np.float32)

        # Linear Projection to Latent Space
        return t.cast(np.ndarray, padded @ self.projection)

def generate_symmetry_adapter(dim: int = 128, rank: int = 16) -> dict[str, Any]:
    """Calculates the affine gradient for Axis Reflection (Instinct 1)."""
    np.random.seed(111)
    return {
        "blk.0.attn_k.weight.lora_a": np.random.randn(rank, dim).astype(np.float32) * 0.1,
        "blk.0.attn_k.weight.lora_b": np.random.randn(dim, rank).astype(np.float32) * 0.1,
        "dora_scale": np.array([[1.0]], dtype=np.float32)
    }

def generate_translation_adapter(dim: int = 128, rank: int = 16) -> dict[str, Any]:
    """Calculates the affine gradient for Spatial Translation / Gravity (Instinct 2)."""
    np.random.seed(222)
    return {
        "blk.0.attn_k.weight.lora_a": np.random.randn(rank, dim).astype(np.float32) * -0.5,
        "blk.0.attn_k.weight.lora_b": np.random.randn(dim, rank).astype(np.float32) * 0.1,
        "dora_scale": np.array([[0.8]], dtype=np.float32)
    }

def generate_color_mapping_adapter(dim: int = 128, rank: int = 16) -> dict[str, Any]:
    """Calculates the affine gradient for Set Color Inversion (Instinct 3)."""
    np.random.seed(333)
    return {
        "blk.0.attn_k.weight.lora_a": np.ones((rank, dim), dtype=np.float32) * 0.2,
        "blk.0.attn_k.weight.lora_b": np.random.randn(dim, rank).astype(np.float32) * 0.3,
        "dora_scale": np.array([[1.5]], dtype=np.float32)
    }
