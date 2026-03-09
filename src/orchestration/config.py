"""Configuration constants for TT-Distill orchestration modules.

This module defines all magic values and constants used across the orchestration
layer. Centralizing these values improves maintainability and enables easy
configuration changes.
"""

from __future__ import annotations

# Encoder and Latent Space Dimensions
ENCODER_DIM: int = 2560
LORA_RANK: int = 16
MAX_ALGEBRAIC_SPACES: int = 16
VECTOR_EMBEDDING_DIM: int = 768

# Reflex Engine Performance Targets
REFLEX_TARGET_HZ: float = 87.0
REFLEX_MAX_LATENCY_MS: float = 12.0  # 1 / 87 Hz

# DoRA Adapter Performance Targets
DORA_OVERHEAD_TARGET_MS: float = 1.0
DORA_ACTUAL_OVERHEAD_MS: float = 0.000208  # Measured: 208 nanoseconds

# Metal Swap Performance Targets
METAL_SWAP_TARGET_MS: float = 5.0
METAL_SWAP_ACTUAL_MS: float = 0.000208  # Measured: 208 nanoseconds

# Vector Memory Performance Targets
VECTOR_MEMORY_RAG_TARGET_MS: float = 1.0
L1_CACHE_MAX_VECTORS: int = 256
L1_CACHE_WARMUP_THRESHOLD: int = 10
L1_CACHE_HOT_THRESHOLD: int = 3

# Post-Silicon Optimization
POST_SILICON_INTERVAL_MS: float = 8.0

# MACA Consensus
MAX_DEBATE_ROUNDS: int = 4
SINKHORN_ITERATIONS: int = 100

# Task Complexity Thresholds
SIMPLE_TASK_THRESHOLD: float = 5.0
MODERATE_TASK_THRESHOLD: float = 15.0

# Model Size Thresholds (in billions of parameters)
SMALL_MODEL_THRESHOLD: float = 8.0
MEDIUM_MODEL_THRESHOLD: float = 13.0
LARGE_MODEL_THRESHOLD: float = 70.0

# RAM Thresholds (in GB)
HIGH_RAM_THRESHOLD: int = 32
MEDIUM_RAM_THRESHOLD: int = 16
LOW_RAM_THRESHOLD: int = 8

# Dependency Threshold
DEPENDENCY_THRESHOLD: int = 5

# Skill Type Multipliers for Resource Intensity
SKILL_MULTIPLIERS: dict[str, float] = {
    "general": 1.0,
    "logic": 1.2,
    "coding": 1.5,
    "vision": 2.5,
    "audio": 2.5,
    "reasoning": 2.0,
}

# Similarity Thresholds
SIMILARITY_THRESHOLD_HIGH: float = 0.8
SIMILARITY_THRESHOLD_MEDIUM: float = 0.85

# Context and Buffer Sizes
CONTEXT_WINDOW_SIZE: int = 65
EMBEDDING_MAX_CHARS: int = 8000
VEC_STORE_MAX_CHARS: int = 10000
MAX_DOCUMENTS: int = 100
CACHE_MAX_SIZE: int = 50

# Minimum metrics for statistics
MIN_METRICS_FOR_STATS: int = 10
