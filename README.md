# TT-Distill: Continuous Neuro-Symbolic Routing via O(1) Latent Algebraic Swapping

![TT-Distill Architecture](assets/hero.png)

*(State of the Art as of March 2026)*

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18905253.svg)](https://doi.org/10.5281/zenodo.18905253)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Checked with mypy](https://www.mypy-lang.org/static/mypy_badge.svg)](https://mypy-lang.org/)

**Repository:** [github.com/daemind/TT-Distill](https://github.com/daemind/TT-Distill)  
**DOI:** [10.5281/zenodo.18905253](https://doi.org/10.5281/zenodo.18905253)

## Abstract

As of early 2026, the AI industry faces an architectural dead-end for real-time autonomous systems: the reliance on massive Test-Time Compute (like Monte Carlo Tree Search) over monolithic models creates prohibitive latency and massive energy waste. **TT-Distill** introduces a radical paradigm shift: migrating intelligence from probabilistic token prediction to algebraic latent projection on Edge hardware.

By decoupling the reasoning process (System 2) from reflexive execution (System 1), TT-Distill acts as a continuous topological router. It discovers mathematical invariants in abstract reasoning tasks and crystallizes them into ultra-lightweight DoRA (Weight-Decomposed Low-Rank Adaptation) modules. Powered by a custom Zero-Copy Apple Metal C++ backend, the architecture achieves $O(1)$ cognitive context switching in 0.0002 milliseconds. TT-Distill transforms AI from a statistical text generator into a highly optimized, deterministic Tri-Engine (Heuristic, Pure Math, Latent Algebraic) capable of executing transductive reasoning at the microsecond scale.

---

## ⚡ Quick Start: The Cognitive Cockpit

To reproduce the $O(1)$ hardware swap benchmarks, you must compile the custom Metal backend:

```bash
# 1. Clone and prepare environment
git clone https://github.com/daemind/TT-Distill.git
cd TT-Distill

# 2. Compile the custom O(1) Metal backend
cd llama.cpp
cmake -S ggml -B build -DGGML_METAL=ON -DBUILD_SHARED_LIBS=ON
cmake --build build -j $(sysctl -n hw.ncpu)
cd ..

# 3. Launch Demo 7 (O(1) Hot-Swap Benchmark)
export GGML_METAL_DYLIB=$(pwd)/llama.cpp/build/src/ggml-metal/libggml-metal.dylib
python3 demos/main.py --demo 7
```

---

## 1. Theoretical Foundations: The End of the Linguistic Illusion

Current LLMs process prompts and generate reasoning step-by-step in natural language, which is computationally expensive and mathematically flawed. TT-Distill posits that language is merely a projection of deeper topological spaces.

### Transductive Algebraic Learning
Instead of prompting a model to "think," TT-Distill modifies the geometry of the model's latent space on the fly. When confronting a novel problem, the system determines the underlying mathematical structure (e.g., Boolean Lattices, Affine Spaces, Homology) and injects the corresponding mathematical laws directly into the weights via DoRA. Intelligence becomes a geometric projection ($W_{new} = W_0 + \sum g_i(B_i A_i)$) rather than a probabilistic guess.

---

## 2. Technical Architecture: The Tri-Engine System

### Pillar 1: The Topological Synthesizer (System 2)
Instead of textual Chain-of-Thought, this module uses Set Theory and Mathematical Morphology to parse reality.
* It evaluates environments as discrete topological spaces.
* It synthesizes deterministic algebraic equations (Unions, Intersections, Vectorial Translations) that describe the transformation required to solve a problem.
* Once verified in an isolated Docker sandbox, the solution is distilled into a permanent, reusable physical reflex (DoRA).

### Pillar 2: The Latent Algebraic Router (System 1)
A highly optimized **Qwen 2 Encoder** acting as the execution engine. By leveraging a standard, robust dense Transformer architecture, TT-Distill proves that deterministic efficiency relies on the routing framework, not exotic base models.
* **Continuous Mixture of Adapters (MoA):** The system dynamically routes input through a dictionary of distilled mathematical "instincts", instantly altering the Qwen 2 attention heads via DoRA.
* **Bypassing the Tokenizer:** Utilizing direct tensor injection (`inputs_embeds`), TT-Distill communicates via pure latent representation, eliminating the discretization bottleneck of standard NLP pipelines and pushing the Qwen 2 encoder to its absolute physical limits.

### Pillar 3: The Hardware Hack — Zero-Copy VRAM Swapping
The most significant engineering breakthrough of TT-Distill is the destruction of the VRAM memory reallocation latency wall.
* Implemented directly within the `ggml-metal.m` C++ backend.
* Utilizes a pre-allocated 256 MB Ring Buffer leveraging Apple Silicon's Unified Memory (`MTLResourceStorageModeShared`).
* Eliminates Metal compute graph recreation, reducing the Hot-Swap latency of neural weights from 215 ms down to 0.0002 ms ($O(1)$ pointer swap).

### Pillar 4: Cross-Platform Portability (Universal O(1))
While the reference implementation is optimized for Apple Silicon Unified Memory, the TT-Distill protocol is designed for cross-platform hardware-agnostic acceleration:
*   **Linux/Unix/Windows Staging:** The architecture supports O(1) swap on discrete GPUs via Host-Mapped Zero-Copy memory.
*   **Technical Spec:** Detailed porting instructions for CUDA/Vulkan/DirectX can be found in the [Universal Zero-Copy Swap](plans/universal_zero_copy_swap.md) specification.

---

## 3. Empirical Performance and Benchmarks

The TT-Distill architecture was benchmarked against highly complex, abstract visual-spatial reasoning grids designed to resist standard LLM pattern matching. The results validate the Tri-Engine architecture on consumer-grade Apple Silicon:

| Cognitive Mode | Execution Method | Median Swap Latency | Total Inference Latency |
| :--- | :--- | :--- | :--- |
| **Heuristic Logic** | Rule-based symbolic fallback | 0.154 ms | ~35.5 ms |
| **Pure Mathematics** | Direct tensor math execution | N/A | ~6.35 ms |
| **Latent Math (O(1))** | Dynamic DoRA Algebraic routing | **0.0002 ms** | **~11.64 ms** |

> **Note:** The context switch between two entirely different mathematical paradigms (e.g., shifting from a Vector Space logic to a Galois Field logic) occurs in 208 nanoseconds, making the cognitive shift physically transparent to the inference loop.

---

## 4. Conclusion: Real-Time Neuro-Symbolic AGI

TT-Distill demonstrates that the future of autonomous agents does not lie in scaling parameter counts to the trillions, nor in brute-forcing compute at test-time. It lies in algebraic compositionality and system-level memory routing.

By uniting rigorous mathematical synthesis with microsecond-level C++ hardware optimization, TT-Distill allows a lightweight Edge model to behave like a continuous dynamical system. It proves that true intelligence is the ability to instantaneously forge and swap the geometric lenses through which a neural network perceives reality.

---

## 5. New Features: DoRA Blending Engine (Phase 1-3)

The `dora_blender` branch introduces a complete neuro-endocrine system for automatic weight optimization:

### Phase 1: Le Moteur de Fusion (DoRA Blending Engine)
- **[`DoraBlender`](src/orchestration/dora_blender.py:63)**: CPU-based adapter blending with triple-buffered ring buffer to prevent GC-induced segfaults
- **Blending Modes**: GEOMETRIC (weighted linear combination), TILING (spatial partitioning), HYBRID (combined)
- **MCP Tools**: [`blend_adapters_manifold()`](src/orchestration/mcp_intelligence_manifold.py:204) for cocktail synaptique fusion

### Phase 2: Le Tampon de Trajectoire Latente
- **[`LatentTrajectoryBuffer`](src/orchestration/latent_trajectory.py:65)**: Short-term memory for tracking latent vectors with automatic tensor detachment to prevent VRAM leaks
- **[`TrajectoryStore`](src/persistence/trajectory_store.py:22)**: JSON-based persistence for cross-session analysis

### Phase 3: La Boucle d'Auto-Distillation
- **[`AutoDistiller`](src/orchestration/auto_distiller.py:51)**: Neuro-endocrine feedback loop that adjusts weights on failure and crystallizes successful configurations
- **MCP Tool**: [`crystallize_weights()`](src/orchestration/mcp_intelligence_manifold.py:267) for saving permanent instincts

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    DoRA Blending Pipeline                       │
├─────────────────────────────────────────────────────────────────┤
│  [Expert A, Expert B, ...] + [w1, w2, ...]                      │
│           ↓                                                     │
│  DoraBlender.blend_adapters()  (CPU numpy)                      │
│           ↓                                                     │
│  Ring Buffer Serialization (triple-buffered)                    │
│           ↓                                                     │
│  Metal preload() + O(1) swap()  (< 0.001ms)                     │
│           ↓                                                     │
│  GPU uses fused weights                                         │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                  Auto-Distillation Loop                         │
├─────────────────────────────────────────────────────────────────┤
│  Solver attempts task → Success? ──No──→ Analyze trajectory     │
│         ↓ Yes                                                    │
│  Crystallize → Save .bin file → Register in manifold            │
└─────────────────────────────────────────────────────────────────┘
```

### Performance Targets

| Metric | Target | Notes |
|--------|--------|-------|
| Metal Swap Only | < 0.001 ms | O(1) pointer swap |
| Blend + Serialize | < 25 ms | CPU numpy operations |
| Trajectory Buffer | O(1) append | FIFO eviction |
| Crystallization | < 100 ms | Disk I/O bound |

### Usage Example

```python
from src.orchestration.dora_blender import DoraBlender, BlendingMode
from src.orchestration.auto_distiller import AutoDistiller
from src.persistence.trajectory_store import TrajectoryStore
from src.orchestration.latent_trajectory import LatentTrajectoryBuffer

# Initialize blending engine
blender = DoraBlender(blending_mode=BlendingMode.GEOMETRIC)

# Blend adapters
timings = blender.blend_and_swap(
    adapters=["expert_a.bin", "expert_b.bin"],
    gating_vector=[0.7, 0.3]
)
# timings = {"merge_ms": 12.5, "swap_ms": 0.0002, "total_ms": 12.51}

# Initialize auto-distiller
trajectory_store = TrajectoryStore("data/trajectories")
distiller = AutoDistiller(
    blender=blender,
    trajectory_buffer=LatentTrajectoryBuffer(),
    trajectory_store=trajectory_store,
    max_attempts=5
)

# Run distillation loop
result = distiller.distill(
    task_data=arc_task,
    initial_weights={"DihedralGroup": 0.5, "ColorField": 0.5},
    solver_fn=my_solver
)

# Crystallize successful configuration
if result.success:
    print(f"Crystallized to: {result.crystallized_path}")
```

### Tests

All components are fully tested:
- [`test_dora_blender.py`](tests/test_dora_blender.py): 25 tests for blending engine
- [`test_latent_trajectory.py`](tests/test_latent_trajectory.py): 20 tests for trajectory buffer
- [`test_auto_distiller.py`](tests/test_auto_distiller.py): 20 tests for distillation loop
- [`test_auto_distillation_loop.py`](tests/test_auto_distillation_loop.py): 12 integration tests

Run tests: `pytest tests/test_dora_blender.py tests/test_latent_trajectory.py tests/test_auto_distiller.py tests/test_auto_distillation_loop.py -v`
