# TT-Distill: Empirical Validation Report (7-Demo Suite)

This document provides a technical summary of the protocols, datasets, and results for the seven fundamental demonstrations of the TT-Distill architecture.

**GitHub:** [github.com/daemind/TT-Distill](https://github.com/daemind/TT-Distill) | **DOI:** [10.5281/zenodo.18827350](https://doi.org/10.5281/zenodo.18827350)

## ⚡ Setup: Enabling the Hardware Acceleration

Before running high-frequency benchmarks (Demo 6 & 7), ensure the custom Metal backend is compiled:

```bash
cd llama.cpp
cmake -S ggml -B build -DGGML_METAL=ON -DBUILD_SHARED_LIBS=ON
cmake --build build -j $(sysctl -n hw.ncpu)
cd ..
export GGML_METAL_DYLIB=$(pwd)/llama.cpp/build/src/ggml-metal/libggml-metal.dylib
```

---

## 🏎️ Demo 1: Stress-Test de Fréquence
**Protocol**: Comparative latency and stability benchmark between the S1 Reflex pipeline (Direct `inputs_embeds` injection) and a classic RAG pipeline.
- **Data**: Synthetic real-time log stream (READ/WRITE/EXEC loops).
- **Metric**: Processing frequency (Hz) and average latency (ms).
- **Results**:
    - **S1 Reflex**: ~95.4 Hz (10.5 ms/req).
    - **RAG Baseline**: ~2.0 Hz (500.0 ms/req).
    - **Advantage**: **47.6x speedup**. Zero-copy bypass of the tokenizer eliminates the retrieval bottleneck.

## 🧠 Demo 2: Autopsie Algébrique
**Protocol**: Visualization of the latent consensus process (MACA) and weight distillation (DoRA).
- **Data**: N=4 high-dimensional latent vectors (dim=2560) representing agent "intents".
- **Metric**: Sinkhorn divergence convergence and SVD reconstruction error.
- **Results**:
    - **Consensus Score**: 0.9994 (Strong convergence).
    - **Compression**: SVD Rank-1 factorization reduced 2560x2560 weight matrices to ultra-lightweight patches.
    - **Reconstruction Error**: < 0.001%.

## 🛡️ Demo 3: Reality Filter
**Protocol**: Verification of compliance against `AGENT.md` invariants (Non-Hallucination protocol).
- **Data**: Intent tensors vs. formal environmental constraints.
- **Metric**: Compliance rate.
- **Results**: **100% compliance**. The filter successfully rejects hallucinated motor commands that violate topological invariants.

## 🔬 Demo 4: Post-Silicon Symbiose
**Protocol**: Deep hardware profiling of the inference pipeline on Apple Silicon (M2 Max).
- **Data**: Real-time telemetry from `dora-rs` and Metal performance counters.
- **Results**:
    - **Latence MACA**: 1.77 ms.
    - **Latence Reflex**: 10.54 ms.
    - **VRAM Footprint**: ~1.9 GB (Stable).
    - **Thermal State**: Nominal. Total hardware frequency achieved: **92.2 Hz**.

## 🧩 Demo 5: Résolution ARC-AGI
**Protocol**: Proof of **Transductive Learning**. Migrating System 2 code-generation (analytical) into System 1 DoRA weight manifolds (reflexive).
- **Data**: ARC-AGI Task `673ef223` (Gravity/Cohesion).
- **Results**:
    - **System 2 (Analytical)**: ~65.1 sec (MCTS + Sandbox).
    - **System 1 (TT-Distill)**: **5.77 sec** (Total context switch).
    - **Latence Reflex**: 22.5 ms/token.
    - **Conclusion**: The DoRA adapter effectively "freezes" the analytical solution into a reflexive tensor.

## 🔀 Demo 6: Résolveur Universel (MoA)
**Protocol**: Mixture of Adapters (MoA) gating across 120 ARC-AGI tasks.
- **Data**: ARC-AGI Training set (Subset of 120 tasks).
- **Results**:
    - **Latency per task**: ~11.6 ms (Latent Math Router).
    - **Gating Performance**: O(1) selection of specialized mathematical sub-routines (Affine, Homology, etc.).

## 🔧 Demo 7: Metal O(1) DoRA Swap
**Protocol**: Direct C++ Metal backend benchmark for sub-millisecond hot-swapping.
- **Data**: Synthetic DoRA weight matrices (Rank-16, Dim=2560).
- **Results**:
    - **Median Swap Latency**: **0.000208 ms**.
    - **Speedup vs Baseline**: **1,033,654x faster** than graph re-creation.
    - **Pipeline Merge+Swap**: **0.262 ms**.
    - **Status**: Validated hardware-native O(1) cognitive switching.

---
**Summary**: TT-Distill converts abstract reasoning into a pure high-frequency reflexive processing task, achieving industrial-grade latency (<15ms) on Edge hardware.
