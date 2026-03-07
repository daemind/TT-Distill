# Phase 4.3: VRAM Double Buffering Architecture (The 215ms Wall)

## The Problem
In the current TT-Distill implementation, dynamically loading a new DoRA adapter (System 2 -> System 1 Reflex transition) using `llama.cpp` on Apple Silicon (M2 Max) takes approximately `~215ms`. 

**Why?**
Because the `llama.cpp` backend (specifically `ggml_metal.m`) destroys and recreates the entire Metal compute graph and reallocates memory buffers whenever weights change. This latency is fatal for high-frequency ARC heuristics switching.

## The Objective
We need the hot-swap latency to be `< 5ms`. Since DoRA adapters are extremely small (~15MB), we can fit *hundreds* in the 96GB Unified Memory. The goal is to avoid Metal graph recreation by swapping memory pointers.

## Proposed C++ Architecture: The Ring Buffer

We propose modifying the `ggml` backend to implement an **Asynchronous Dual-Buffer (Ping-Pong Buffer)** for `lora_a` and `lora_b` tensors.

### 1. Pre-allocation
Instead of allocating memory *exactly* for the model, `llama.cpp` must allocate a dedicated "Prior Ring Buffer" in the Metal backend upon startup.
- **Buffer Size**: E.g., `256MB`
- **Capacity**: Can hold roughly 16 DoRA adapters simultaneously without triggering eviction.

### 2. Double Buffering & Pointer Swaps
The `ggml_metal_context` needs two sets of pointers for the active adapter:
- `*active_dora_buffer`
- `*preload_dora_buffer`

When the S2 Python Orchestrator (`TTDistillBridge`) anticipates a skill change, it asynchronously loads the fused `.bin` from `MoAGater` into `preload_dora_buffer`.

### 3. The $O(1)$ Swap
When the user calls `engine.inference()`, if a pre-loaded buffer is ready, `ggml_metal.m` simply swaps the pointer reference:
```objective-c
// Pseudo-code inside ggml_metal.m
void ggml_metal_swap_dora(struct ggml_metal_context * ctx) {
    if (ctx->preload_ready) {
        void * temp = ctx->active_dora_buffer;
        ctx->active_dora_buffer = ctx->preload_dora_buffer;
        ctx->preload_dora_buffer = temp; // Set to be overwritten
        ctx->preload_ready = false;
        
        // NO GRAPH RECREATION. NO `mmap` TEARDOWN.
    }
}
```
This reduces the `215ms` blocking I/O wall to a pure $O(1)$ memory address swap taking less than `0.1ms`.

## Execution Roadmap
Modifying `llama.cpp` is out of scope for the current Python-centric agent operations, as it requires a `C++/Objective-C` recompilation environment and deep integration with `ggml`.
This document serves as the formal specification for the future C++ integration phase. Current Python layers (`maca.py` and `moa_gating.py`) are fully prepared to feed this future Ring Buffer asynchronously.
