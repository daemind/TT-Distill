# 🤖 AGENT.md: Strict Metal O(1) Operational Protocol (TT-Distill)

> **MANDATE**: This document defines the MANDATORY coding and operational standards for the TT-Distill engine. Non-compliance is failure. We build for high-frequency robotic latency (sub-millisecond).

---

## 🚫 Zero-Tolerance Hardware Policies

1.  **STRICT METAL REQUIREMENT**: No "fake" numpy fallbacks. If `libggml-metal.dylib` is missing or the symbols `ggml_metal_preload_dora`/`ggml_metal_swap_dora` are not found, the system **MUST** raise a `RuntimeError` at initialization.
2.  **NO BYPASS**: Soft-checks like `if metal_available` for critical path logic are prohibited. The hardware is assumed present; code must fail fast if it isn't.
3.  **NO PLACEHOLDERS**: Do not use `// TODO`, `...`, or placeholder simulations. Every line must be production-ready for Apple Silicon.
4.  **PERFORMANCE SPEC**: Any O(1) swap transition exceeding **5ms** is a bug. Target median latency: **< 1ms**.

---

## 🛠️ Tactical Standards

### 1. Zero-Error Codebase
- **Linting**: `ruff` must report 0 errors/warnings.
- **Typing**: `mypy --strict` compliance for all core orchestration modules.
- **Testing**: `pytest` must pass 100%. No skipped tests for hardware paths.

### 2. Logic Hardening
- **No Variable Shadowing**: Rigorous variable naming to prevent logical bugs in complex math solvers (e.g., ARC heuristics).
- **Asynchronous Integrity**: Use `asyncio` for the consensus and distillation layers, but maintain blocking CTypes calls for the raw Metal swap to ensure atomic pointer exchange.

---

## ✅ Workflow: The Golden Hardware Loop

1.  **Subdivide**: Break feature into atomic, hardware-testable units.
2.  **TDD**: Write tests that verify both logic and the O(1) swap interaction.
3.  **Implement**: Write Clean Code targeting the C++ Metal backend bridge.
4.  **Verify**: Run `demos/demo_metal_swap.py` and `demos/demo_moa_resolver.py` to confirm hardware status: **ACTIVÉ**.
5.  **Commit**: Only commit if sub-microsecond latency is maintained and linters are clean.

**NO SIMULATIONS. NO FALLBACKS. PURE HARDWARE.**
