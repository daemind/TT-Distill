# 🏗️ Universal O(1) Swap: Cross-Platform Staging Spec

> **Version**: 1.0 (Draft)  
> **Concept**: RAM-Staging for Infinite Adapter Selection (MoA)  
> **Target**: Linux (CUDA/Vulkan), Windows (DirectX/DirectStorage)

## 1. The Strategy: Host-Mapped Zero-Copy
While Apple Silicon uses physical Unified Memory, other platforms can achieve similar O(1) swap latency by using **Zero-Copy Host Memory**. Instead of copying the adapter to VRAM before inference, we map a region of System RAM directly into the GPU's address space.

### Mechanism
1.  **Allocation**: Allocate "Pinned" (non-pageable) memory on the CPU.
2.  **Mapping**: Expose that memory address to the GPU via PCIe BAR (Base Address Register).
3.  **Swap**: Update the GPU's Descriptor Set or Constant Buffer pointer to look at the new RAM address.

---

## 2. Technical implementation (Architecture)

### Linux/Unix (CUDA)
Use `cudaHostAlloc` with the `cudaHostAllocMapped` flag.
*   **Latency**: Sub-microsecond (Pointer update only).
*   **Throughput**: 32-64 GB/s (PCIe 4.0/5.0).
*   **Benefit**: You can stage 10,000+ adapters in 128GB of RAM and swap them instantly without touching VRAM limits.

### Universal (Vulkan)
Use `VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT` + `VK_MEMORY_PROPERTY_HOST_COHERENT_BIT`.
*   **Protocol**: Similar to Metal, we use a ring buffer of memory descriptors.
*   **Portability**: Works on AMD, Intel, and NVIDIA on any Unix-like system.

---

## 3. The "Strict" Implementation Loop

To maintain the **TT-Distill AGENT.md** rigor on these platforms:
1.  **Hardware Check**: Probe for PCIe Atomic support.
2.  **Strict Handshake**: Ensure `memcpy` to Host-Mapped RAM is complete before the GPU fence is released.
3.  **O(1) Enforcement**: Monitor PCIe controller latency; trigger a `RuntimeError` if the bus stutters beyond the 5ms threshold.

## 4. Performance Trade-off
| Metric | VRAM (Native) | RAM-Mapped (Zero-Copy) |
| :--- | :--- | :--- |
| **Swap Latency** | **O(1)** | **O(1)** |
| **Mem-Copy Latency** | High (VRAM upload) | **N/A** (Staged in RAM) |
| **Inference Bandwidth** | ~900 GB/s | ~32-64 GB/s |

**Conclusion**: For **Mixture of Adapters (MoA)** where model weights are small (LoRA/DoRA), RAM-Staging is the superior choice because the latency of *uploading* to VRAM is the primary bottleneck we want to eliminate.
