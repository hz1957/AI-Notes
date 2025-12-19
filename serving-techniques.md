# Serving Techniques

Optimizing the 'engine' that runs the model is just as important as the model itself.

## 1. Continuous Batching (In-flight Batching)

Static batching is inefficient because requests naturally vary in length.

### Visual Comparison

**Static Batching**:
GPU is stuck waiting for the longest request (Request 3).
Wait times represented by `---`.
```text
GPU Timeline:
| Req 1 (Len 10) ---(Idle)---|
| Req 2 (Len 20) -(Idle)-|
| Req 3 (Len 100) -------| -> Batch Finishes
```

**Continuous Batching**:
As soon as Req 1 finishes, Req 4 is inserted into that slot.
```text
GPU Timeline:
| Req 1 (Len 10) | Req 4 (Len 50) ... |
| Req 2 (Len 20) | Req 5 (Len 30) ... |
| Req 3 (Len 100) ................... |
```
*   **Throughput**: Throughput can increase by 10x-20x because the GPU is never 'waiting'.
*   **Implementation**: Requires sophisticated schedulers (like in vLLM, TGI, TRT-LLM) that manage the KV cache dynamically at each step.

## 2. PagedAttention

Standard attention implementations require contiguous memory for KV Cache.
*   *Problem*: You must allocate VRAM for the `max_seq_len` (e.g., 2048) even if the prompt is short.
*   *Result*: Huge memory fragmentation.

### The OS Analogy (Virtual Memory)
PagedAttention treats KV Cache like an Operating System treats RAM.

*   **Logical Blocks**: The model sees a continuous sequence of tokens `[0, 1, 2, ... N]`.
*   **Physical Blocks**: The data is actually stored in scattered fixed-size blocks (e.g., Block size = 16 tokens) anywhere in VRAM.
*   **Block Table**: A mapping table (like a Page Table in OS) translates `Logical Block 0` $\rightarrow$ `Physical Address 0xA000`.

**Impact**:
*   **Zero Waste**: Only allocate a new small block when needed.
*   **Sharing**: Different sequences can share standard blocks (e.g., a common "System Prompt" block) simply by pointing to the same physical address. This is massive for beam search or "Prompt Caching".

## 3. FlashAttention

**Memory I/O Bottleneck**:
In standard Attention: $A = \text{softmax}(Q \cdot K^T) \cdot V$.
The $N \times N$ matrix $A$ is huge. Writing it to HBM (High Bandwidth Memory) and reading it back is the slow part.

**Tiling Optimization**:
FlashAttention loads small block tiles of $Q, K, V$ into the GPU's fast SRAM (L1 Cache), computes the attention for that tile, and writes *only* the result back to HBM. It never materializes the full $N \times N$ matrix in slow global memory.

*   **Speedup**: 2-4x faster than standard attention.
*   **Long Context**: Enables 32k, 100k+ context windows because memory usage scales linearly $O(N)$ instead of quadratically $O(N^2)$.
