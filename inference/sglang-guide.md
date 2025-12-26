# SGLang Configuration Guide

This guide maps the theoretical optimizations discussed in previous sections to concrete **SGLang** command-line arguments. This serves as a practical reference for tuning your inference engine.

## 1. Parallelism (TP & PP)
*Ref: [Parallelism Strategies](./parallelism.md)*

*   **`--tp-size` / `--tensor-parallel-size`**:
    *   **Definition**: Sets the Tensor Parallelism degree.
    *   **Usage**: Set to average number of GPUs on a single node (e.g., 2, 4, 8). Essential for fitting 70B+ models or reducing latency.
*   **`--pp-size` / `--pipeline-parallel-size`**:
    *   **Definition**: Sets the Pipeline Parallelism degree.
    *   **Usage**: Used when a model is too large for one node, or to split layers across GPUs. Remember PP introduces "bubbles".
*   **`--pp-max-micro-batch-size`**:
    *   **Definition**: Max micro-batch size for PP.
    *   **Optimization**: Tuning this can help fill Pipeline bubbles (see [Parallelism > Bubble Problem](./parallelism.md#11-the-bubble-problem)).

## 2. Memory & KV Cache
*Ref: [KV Cache Strategies](./kv-cache.md)* & *[Serving Techniques](./serving-techniques.md)*

### Allocation
*   **`--mem-fraction-static`**:
    *   **Definition**: The fraction of GPU memory allowed for the Model Weights + KV Cache Pool.
    *   **Usage**: Decrease this if you encounter OOM errors (e.g., from 0.9 to 0.8).
*   **`--max-total-tokens`**:
    *   **Definition**: Hard limit on the total number of tokens in the KV memory pool.
    *   **Usage**: Overrides the automatic calculation. Useful for strict multi-tenant resource control.
*   **`--page-size`**:
    *   **Definition**: Number of tokens per "Page" in PagedAttention.
    *   **Usage**: Default is usually 16. Tuning this affects memory fragmentation (like OS page size).

### Optimization
*   **`--kv-cache-dtype`**:
    *   **Definition**: Data type for storing K/V matrices.
    *   **Options**: `fp16` (default), `fp8_e4m3`.
    *   **Usage**: Using `fp8` almost doubles your effective context length capacity (or batch size) with negligible accuracy loss.
*   **`--disable-radix-cache`**:
    *   **Definition**: Disables RadixAttention (Prefix Caching).
    *   **Usage**: Use if your workload has **zero** shared prefixes (random distinct requests) to save some overhead. Otherwise, keep enabled to reuse KV cache across requests (e.g., shared system prompts).

## 3. Scheduling & Throughput
*Ref: [Serving Techniques > Continuous Batching](./serving-techniques.md)*

*   **`--chunked-prefill-size`**:
    *   **Definition**: Splits a long prompt (Prefill phase) into smaller chunks.
    *   **Usage**: Critical for **Preventing OOM** during the prefill of very long documents (e.g., 100k tokens). It allows decoding requests to interleave with prefill chunks.
*   **`--enable-mixed-chunk`**:
    *   **Definition**: Allows mixing **Prefill chunks** and **Decode tokens** in the same batch.
    *   **Usage**: Increases GPU utilization by filling "gaps" in compute.

## 4. Attention Backends
*Ref: [Serving Techniques > FlashAttention](./serving-techniques.md)*

*   **`--attention-backend`**:
    *   **Options**: `flashinfer`, `triton`, `flash-attn`.
    *   **Detail**:
        *   `--prefill-attention-backend`
        *   `--decode-attention-backend`
    *   **Usage**: SGLang generally picks the best default, but you can force specific kernels for debugging or specific hardware performance. `FlashInfer` is often the fastest for PagedAttention.

## 5. Speculative Decoding
*Ref: [Speculative Decoding](./speculative-decoding.md)*

*   **`--speculative-algorithm`**:
    *   **Options**: `None`, `EAGLE`, `EAGLE3`, etc.
*   **`--speculative-draft-model-path`**:
    *   **Usage**: Path to the smaller "Draft Model" (or Eagle head weights).
*   **`--speculative-num-steps`**:
    *   **Definition**: How many draft tokens to propose in one step ($\gamma$).
    *   **Trade-off**: Too small = low speedup; Too large = high acceptance failure rate.
*   **`--speculative-eagle-topk`**:
    *   Control the tree expansion in Eagle.
*   **`--speculative-token-map`**:
    *   Advanced: Custom token map for FR-Spec.

> **Note**: Environmental variable `SGLANG_ENABLE_SPEC_V2=True` might be needed to enable experimental overlap schedulers for speculative decoding.

## 6. System & Kernel Optimizations

*   **CUDA Graph**:
    *   **`--cuda-graph-max-bs`** / **`--disable-cuda-graph`**
    *   **Concept**: "Records" the kernel launch sequence to reduce CPU-to-GPU launch overhead. Critical for small batch sizes (latency-bound).
*   **Torch Compile**:
    *   **`--enable-torch-compile`**: Uses `torch.compile` to optimize the model graph.
*   **Overlap**:
    *   **`--enable-two-batch-overlap`** / **`--enable-single-batch-overlap`**
    *   **Concept**: Hides communication/CPU overhead by overlapping it with GPU computation.
