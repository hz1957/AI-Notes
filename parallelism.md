# Parallelism Strategies

Scaling inference for massive models (70B, 405B) requires distributing the workload across multiple GPUs.

## Summary & Comparison

| Strategy | Slice Dimension | What is Sliced? | Metric Optimized | Training | Inference |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Data Parallel (DP)** | `batch` | Samples | Throughput | ✅ Core | ❌ |
| **Pipeline Parallel (PP)** | `layer` | Model Layers | VRAM | ✅ | ⚠️ |
| **Tensor Parallel (TP)** | `hidden` / `head` | Large Matrices | Compute + VRAM | ✅ | ✅ Core |
| **Sequence Parallel (SP)** | `seq_len` | Activation Matrix | Activation VRAM | ✅ | ⚠️ |

*   **Intra-Layer Parallelism**: TP + SP (Slices inside a layer)
*   **Inter-Layer Parallelism**: DP + PP (Slices outside/between layers)

---

## 1. Data Parallelism (DP)
*The most intuitive approach.*

### Matrix View
Training core calculation: $X_{(\text{batch}, \text{hidden})} \cdot W_{(\text{hidden}, \text{hidden})}$.

**Action**: Slice $X$ along the `batch` dimension.
$$ X = [1024 \times 4096] \rightarrow \text{GPU}_0: X[0:256], \dots, \text{GPU}_3: X[768:1024] $$

*   **State**: Every GPU holds a **complete copy** of Model Weights $W$.
*   **Compute**: Every GPU processes a fraction of the samples.
*   **Sync**: Gradients are summed up (All-Reduce) at the end of backward pass.

### Analysis
*   **Solves**: Throughput (samples/sec). Single GPU cannot fit a large batch.
*   **Training vs Inference**:
    *   ✅ **Training**: The default standard.
    *   ❌ **Inference**: Useless for reducing latency of a single request. Only helps if you have massive concurrent traffic (which is effectively just load balancing).
*   **One-Liner**: **"Splitting the students (data) among different teachers (GPUs)."**

---

## 2. Pipeline Parallelism (PP)
*The "Relay Race" approach.*

### Matrix/Model View
Action: Slice the Model Weights $W$ along the `layer` dimension.
$$ [\text{Layers 1-10}] \rightarrow \text{GPU}_0 \quad | \quad [\text{Layers 11-20}] \rightarrow \text{GPU}_1 $$

*   **Flow**: Input $\rightarrow \text{GPU}_0 \rightarrow \text{GPU}_1 \rightarrow \dots \rightarrow \text{Output}$.

### Analysis
*   **Solves**: Model VRAM. The model model is too big to fit on one GPU.
*   **Problem**: **Pipeline Bubble**. Computational dependencies mean GPUs sit idle waiting for data.
    ```mermaid
    sequenceDiagram
        participant GPU0
        participant GPU1
        Note over GPU0: Batch 1 (Layers 1-10)
        Note over GPU1: Idle (Waiting)
        GPU0->>GPU1: Send Activations
        Note over GPU0: Idle (Waiting)
        Note over GPU1: Batch 1 (Layers 11-20)
    ```
*   **Training vs Inference**:
    *   ✅ **Training**: Common for massive models.
    *   ⚠️ **Inference**: High latency due to communication overhead and sequential processing. Used only when TP is not enough.
*   **One-Liner**: **"Splitting the assembly line (layers) among different workers."**

---

## 3. Tensor Parallelism (TP) ⭐
*The core of LLM Inference.*

### Matrix View
Action: Slice the Weight Matrix $W$ along the `hidden` dimension.
Calculation: $Y = X \cdot W$

$$ W = [ W_0 | W_1 | W_2 | W_3 ] $$

*   **Compute**: Each GPU computes $Y_i = X \cdot W_i$.
*   **Result**: $Y = \text{Concat}(Y_0, Y_1, Y_2, Y_3)$.
*   **Attention**: In Multi-Head Attention, we naturally split by **Heads** (e.g., 32 heads $\rightarrow$ 4 GPUs $\times$ 8 heads).

### Analysis
*   **Solves**: Single layer matrix is too big (Compute & VRAM bottleneck).
*   **Training vs Inference**:
    *   ✅ **Training**: Essential for large models.
    *   ✅ **Inference**: Critical. Allows multiple GPUs to solve **one single token generation** together.
*   **One-Liner**: **"Splitting one big math problem (matrix) into pieces to solve simultaneously."**

---

## 4. Sequence Parallelism (SP) ⭐
*Parallelizing Activations, not Weights.*

### Matrix View
Some operators like `LayerNorm` or `Dropout` cannot be tensor-split easily because they couple the hidden dimension.
However, tokens in a sequence are independent!

Action: Slice Input $X_{(\text{batch}, \text{seq\_len}, \text{hidden})}$ along `seq_len`.
$$ \text{seq\_len} = 4096 \rightarrow \text{GPU}_0: [0:1024], \dots, \text{GPU}_3: [3072:4096] $$

*   **State**: Each GPU holds 1/4 of the **Activation Memory**.
*   **Compute**: `LayerNorm` / `Dropout` executed locally on partial sequence.

### Analysis
*   **Solves**: Activation memory explosion in long contexts.
*   **Training vs Inference**:
    *   ✅ **Training**: Critical for Long Context training.
    *   ⚠️ **Inference**: Useful for very long context inference (Prefill phase).
*   **One-Liner**: **"Splitting the book (sequence) pages among readers."**

---

## Final Summary

*   **Data Parallel**: Cut the **Batch**. (Scale Throughput)
*   **Pipeline Parallel**: Cut the **Layers**. (Scale Model Size)
*   **Tensor Parallel**: Cut the **Matrix/Heads**. (Scale Model Size + Speed)
*   **Sequence Parallel**: Cut the **Sequence**. (Scale Context Length)

**Inference Recipe**:
*   **TP** is the default go-to.
*   **PP** if model is too big for TP alone (inter-node).
*   **SP** if context is too long (OOM during prefill).
