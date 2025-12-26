# Model Optimization

Beyond caching and architectural changes, we can optimize the model weights themselves to reduce memory footprint and increase inference speed.

## 1. Quantization

Quantization reduces the precision of model weights and activations, typically from FP16 (16-bit) to INT8 or INT4.

### 1.1 Types of Quantization

*   **Weight-Only Quantization**: Only quantizes the weights ($W$). Activations ($X$) remain in FP16. De-quantization happens on the fly during matrix multiplication.
    *   *Pros*: Easiest to implement, saves VRAM.
    *   *Examples*: `AWQ`, `GPTQ`.
*   **Weight & Activation Quantization**: Quantizes both. Allows for integer matrix multiplication (which is faster on Tensor Cores).
    *   *Examples*: `SmoothQuant`.

### 1.2 Quantization Schemes

*   **Symmetric/Asymmetric**:
    *   *Symmetric*: Maps the range $[-max, max]$ to integers. Zero point is at 0.
    *   *Asymmetric*: Maps $[min, max]$ to integers. Zero point can be arbitrary. (More accurate but slightly slower).
*   **PTQ vs QAT**:
    *   **Post-Training Quantization (PTQ)**: Quantize a pre-trained model using a small calibration dataset. (Most common for inference).
    *   **Quantization-Aware Training (QAT)**: Finetune the model with "fake quantization" nodes to adapt it to low precision. (Better accuracy, but requires training).

### 1.3 The "Outlier" Problem
In large Transformers (>6B params), activation outliers emerge (values > 100 while others are < 1). Naive quantization destroys these outliers, ruining accuracy.
*   **LLM.int8() Solution**: Decompose matrix multiplication. 99.9% of normal values use INT8. The 0.1% outliers are computed in FP16.

## 2. Sparsity

Sparsity prunes the model by forcing weight values to zero.

*   **Unstructured Sparsity**: Random zeros.
    *   *Hardware Support*: Poor. GPUs hate random access.
*   **Structured Sparsity (N:M)**: e.g., NVIDIA's **2:4 Sparsity** (Ampere+ GPUs).
    *   In every contiguous block of 4 weights, at least 2 must be zero.
    *   Tensor Cores can effectively "skip" the math for zeros, yielding up to 2x speedup.

## 3. Distillation

Distillation transfers knowledge from a large "Teacher" model to a smaller "Student" model.

### 3.1 Mechanism
Instead of training the student on just "hard labels" (e.g., ground truth token is "The"), we train it to match the Teacher's **Logits**.
*   **Teacher Logits**: "The" (0.8), "A" (0.15), "An" (0.05).
*   **Information**: The teacher knows "A" is a better mistake than "Horse". This "dark knowledge" helps the student learn faster and generalize better.

### 3.2 Types
*   **White-box Distillation**: You have access to the teacher's weights/logits (e.g., training a 1B model from a 7B model you own).
*   **Black-box Distillation**: You only see the teacher's text output (e.g., usage of GPT-4 generated synthetic data to finetune Llama-3). This is effectively "Knowledge Distillation via data".
