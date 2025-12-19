# LLM Generation Parameters

This guide details common parameters and strategies used in LLM inference, including Search Strategies (Greedy/Beam) and Sampling Strategies (Top-K, Top-P, Temperature). 

## 1. Introduction

When tuning LLM outputs, parameters like $top\_k$, $top\_p$, $temperature$, and $repetition\_penalty$ are critical. Understanding the difference between `greedy search` and `beam search` is also fundamental.

## 2. Background

Modern LLMs typically use a **Transformer Decoder** architecture.
-   **Context**: The model predicts the next token based on the history of previous tokens.
-   **Mechanism**: For an input like "`a robot must obey the orders given it`", the model processes the embeddings and outputs a new embedding for the last token position ("`it`").
-   **Logits**: This new embedding interacts with the **Token Embeddings** matrix (via dot product) to produce a similarity score (**logit**) for every token in the vocabulary. The higher the logit, the more likely the token is the next word.

## 3. Greedy Search

**Definition**: Surprisingly simple. At each step, just choose the token with the **highest logit score**.

**Example Walkthrough**:
Suppose we have a small vocabulary: `["a", "given", "human", "it", "must", "obey", "Okay", "orders", "robot", "the", ".", "EOS"]`.

1.  **Step 1**: Input "`...given it`". Model computes logits.
    *   **"Okay"**: 0.91 (Highest)
    *   ".": 0.84
    *   **Action**: Choose "**Okay**".
    *   Current Sentence: "`...given it Okay`"

2.  **Step 2**: Input "`...given it Okay`". Model computes logits.
    *   **"human"**: 0.89 (Highest)
    *   "the": 0.72
    *   **Action**: Choose "**human**".
    *   Current Sentence: "`...given it Okay human`"

3.  **Step 3**: Input "`...given it Okay human`". Model computes logits.
    *   **"."**: 0.78 (Highest)
    *   **Action**: Choose "**.**".
    *   Current Sentence: "`...given it Okay human.`"

4.  **Step 4**: Input "`...given it Okay human .`". Model computes logits.
    *   **"EOS"**: 0.90 (Highest)
    *   **Action**: Stop generation.

**Summary**: Greedy search is deterministic. For a fixed model and input, the output is always identical.

## 4. Beam Search

**Definition**: An improvement over Greedy Search. Instead of keeping only the single best path, it maintains the top `beam_size` most likely **sequences** at every step.

**Example Walkthrough (`beam_size = 2`)**:

1.  **Step 1**: Input "`...given it`". Top 2 tokens are:
    *   **Seq A**: "`...it `**`Okay`**" (Score 0.91)
    *   **Seq B**: "`...it `**`.`**" (Score 0.84)

2.  **Step 2**: Expand BOTH Sequence A and Sequence B.
    *   **From Seq A ("Okay")**:
        *   Next "human" (0.89) $\rightarrow$ Total Score: 0.8099 (e.g., $0.91 \times 0.89$)
        *   Next "the" (0.72) $\rightarrow$ Total Score: 0.6552
    *   **From Seq B (".")**:
        *   Next "the" (0.92) $\rightarrow$ Total Score: 0.7728
        *   Next "EOS" (0.70) $\rightarrow$ Total Score: 0.5880
    *   **Pruning**: Keep top 2 sequences from ALL 4 candidates above.
        *   **Winner 1**: "`Okay human`" (0.8099)
        *   **Winner 2**: "`. the`" (0.7728)

3.  **Step 3**: Expand the two winners...
    *   After calculation, we keep:
        *   "`Okay human.`" (Score 0.6317)
        *   "`. the human`" (Score 0.6128)

4.  **Step 4**: Expand again...
    *   One candidate reaches `EOS` with high score. Terminate.

**Pros/Cons**: Beam search finds better overall sentences by looking ahead, but costs `beam_size` times more compute/memory.

## 5. Top-K Sampling

**Definition**: To introduce diversity (creativity), we don't just pick the #1 token. We pick the **Top K** highest scoring tokens and sample randomly among them.

**Example Walkthrough (`K = 3`)**:

1.  **Step 1**: Model predicts logits for next token.
    *   Top 3 candidates: `["Okay", ".", "EOS"]` with scores `[0.91, 0.84, 0.72]`.
    *   **Action**: Randomly pick one weighted by score. Suppose we get "**Okay**". (Even though "." was 0.84, we *could* have picked it).

2.  **Step 2**: Next prediction.
    *   Top 3 candidates: `["human", "robot", "the"]` with scores `[0.89, 0.65, 0.72]`.
    *   **Action**: Randomly pick "**the**". (Note: "human" was higher, but we rolled "the").

**Summary**: If $K=1$, it becomes Greedy Search. Larger K = more random.

## 6. Top-P (Nucleus) Sampling

**Definition**: Instead of a fixed $K$, we choose the smallest set of tokens whose **cumulative probability** exceeds $P$.

**Why?**:
*   Sometimes the valid options are few (e.g., "New [York, Jersey, Hampshire]"), so $K=10$ would include nonsense.
*   Sometimes valid options are many, so $K=10$ cuts off good words.
*   $P$ adapts dynamically.

**Example Walkthrough (`P = 2.2`** *using raw scores sum for simplicity*):

1.  **Step 1**: Sort tokens by score.
    *   `Okay` (0.91), `.` (0.84), `EOS` (0.72) ...
    *   Sum: $0.91+0.84+0.72 = 2.47 > 2.2$.
    *   **Pool**: `{Okay, ., EOS}`. Sample from these. Pick "**Okay**".

2.  **Step 2**: Next prediction.
    *   Sorted: `human` (0.89), `the` (0.72), `robot` (0.65) ...
    *   Sum: $0.89+0.72+0.65 = 2.26 > 2.2$.
    *   **Pool**: `{human, the, robot}`. Sample from these. Pick "**the**".

3.  **Step 3**: Next prediction.
    *   Sorted: `human` (0.82), `robot` (0.53), `.` (0.48), `obey` (0.41)...
    *   To reach sum > 2.2, we might need **4** tokens this time.
    *   **Pool**: `{human, robot, ., obey}`.

**Summary**: Top-P is usually preferred over Top-K for generating natural text because it adapts to the uncertainty of the context.

## 7. Temperature

**Definition**: A parameter $T$ that scales logits before the softmax.

$$ \operatorname{softmax}(y_i) = \frac{e^{y_i/T}}{\sum_j e^{y_j/T}} $$

*   **$T=1$**: Standard Softmax.
*   **$T < 1$** (e.g., 0.1): Low temperature. The distribution becomes **sharper**. High scores get relatively *much* higher. Model becomes **confident, repetitive, conservative**.
*   **$T > 1$** (e.g., 2.0): High temperature. The distribution becomes **flatter**. Differences shrink. Low probability tokens get a better chance. Model becomes **creative, random, error-prone**.

**Visual Effect**:
Imagine scores `[0.92, 0.11, 0.33, 0.04]`.
*   **Low T**: `[0.999, 0.000, 0.001, 0.000]` -> Almost certainly picks the first one.
*   **High T**: `[0.35, 0.20, 0.25, 0.20]` -> Any token is fair game.

## 8. Repetition Penalty

**Definition**: Specifically penalize tokens that have appeared in the generated history $g$.

$$ p_i = \frac{\exp(x_i / (T \cdot \text{penalty}))}{\sum \dots} $$

where $\text{penalty} = \theta$ if token $i \in g$, otherwise $1$.

**Example**:
*   Candidate tokens: `["human", "obey", "robot", "EOS"]` with scores `[0.92, 0.11, 0.33, 0.04]`.
*   History: `g = ["robot", "it"]`.
*   **Case $\theta = 3.0$ (Avoid Repetition)**:
    *   The score for "robot" (0.33) is heavily penalized (divided/reduced). Its probability drops significantly.
*   **Case $\theta = 0.5$ (Encourage Repetition)**:
    *   The score for "robot" is boosted. Its probability increases.

## 9. Conclusion

*   **Greedy/Beam Search**: Best for tasks requiring exactness (Math, Coding). Deterministic.
*   **Top-K/Top-P**: Best for open-ended generation (Chat, Storytelling). Adds variety.
*   **Temperature**: The master knob for "Creativity vs. Stability".
*   **Repetition Penalty**: A specific fix for the "looping" problem.
