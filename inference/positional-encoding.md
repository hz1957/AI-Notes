# Positional Encodings in LLMs

## 1. Why do we need them? (The "Bag of Words" Problem)

Transformers process all tokens in a sentence **in parallel**. Without explicit position information, the model sees:
*   "The cat ate the mouse"
*   "The mouse ate the cat"

as identical collections of words: `{"the", "the", "ate", "cat", "mouse"}`.

Positional Encoding is the method we use to tell the model **"This 'the' is at position 1, and that 'cat' is at position 2."**

---

## 2. Absolute Positional Embedding (APE)
*The "Name Tag" Approach*

**Concept**:
Imagine every position ($0, 1, 2, ...$) has a unique "Name Tag" (a vector). When a word enters the model, we stick the corresponding position's name tag onto it.

$$ \mathbf{FinalInput} = \mathbf{WordEmbedding} + \mathbf{PositionEmbedding} $$

*   **Used in**: BERT, GPT-2, original Transformer.
*   **How it works**: The model learns a specific vector for "Position 1," another for "Position 2," etc.

**Why it fails at Extrapolation (The "Unknown Number" Problem)**:
If you train your model with a max length of 1024, it has learned "Name Tags" for positions 0 to 1023.
During inference, if you feed it a sequence of length 1025, the model looks for the "Position 1024" tag, finds it doesn't exist (or is randomized), and crashes or outputs nonsense. It's like asking a student to count to 100 when they only learned numbers up to 50.

---

## 3. RoPE (Rotary Positional Embedding)
*The "Clock Hand" Analogy*

**Concept**:
Instead of adding a "tag," RoPE **rotates** the word vector.
Imagine each word vector is a hand on a clock.
*   **Word at Pos 0**: Don't rotate.
*   **Word at Pos 1**: Rotate by $10^\circ$.
*   **Word at Pos 2**: Rotate by $20^\circ$.
*   ...
*   **Word at Pos $m$**: Rotate by $m \times 10^\circ$.

**The Magic of Relative Distance**:
To find the relationship between two words at position $m$ and $n$, we calculate their dot product (attention score).
*   In the rotation world, if you take the dot product of two vectors, the result depends mostly on the **angle between them**.
*   Angle of word $m$: $10m$. Angle of word $n$: $10n$.
*   The difference is $10(n-m)$.

**Intuition**:
The model doesn't need to know "I am at absolute position 1052". It only cares "That word is 5 steps behind me". RoPE captures this perfectly because the angle difference depends only on the relative distance.

**Why it's better**:
*   **Relative is what matters**: "King" - "Man" + "Woman" = "Queen". Relationships are relative.
*   **Better Extrapolation**: Even if the model hasn't seen position 1052, understanding "rotation" is a general concept. However, pure RoPE still struggles with *extreme* distances without tweaks (like altering the "frequency" of rotation).

---

## 4. ALiBi (Attention with Linear Biases)
*The "Fading Spotlight" Analogy*

**Concept**:
ALiBi says: "Forget embedding positions. Let's just modify the attention score directly."
It creates a rigid rule: **The further away a token is, the less attention it gets.**

**How it works**:
When calculating the attention score between a query $q$ (at pos $i$) and a key $k$ (at pos $j$):
1. Compute the standard score: $S = q \cdot k$.
2. **Subtract a penalty** based on distance: $S_{final} = S - m \times |i - j|$.

**Visualizing the Penalty Matrix**:
Imagine looking at recent history.
*   Distance 0 (Self): Penalty 0.
*   Distance 1 (Prev word): Penalty -1.
*   Distance 2: Penalty -2.
*   Distance 100: Penalty -100.

**Why it Extrapolates Perfectly**:
*   The model learns the concept: "Distant things are less relevant."
*   If trained on 1024 tokens, it learns to handle penalties like -1, -50, -500.
*   If tested on 2048 tokens, it encounters a penalty of -1500. To the model, -1500 is just "very far away" (similar to -500). It doesn't break; it just pays very little attention. This is why ALiBi has **true zero-shot extrapolation**.

---

## Summary Comparison

| Feature | APE (Old School) | RoPE (Standard) | ALiBi (Specialist) |
| :--- | :--- | :--- | :--- |
| **Analogy** | **Name Tags**: Each seat has a number. | **Clocks**: Each step rotates the hand. | **Spotlight**: Light fades with distance. |
| **How** | `Input + PositionVector` | `Rotate(Input)` | `Score - DistancePenalty` |
| **Main Idea** | Absolute (I am at 5) | Relative via Absolute (I am rotated 50deg) | Purely Relative (You are 5 away) |
| **Extrapolation** | ❌ **Fail**: New positions have no tags. | ⚠️ **Okay**: Needs math tricks (NTK scaling) to extend. | ✅ **Perfect**: "Far" is just "More Negative". |
