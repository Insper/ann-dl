!!! success inline end "Deadline and Submission"

    :date: TBD
    
    :clock1: Commits until 23:59

    :material-account: Individual

    :simple-github: GitHub Pages link via [insper.blackboard.com](https://insper.blackboard.com){:target="_blank"}.

**Activity: Building Attention and Transformers from Scratch**

This activity solidifies your understanding of attention mechanisms and Transformer architecture by implementing them from the ground up using **only NumPy and Python** (no PyTorch or TensorFlow for the core logic).

---

## Exercise 1 — Scaled Dot-Product Attention

Implement the full scaled dot-product attention function:

$$
\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
$$

### Instructions

1. **Implement `softmax(x)`** — numerically stable version (subtract max before exponentiating)
2. **Implement `scaled_dot_product_attention(Q, K, V, mask=None)`**:
   - Compute raw scores: `scores = Q @ K.T / sqrt(d_k)`
   - Apply mask if provided (set masked positions to `-inf` before softmax)
   - Apply softmax row-wise
   - Return weighted sum of Values: `output = attn_weights @ V`
3. **Test with the following inputs:**

```python
import numpy as np

d_k = 4
Q = np.array([[1.0, 0.0, 1.0, 0.0],   # token 1 query
              [0.0, 1.0, 0.0, 1.0]])   # token 2 query
K = np.array([[1.0, 0.0, 1.0, 0.0],   # token 1 key
              [0.0, 1.0, 0.0, 1.0],   # token 2 key
              [1.0, 1.0, 0.0, 0.0]])  # token 3 key
V = np.array([[1.0, 0.0],
              [0.0, 1.0],
              [0.5, 0.5]])
```

4. **Plot the attention weight matrix** as a heatmap (use matplotlib). What pattern do you observe? Does token 1 attend more to token 1 or token 3? Why?

5. **Apply a causal mask** (lower-triangular) and re-run. Show how the attention weights change and explain why this mask is necessary for autoregressive generation.

### Expected output

Report:

- The attention weight matrix (2×3) before and after masking
- The output matrix (2×2)
- Heatmap visualizations
- A brief explanation of why Q1 attends more to K1 than to K2

---

## Exercise 2 — Multi-Head Attention from Scratch

Extend your implementation to **Multi-Head Attention** with $h=2$ heads.

### Architecture

$$
\text{MultiHead}(Q,K,V) = \text{Concat}(\text{head}_1, \text{head}_2)\,W^O
$$

where each head uses its own projection matrices.

### Instructions

1. **Implement the class `MultiHeadAttention`** with:
   - `__init__(d_model, num_heads)` — initialize random weight matrices $W_Q^i, W_K^i, W_V^i \in \mathbb{R}^{d_{model} \times d_k}$ and $W^O \in \mathbb{R}^{d_{model} \times d_{model}}$ for each head $i$
   - `forward(Q, K, V)` — project, apply per-head attention, concatenate, project output
2. **Use fixed random seed** `np.random.seed(42)` for reproducibility
3. **Test with a sequence of 5 tokens**, each with `d_model=8`, `num_heads=2` (so `d_k=4` per head)
4. **Verify** the output shape is `(5, 8)` — same as input

### Questions to answer in your report

- Why does using $h=2$ heads with $d_k = d_{model}/h$ keep the total computation similar to $h=1$?
- If head 1 learns to attend to nearby tokens and head 2 to distant tokens, how does the concatenated output benefit from both?

---

## Exercise 3 — Single-Layer Transformer Block

Combine your attention with a Feed-Forward Network to implement a **Transformer Encoder Block**:

$$
x' = \text{LayerNorm}(x + \text{MultiHeadAttn}(x, x, x))
$$
$$
x'' = \text{LayerNorm}(x' + \text{FFN}(x'))
$$

### Instructions

1. **Implement `layer_norm(x)`** — normalize per row (subtract mean, divide by std + ε), with learnable γ=1, β=0
2. **Implement `ffn(x, W1, b1, W2, b2)`** — two linear layers with ReLU: $\text{FFN}(x) = \text{ReLU}(xW_1 + b_1)W_2 + b_2$, where $d_{ff} = 4 \times d_{model}$
3. **Stack it all**: build `transformer_encoder_block(x, mha, W1, b1, W2, b2)` using your implementations above
4. **Test with 5 tokens, d_model=8**

### Visualization

Plot the **token representations before and after the block** as a heatmap (tokens × dimensions). Do the representations become richer after the block? Compute and report the cosine similarity matrix between tokens before and after the block.

---

## Exercise 4 — Positional Encoding

Implement sinusoidal positional encoding and visualize it.

$$
PE_{(pos, 2i)} = \sin\!\left(\frac{pos}{10000^{2i/d}}\right), \quad PE_{(pos, 2i+1)} = \cos\!\left(\frac{pos}{10000^{2i/d}}\right)
$$

### Instructions

1. Implement `positional_encoding(max_len, d_model)` → returns matrix of shape `(max_len, d_model)`
2. **Plot two visualizations**:
   - Heatmap: rows=positions (0–99), cols=dimensions, color=PE value
   - Line plot: PE values for positions 0, 10, 50 across all dimensions

### Questions

- Which dimensions encode high-frequency oscillations and which encode low-frequency?
- Why does adding PE to the token embedding allow the Transformer to distinguish position 1 from position 50?
- What happens if you add the same positional encoding to shuffled tokens?

---

## Evaluation Criteria

!!! failure "Usage of Toolboxes"
    You may only use **NumPy** for matrix operations and **Matplotlib/Seaborn** for plots. PyTorch, TensorFlow, and other ML frameworks are **strictly prohibited** for the core implementation. Verify your results against PyTorch's `nn.MultiheadAttention` output as a sanity check only.

    **Failure to comply will result in your submission being rejected.**

| Criteria | Points |
|:---:|---|
| **3 pts** | Correct implementations of attention (Ex. 1) and Multi-Head Attention (Ex. 2) |
| **2 pts** | Transformer block (Ex. 3): correct layer norm, FFN, and residual connections |
| **2 pts** | Positional encoding (Ex. 4): correct implementation and visualizations |
| **2 pts** | Visualizations: attention heatmaps, PE plots, token representation heatmaps |
| **1 pt** | Report quality: clear explanations, mathematical notation, and discussion of results |

**Submission format:** GitHub Pages (using the [course template](https://hsandmann.github.io/documentation.template/){:target="_blank"}). No other format accepted.

**AI Collaboration:** Allowed, but every student must be able to explain all code and analysis. Oral exams may be required.
