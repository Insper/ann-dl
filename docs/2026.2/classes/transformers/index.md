## Transformers

In 2017, the paper "Attention Is All You Need"[^1] eliminated recurrence and convolutions from sequence models, replacing them entirely with attention. The result was the **Transformer** — the architecture powering GPT, BERT, ViT, Whisper, and virtually every state-of-the-art model today.

---

## Architecture Overview

The original Transformer is an *encoder-decoder* model for machine translation. Today there are encoder-only variants (BERT, for classification/representation) and decoder-only variants (GPT, for generation).

<div id="transformer-viz" style="background:#0d1117;border-radius:12px;padding:1.5rem;margin:2rem 0;overflow:hidden;">
<canvas id="tf-canvas" style="width:100%;display:block;"></canvas>
<div style="display:flex;gap:1rem;justify-content:center;margin-top:1rem;flex-wrap:wrap;" id="tf-controls">
  <button onclick="tfStep(-1)" style="padding:6px 20px;background:#21262d;color:#c9d1d9;border:1px solid #30363d;border-radius:6px;cursor:pointer;">← Previous</button>
  <span id="tf-step-label" style="color:#8b949e;font-family:monospace;line-height:2;"></span>
  <button onclick="tfStep(1)" style="padding:6px 20px;background:#21262d;color:#c9d1d9;border:1px solid #30363d;border-radius:6px;cursor:pointer;">Next →</button>
</div>
<div id="tf-desc" style="color:#c9d1d9;font-family:Inter,sans-serif;font-size:.9rem;text-align:center;margin-top:.8rem;min-height:2.5rem;padding:0 1rem;"></div>
</div>

<script>
(function() {
  const canvas = document.getElementById('tf-canvas');
  const ctx = canvas.getContext('2d');
  let currentStep = 0;

  const steps = [
    { label: "Input", desc: "Input tokens are converted to embeddings and summed with Positional Encoding." },
    { label: "Multi-Head Attention", desc: "Self-attention: each token attends to all others. 8 parallel heads capture different relationship types." },
    { label: "Add & Norm", desc: "Residual connection sums input + attention output. Layer Normalization stabilizes training." },
    { label: "Feed-Forward", desc: "FFN: two linear layers with ReLU. Applies non-linear transformations token-by-token." },
    { label: "Add & Norm", desc: "Second residual block + normalization. Each encoder block repeats steps 2-4 N times (typically N=6)." },
    { label: "Cross-Attention", desc: "Decoder uses Q from generated outputs, but K and V come from the Encoder — connecting source and target." },
    { label: "Output Linear + Softmax", desc: "Final projection to vocabulary → softmax → probabilities → next token (autoregressive generation)." },
  ];

  window.tfStep = function(d) {
    currentStep = Math.max(0, Math.min(steps.length - 1, currentStep + d));
    render();
  };

  function render() {
    const W = canvas.parentElement.offsetWidth - 48;
    const H = 320;
    canvas.width = W; canvas.height = H;
    canvas.style.height = H + 'px';
    ctx.fillStyle = '#0d1117';
    ctx.fillRect(0,0,W,H);

    const step = steps[currentStep];
    document.getElementById('tf-step-label').textContent = 'Step ' + (currentStep+1) + '/' + steps.length + ': ' + step.label;
    document.getElementById('tf-desc').textContent = step.desc;

    const cx = W / 2;
    const activeColor = '#f0883e';
    const dimColor = '#21262d';
    const dimText = '#484f58';

    const components = [
      { label: "Input Embeddings + Pos. Enc.", y: 20, h: 32, step: 0 },
      { label: "Multi-Head Self-Attention", y: 68, h: 36, step: 1 },
      { label: "Add & Norm", y: 116, h: 24, step: 2 },
      { label: "Feed-Forward Network", y: 152, h: 36, step: 3 },
      { label: "Add & Norm", y: 200, h: 24, step: 4 },
      { label: "Cross-Attention (Decoder)", y: 240, h: 36, step: 5 },
      { label: "Linear → Softmax → Token", y: 288, h: 28, step: 6 },
    ];

    const boxW = Math.min(500, W - 40);
    const x0 = cx - boxW/2;

    components.forEach(c => {
      const active = c.step === currentStep;
      const past = c.step < currentStep;
      const bg = active ? activeColor : past ? '#1f3244' : dimColor;
      const tc = active ? '#0d1117' : past ? '#58a6ff' : dimText;
      const lw = active ? 2.5 : past ? 1 : 0.5;

      ctx.fillStyle = bg;
      ctx.beginPath(); ctx.roundRect(x0, c.y, boxW, c.h, 6); ctx.fill();
      if (active) {
        ctx.strokeStyle = activeColor; ctx.lineWidth = lw;
        ctx.beginPath(); ctx.roundRect(x0-2, c.y-2, boxW+4, c.h+4, 8); ctx.stroke();
      }

      ctx.fillStyle = tc;
      ctx.font = (active ? 'bold ' : '') + Math.min(14, boxW/30) + 'px Inter,sans-serif';
      ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
      ctx.fillText(c.label, cx, c.y + c.h/2);

      if (c !== components[components.length-1]) {
        const nextY = components[components.indexOf(c)+1].y;
        ctx.strokeStyle = past || active ? '#30363d' : '#1c2128';
        ctx.lineWidth = 1.5; ctx.setLineDash([3,3]);
        ctx.beginPath(); ctx.moveTo(cx, c.y+c.h); ctx.lineTo(cx, nextY); ctx.stroke();
        ctx.setLineDash([]);
      }
    });

    if (currentStep === 1 || currentStep === 5) {
      ctx.fillStyle = '#58a6ff22';
      ctx.beginPath(); ctx.roundRect(x0 + boxW + 10, 60, 120, 90, 8); ctx.fill();
      ctx.fillStyle = '#58a6ff'; ctx.font = '10px monospace'; ctx.textAlign = 'center';
      const notes = currentStep === 1 ?
        ["8 heads", "d_model=512", "d_k=64", "O(n²d)"] :
        ["Q ← Decoder", "K,V ← Encoder", "Aligns", "source↔target"];
      notes.forEach((n,i) => ctx.fillText(n, x0+boxW+70, 82+i*18));
    }
  }

  render();
  window.addEventListener('resize', render);
})();
</script>

---

## The Encoder Block in Detail

Each of the $N$ identical encoder blocks consists of:

$$
\text{SubLayer}_1(x) = \text{LayerNorm}(x + \text{MultiHead}(x, x, x))
$$

$$
\text{SubLayer}_2(x) = \text{LayerNorm}(x + \text{FFN}(x))
$$

The **FFN** (Feed-Forward Network) is applied independently to each position:

$$
\text{FFN}(x) = \max(0,\; xW_1 + b_1)W_2 + b_2
$$

Typically $d_{\text{model}} = 512$, $d_{\text{ff}} = 2048$ — a *bottleneck* 4× wider than the embedding.

**Residual connections** (inspired by ResNets) ensure gradients flow directly through many layers. **Layer Normalization** normalizes along the feature dimension (not batch), which is more stable for variable-length sequences.

---

## Decoder and Autoregressive Generation

The decoder generates tokens one at a time, conditioned on everything generated so far:

$$
p(\text{output}) = \prod_{t=1}^{T} p(y_t \mid y_{<t},\; \text{encoder output})
$$

To prevent token $t$ from seeing future tokens during training, **masked self-attention** applies a causal mask:

<div style="background:#161b22;border-radius:8px;padding:1.2rem;margin:1.5rem 0;font-family:monospace;overflow-x:auto;">

```
Causal mask (n=4 tokens):
     pos0  pos1  pos2  pos3
pos0 [ 0   -inf  -inf  -inf ]   (sees only itself)
pos1 [ 0    0   -inf  -inf ]    (sees pos0 and pos1)
pos2 [ 0    0    0   -inf ]
pos3 [ 0    0    0    0   ]     (sees everything)
```

</div>

---

## BERT vs. GPT — Encoder vs. Decoder

<div style="display:grid;grid-template-columns:1fr 1fr;gap:1rem;margin:1.5rem 0;">

<div style="background:#0d1117;border-left:3px solid #58a6ff;padding:1rem;border-radius:8px;">
<strong style="color:#58a6ff;">BERT (Encoder-only)</strong><br><br>
<ul style="color:#c9d1d9;font-size:.9rem;margin:0;padding-left:1.2rem;">
<li>Bidirectional: sees left <em>and</em> right context</li>
<li>Pre-trained with <strong>Masked Language Model</strong></li>
<li>Excellent for classification, NER, QA</li>
<li>Not generative</li>
</ul>
</div>

<div style="background:#0d1117;border-left:3px solid #3fb950;padding:1rem;border-radius:8px;">
<strong style="color:#3fb950;">GPT (Decoder-only)</strong><br><br>
<ul style="color:#c9d1d9;font-size:.9rem;margin:0;padding-left:1.2rem;">
<li>Causal: sees only left context</li>
<li>Pre-trained with <strong>Next Token Prediction</strong></li>
<li>Excellent for text generation, chat, code</li>
<li>Scale leads to emergent capabilities</li>
</ul>
</div>

</div>

---

## Vision Transformer (ViT)

The encoder is not limited to text. In 2020, Dosovitskiy et al.[^3] applied this same architecture to images by dividing them into $16 \times 16$ pixel **patches** and treating each patch as a token — surpassing CNNs on ImageNet at sufficient data scale. The next class, [Vision Transformers](../vision-transformers/index.md), is dedicated to this idea.

---

## Scaling Laws and Large Language Models

Kaplan et al. (2020)[^4] discovered that language model loss follows power-law **scaling laws**:

$$
L(N, D) = \left(\frac{N_c}{N}\right)^{\alpha_N} + \left(\frac{D_c}{D}\right)^{\alpha_D} + L_\infty
$$

where $N$ is the number of parameters, $D$ the data size, and $L_\infty$ the irreducible loss.

This led to the **LLM** paradigm: training enormous models (billions of parameters) on trillions of tokens. The next class explores this world.

---

## Quick Implementation Reference

=== "PyTorch (Attention)"
    ```python
    import torch
    import torch.nn as nn

    class MultiHeadAttention(nn.Module):
        def __init__(self, d_model, num_heads):
            super().__init__()
            self.num_heads = num_heads
            self.d_k = d_model // num_heads
            self.W_q = nn.Linear(d_model, d_model)
            self.W_k = nn.Linear(d_model, d_model)
            self.W_v = nn.Linear(d_model, d_model)
            self.W_o = nn.Linear(d_model, d_model)

        def forward(self, q, k, v, mask=None):
            B, n, _ = q.shape
            Q = self.W_q(q).view(B, n, self.num_heads, self.d_k).transpose(1,2)
            K = self.W_k(k).view(B, -1, self.num_heads, self.d_k).transpose(1,2)
            V = self.W_v(v).view(B, -1, self.num_heads, self.d_k).transpose(1,2)

            scores = Q @ K.transpose(-2,-1) / self.d_k**0.5
            if mask is not None:
                scores = scores.masked_fill(mask==0, float('-inf'))
            attn = scores.softmax(dim=-1)
            out = (attn @ V).transpose(1,2).reshape(B, n, -1)
            return self.W_o(out)
    ```

=== "PyTorch (Transformer Block)"
    ```python
    class TransformerBlock(nn.Module):
        def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
            super().__init__()
            self.attn = MultiHeadAttention(d_model, num_heads)
            self.ff = nn.Sequential(
                nn.Linear(d_model, d_ff), nn.GELU(),
                nn.Linear(d_ff, d_model)
            )
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
            self.drop = nn.Dropout(dropout)

        def forward(self, x, mask=None):
            x = self.norm1(x + self.drop(self.attn(x, x, x, mask)))
            x = self.norm2(x + self.drop(self.ff(x)))
            return x
    ```

---

[^1]: Vaswani, A. et al. (2017). [Attention Is All You Need](https://arxiv.org/abs/1706.03762){:target="_blank"}. NeurIPS.
[^2]: Devlin, J. et al. (2019). [BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805){:target="_blank"}.
[^3]: Dosovitskiy, A. et al. (2020). [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929){:target="_blank"}.
[^4]: Kaplan, J. et al. (2020). [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361){:target="_blank"}.


---

--8<-- "docs/2026.2/classes/transformers/quiz.md"
