## Transformers

Em 2017, o artigo "Attention Is All You Need"[^1] eliminou recorrência e convoluções dos modelos de sequência, substituindo-os inteiramente por atenção. O resultado foi o **Transformer** — a arquitetura que impulsiona GPT, BERT, ViT, Whisper e virtualmente todo modelo de estado da arte hoje.

---

## Visão Geral da Arquitetura

O Transformer original é um modelo *encoder-decoder* para tradução automática. Hoje existem variantes apenas-encoder (BERT, para classificação/representação) e apenas-decoder (GPT, para geração).

<div id="transformer-viz" style="background:#0d1117;border-radius:12px;padding:1.5rem;margin:2rem 0;overflow:hidden;">
<canvas id="tf-canvas" style="width:100%;display:block;"></canvas>
<div style="display:flex;gap:1rem;justify-content:center;margin-top:1rem;flex-wrap:wrap;" id="tf-controls">
  <button onclick="tfStep(-1)" style="padding:6px 20px;background:#21262d;color:#c9d1d9;border:1px solid #30363d;border-radius:6px;cursor:pointer;">← Anterior</button>
  <span id="tf-step-label" style="color:#8b949e;font-family:monospace;line-height:2;"></span>
  <button onclick="tfStep(1)" style="padding:6px 20px;background:#21262d;color:#c9d1d9;border:1px solid #30363d;border-radius:6px;cursor:pointer;">Próximo →</button>
</div>
<div id="tf-desc" style="color:#c9d1d9;font-family:Inter,sans-serif;font-size:.9rem;text-align:center;margin-top:.8rem;min-height:2.5rem;padding:0 1rem;"></div>
</div>

<script>
(function() {
  const canvas = document.getElementById('tf-canvas');
  const ctx = canvas.getContext('2d');
  let currentStep = 0;

  const steps = [
    { label: "Entrada", desc: "Tokens de entrada são convertidos em embeddings e somados ao Positional Encoding." },
    { label: "Atenção Multi-Cabeça", desc: "Self-attention: cada token atende a todos os outros. 8 cabeças paralelas capturam diferentes tipos de relacionamento." },
    { label: "Somar & Normalizar", desc: "Conexão residual soma entrada + saída da atenção. Layer Normalization estabiliza o treinamento." },
    { label: "Feed-Forward", desc: "FFN: duas camadas lineares com ReLU. Aplica transformações não-lineares token por token." },
    { label: "Somar & Normalizar", desc: "Segundo bloco residual + normalização. Cada bloco encoder repete os passos 2-4 N vezes (tipicamente N=6)." },
    { label: "Atenção Cruzada (Decoder)", desc: "O decoder usa Q das saídas geradas, mas K e V vêm do Encoder — conectando origem e alvo." },
    { label: "Linear de Saída + Softmax", desc: "Projeção final para vocabulário → softmax → probabilidades → próximo token (geração autorregressiva)." },
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
    document.getElementById('tf-step-label').textContent = 'Passo ' + (currentStep+1) + '/' + steps.length + ': ' + step.label;
    document.getElementById('tf-desc').textContent = step.desc;

    const cx = W / 2;
    const activeColor = '#f0883e';
    const dimColor = '#21262d';
    const dimText = '#484f58';

    const components = [
      { label: "Embeddings de Entrada + Pos. Enc.", y: 20, h: 32, step: 0 },
      { label: "Atenção Multi-Cabeça (Self)", y: 68, h: 36, step: 1 },
      { label: "Somar & Normalizar", y: 116, h: 24, step: 2 },
      { label: "Feed-Forward Network", y: 152, h: 36, step: 3 },
      { label: "Somar & Normalizar", y: 200, h: 24, step: 4 },
      { label: "Atenção Cruzada (Decoder)", y: 240, h: 36, step: 5 },
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
        ["8 cabeças", "d_model=512", "d_k=64", "O(n²d)"] :
        ["Q ← Decoder", "K,V ← Encoder", "Alinha", "fonte↔alvo"];
      notes.forEach((n,i) => ctx.fillText(n, x0+boxW+70, 82+i*18));
    }
  }

  render();
  window.addEventListener('resize', render);
})();
</script>

---

## O Bloco Encoder em Detalhe

Cada um dos $N$ blocos encoder idênticos consiste em:

$$
\text{SubCamada}_1(x) = \text{LayerNorm}(x + \text{MultiHead}(x, x, x))
$$

$$
\text{SubCamada}_2(x) = \text{LayerNorm}(x + \text{FFN}(x))
$$

A **FFN** (Feed-Forward Network) é aplicada independentemente a cada posição:

$$
\text{FFN}(x) = \max(0,\; xW_1 + b_1)W_2 + b_2
$$

Tipicamente $d_{\text{model}} = 512$, $d_{\text{ff}} = 2048$ — um *gargalo* 4× mais largo que o embedding.

**Conexões residuais** (inspiradas em ResNets) garantem que gradientes fluam diretamente por muitas camadas. **Layer Normalization** normaliza ao longo da dimensão de features (não do batch), o que é mais estável para sequências de comprimento variável.

---

## Decoder e Geração Autorregressiva

O decoder gera tokens um de cada vez, condicionado em tudo o que foi gerado até o momento:

$$
p(\text{saída}) = \prod_{t=1}^{T} p(y_t \mid y_{<t},\; \text{saída do encoder})
$$

Para impedir que o token $t$ veja tokens futuros durante o treinamento, a **self-attention mascarada** aplica uma máscara causal:

<div style="background:#161b22;border-radius:8px;padding:1.2rem;margin:1.5rem 0;font-family:monospace;overflow-x:auto;">

```
Máscara causal (n=4 tokens):
     pos0  pos1  pos2  pos3
pos0 [ 0   -inf  -inf  -inf ]   (vê apenas a si mesmo)
pos1 [ 0    0   -inf  -inf ]    (vê pos0 e pos1)
pos2 [ 0    0    0   -inf ]
pos3 [ 0    0    0    0   ]     (vê tudo)
```

</div>

---

## BERT vs. GPT — Encoder vs. Decoder

<div style="display:grid;grid-template-columns:1fr 1fr;gap:1rem;margin:1.5rem 0;">

<div style="background:#0d1117;border-left:3px solid #58a6ff;padding:1rem;border-radius:8px;">
<strong style="color:#58a6ff;">BERT (Apenas Encoder)</strong><br><br>
<ul style="color:#c9d1d9;font-size:.9rem;margin:0;padding-left:1.2rem;">
<li>Bidirecional: vê o contexto esquerdo <em>e</em> direito</li>
<li>Pré-treinado com <strong>Masked Language Model</strong></li>
<li>Excelente para classificação, NER, QA</li>
<li>Não é generativo</li>
</ul>
</div>

<div style="background:#0d1117;border-left:3px solid #3fb950;padding:1rem;border-radius:8px;">
<strong style="color:#3fb950;">GPT (Apenas Decoder)</strong><br><br>
<ul style="color:#c9d1d9;font-size:.9rem;margin:0;padding-left:1.2rem;">
<li>Causal: vê apenas o contexto à esquerda</li>
<li>Pré-treinado com <strong>Predição do Próximo Token</strong></li>
<li>Excelente para geração de texto, chat, código</li>
<li>A escala leva a capacidades emergentes</li>
</ul>
</div>

</div>

---

## Vision Transformer (ViT)

O encoder não se limita a texto. Em 2020, Dosovitskiy et al.[^3] aplicaram essa mesma arquitetura a imagens dividindo-as em **patches** de $16 \times 16$ pixels e tratando cada patch como um token — superando CNNs no ImageNet com escala de dados suficiente. A próxima aula, [Vision Transformers](../vision-transformers/index.md), é dedicada a essa ideia.

---

## Leis de Escala e Modelos de Linguagem de Grande Escala

Kaplan et al. (2020)[^4] descobriram que a perda de modelos de linguagem segue **leis de escala** em lei de potência:

$$
L(N, D) = \left(\frac{N_c}{N}\right)^{\alpha_N} + \left(\frac{D_c}{D}\right)^{\alpha_D} + L_\infty
$$

onde $N$ é o número de parâmetros, $D$ o tamanho dos dados e $L_\infty$ a perda irredutível.

Isso levou ao paradigma dos **LLMs**: treinar modelos enormes (bilhões de parâmetros) em trilhões de tokens. A próxima aula explora esse mundo.

---

## Referência Rápida de Implementação

=== "PyTorch (Atenção)"
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

=== "PyTorch (Bloco Transformer)"
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

--8<-- "docs/2026.2/classes/transformers/quiz.pt.md"
