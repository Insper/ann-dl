## Mecanismos de Atenção

O mecanismo de atenção é uma das inovações mais impactantes na história do aprendizado profundo. Introduzido por Bahdanau et al. (2015)[^1] para tradução automática, ele permitiu que redes neurais aprendessem **onde olhar** em uma sequência de entrada — em vez de comprimir tudo em um único vetor de contexto.

Intuitivamente, atenção é o que você faz ao ler esta frase: seus olhos e cérebro não processam todas as palavras com igual peso. Ao interpretar "O gato sentou no **tapete** porque ele estava confortável", o pronome *ele* direciona a atenção para *tapete* — não *gato* ou *sentou*. Redes com atenção aprendem esse comportamento automaticamente.

---

## Intuição: Query, Key e Value

O mecanismo de atenção é formalizado por três conceitos: **Query (Q)**, **Key (K)** e **Value (V)**.

Pense na analogia de uma busca em banco de dados:

- **Query** — o que você está buscando (ex: vetor da palavra "it")
- **Key** — o índice de cada item disponível (ex: vetor de cada palavra na frase)
- **Value** — o conteúdo real retornado na correspondência (ex: representação semântica de cada palavra)

A atenção calcula um produto escalar entre a Query e cada Key, normaliza com *softmax* e usa os pesos resultantes para combinar os Values:

$$
\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
$$

O fator $\sqrt{d_k}$ estabiliza gradientes quando a dimensão $d_k$ é grande.

---

## Interativo: Atenção sobre uma Frase

Clique em qualquer palavra para ver como ela "presta atenção" às outras palavras. Os pesos são ilustrativos e pré-computados para demonstrar o conceito.

<div id="attention-demo" style="font-family: 'Inter', sans-serif; max-width: 860px; margin: 2rem auto; padding: 1.5rem; background: #0d1117; border-radius: 12px; color: #e6edf3;">

<div style="display: flex; gap: 8px; flex-wrap: wrap; justify-content: center; margin-bottom: 1.5rem;" id="word-buttons"></div>

<div style="position: relative; overflow: hidden; border-radius: 8px; background: #161b22; padding: 1rem;">
  <canvas id="attention-canvas" style="display:block; width:100%;"></canvas>
</div>

<div style="margin-top: 1rem; text-align: center; font-size: 0.85rem; color: #8b949e;" id="attention-label">← Clique em uma palavra acima</div>

</div>

<script>
(function() {
  const words = ["The", "cat", "sat", "on", "the", "mat", "because", "it", "was", "comfortable"];
  const weights = [
    [0.65, 0.12, 0.05, 0.03, 0.04, 0.04, 0.02, 0.02, 0.01, 0.02],
    [0.08, 0.52, 0.10, 0.05, 0.04, 0.08, 0.03, 0.06, 0.02, 0.02],
    [0.04, 0.12, 0.46, 0.08, 0.08, 0.10, 0.05, 0.04, 0.02, 0.01],
    [0.03, 0.05, 0.06, 0.50, 0.18, 0.12, 0.03, 0.01, 0.01, 0.01],
    [0.04, 0.05, 0.06, 0.14, 0.48, 0.14, 0.03, 0.02, 0.02, 0.02],
    [0.03, 0.07, 0.07, 0.09, 0.10, 0.50, 0.04, 0.05, 0.03, 0.02],
    [0.02, 0.04, 0.08, 0.04, 0.05, 0.06, 0.48, 0.10, 0.07, 0.06],
    [0.02, 0.07, 0.05, 0.05, 0.06, 0.32, 0.05, 0.22, 0.09, 0.07],
    [0.02, 0.04, 0.09, 0.03, 0.09, 0.12, 0.07, 0.08, 0.38, 0.08],
    [0.02, 0.03, 0.05, 0.04, 0.16, 0.22, 0.07, 0.09, 0.16, 0.16],
  ];

  let selected = 7;
  const N = words.length;
  const container = document.getElementById('attention-demo');
  const btnsDiv = document.getElementById('word-buttons');
  const canvas = document.getElementById('attention-canvas');
  const label = document.getElementById('attention-label');
  const ctx = canvas.getContext('2d');

  words.forEach((w, i) => {
    const btn = document.createElement('button');
    btn.textContent = w;
    btn.style.cssText = 'padding:6px 14px;border-radius:20px;border:none;cursor:pointer;font-size:1rem;font-weight:600;transition:all .2s;';
    btn.onclick = () => { selected = i; render(); };
    btn.id = 'btn-' + i;
    btnsDiv.appendChild(btn);
  });

  function heatColor(v) {
    const r = Math.round(40 + v * 215), g = Math.round(80 + v * 85 * 0.6), b2 = Math.round(180 - v * 160);
    return 'rgb(' + r + ',' + g + ',' + b2 + ')';
  }

  function render() {
    const W = container.offsetWidth - 32;
    const cellW = Math.max(38, Math.floor(W / N));
    const cellH = 56;
    canvas.width = cellW * N; canvas.height = cellH + 36;
    canvas.style.height = (cellH + 36) + 'px';
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    const row = weights[selected];

    for (let j = 0; j < N; j++) {
      const v = row[j], x = j * cellW;
      ctx.fillStyle = heatColor(v);
      ctx.beginPath(); ctx.roundRect(x + 2, 2, cellW - 4, cellH - 4, 8); ctx.fill();
      ctx.fillStyle = v > 0.4 ? '#0d1117' : '#e6edf3';
      ctx.font = 'bold ' + Math.max(11, cellW / 5.5) + 'px Inter,sans-serif';
      ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
      ctx.fillText(words[j], x + cellW / 2, cellH / 2);
      ctx.fillStyle = v > 0.4 ? 'rgba(0,0,0,0.7)' : 'rgba(255,255,255,0.5)';
      ctx.font = Math.max(9, cellW / 7) + 'px monospace';
      ctx.fillText(v.toFixed(2), x + cellW / 2, cellH * 0.78);
    }

    const maxIdx = row.indexOf(Math.max(...row));
    if (maxIdx !== selected) {
      const sx = selected * cellW + cellW / 2, ex = maxIdx * cellW + cellW / 2, y0 = cellH + 4;
      ctx.beginPath(); ctx.moveTo(sx, y0); ctx.bezierCurveTo(sx, y0 + 28, ex, y0 + 28, ex, y0);
      ctx.strokeStyle = '#f0883e'; ctx.lineWidth = 2.5; ctx.setLineDash([5, 3]); ctx.stroke(); ctx.setLineDash([]);
      ctx.fillStyle = '#f0883e'; ctx.beginPath(); ctx.moveTo(ex, y0); ctx.lineTo(ex - 6, y0 + 8); ctx.lineTo(ex + 6, y0 + 8); ctx.fill();
    }

    words.forEach((_, i) => {
      const btn = document.getElementById('btn-' + i);
      btn.style.background = i === selected ? '#f0883e' : '#21262d';
      btn.style.color = i === selected ? '#0d1117' : '#c9d1d9';
      btn.style.transform = i === selected ? 'scale(1.1)' : 'scale(1)';
    });
    label.textContent = '"' + words[selected] + '" presta mais atenção a "' + words[row.indexOf(Math.max(...row))] + '" (peso: ' + Math.max(...row).toFixed(2) + ')';
  }

  render(); window.addEventListener('resize', render);
})();
</script>

---

## Atenção por Produto Escalar Escalado — Passo a Passo

Dado um conjunto de vetores de entrada $X \in \mathbb{R}^{n \times d}$ (n tokens, dimensão d), matrizes de projeção $W_Q, W_K, W_V$ produzem:

$$
Q = X W_Q, \quad K = X W_K, \quad V = X W_V
$$

**Passo 1 — Pontuações de similaridade:**

$$
S = \frac{Q K^\top}{\sqrt{d_k}} \in \mathbb{R}^{n \times n}
$$

**Passo 2 — Normalização Softmax:**

$$
A = \text{softmax}(S), \quad A_{ij} = \frac{e^{S_{ij}}}{\sum_k e^{S_{ik}}}
$$

**Passo 3 — Saída ponderada:**

$$
\text{Saída} = A \cdot V
$$

A matriz $A$ é a **matriz de atenção**: cada linha soma 1 e representa quanto o token $i$ presta atenção a todos os outros tokens.

---

## Playground de Pesos de Atenção

O playground abaixo calcula atenção com vetores bidimensionais. Ajuste os valores e observe os pesos mudarem.

<div id="attn-playground" style="background:#0d1117;border-radius:12px;padding:1.5rem;margin:2rem 0;color:#e6edf3;font-family:monospace;">

<div style="display:grid;grid-template-columns:1fr 1fr;gap:1rem;margin-bottom:1.2rem;">
  <div>
    <div style="color:#8b949e;margin-bottom:.4rem;font-size:.85rem;">Q (query): [q1, q2]</div>
    <div style="display:flex;gap:.5rem;">
      <input id="q1" type="range" min="-2" max="2" step="0.1" value="1" style="flex:1;accent-color:#58a6ff">
      <input id="q2" type="range" min="-2" max="2" step="0.1" value="0.5" style="flex:1;accent-color:#58a6ff">
    </div>
    <div id="q-val" style="font-size:.8rem;color:#58a6ff;"></div>
  </div>
  <div>
    <div style="color:#8b949e;margin-bottom:.4rem;font-size:.85rem;">d_k (dimensão das chaves)</div>
    <input id="dk" type="range" min="1" max="8" step="1" value="2" style="width:100%;accent-color:#3fb950">
    <div id="dk-val" style="font-size:.8rem;color:#3fb950;"></div>
  </div>
</div>

<div style="margin-bottom:1rem;">
  <div style="color:#8b949e;margin-bottom:.4rem;font-size:.85rem;">Chaves K (3 tokens × 2 dims)</div>
  <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:.5rem;">
    <div>K1: <input id="k1a" type="number" value="1.2" step="0.1" style="width:55px;background:#161b22;color:#e6edf3;border:1px solid #30363d;border-radius:4px;padding:2px 4px;">, <input id="k1b" type="number" value="0.3" step="0.1" style="width:55px;background:#161b22;color:#e6edf3;border:1px solid #30363d;border-radius:4px;padding:2px 4px;"></div>
    <div>K2: <input id="k2a" type="number" value="-0.5" step="0.1" style="width:55px;background:#161b22;color:#e6edf3;border:1px solid #30363d;border-radius:4px;padding:2px 4px;">, <input id="k2b" type="number" value="1.8" step="0.1" style="width:55px;background:#161b22;color:#e6edf3;border:1px solid #30363d;border-radius:4px;padding:2px 4px;"></div>
    <div>K3: <input id="k3a" type="number" value="0.8" step="0.1" style="width:55px;background:#161b22;color:#e6edf3;border:1px solid #30363d;border-radius:4px;padding:2px 4px;">, <input id="k3b" type="number" value="-1.0" step="0.1" style="width:55px;background:#161b22;color:#e6edf3;border:1px solid #30363d;border-radius:4px;padding:2px 4px;"></div>
  </div>
</div>

<div id="attn-output" style="background:#161b22;border-radius:8px;padding:1rem;font-size:.85rem;line-height:1.8;"></div>
</div>

<script>
(function() {
  function softmax(arr) { const m=Math.max(...arr),e=arr.map(x=>Math.exp(x-m)),s=e.reduce((a,b)=>a+b,0); return e.map(x=>x/s); }
  function dot(a,b) { return a.reduce((s,v,i)=>s+v*b[i],0); }
  function fmt(x) { return x.toFixed(4); }

  function compute() {
    const q = [+document.getElementById('q1').value, +document.getElementById('q2').value];
    const dk = +document.getElementById('dk').value;
    const keys = [
      [+document.getElementById('k1a').value, +document.getElementById('k1b').value],
      [+document.getElementById('k2a').value, +document.getElementById('k2b').value],
      [+document.getElementById('k3a').value, +document.getElementById('k3b').value],
    ];
    document.getElementById('q-val').textContent = 'Q = [' + q[0].toFixed(1) + ', ' + q[1].toFixed(1) + ']';
    document.getElementById('dk-val').textContent = 'sqrt(d_k) = ' + Math.sqrt(dk).toFixed(3);
    const scores = keys.map(k => dot(q, k) / Math.sqrt(dk));
    const attn = softmax(scores);
    const bars = attn.map((a, i) => {
      const w = Math.round(a * 120), col = a > 0.5 ? '#f0883e' : a > 0.3 ? '#d29922' : '#58a6ff';
      return '<div style="margin:.3rem 0;"><span style="display:inline-block;width:20px;color:#8b949e;">K' + (i+1) + '</span><span style="display:inline-block;background:' + col + ';width:' + w + 'px;height:14px;border-radius:3px;vertical-align:middle;transition:width .3s;"></span><span style="color:' + col + ';margin-left:.5rem;">score=' + fmt(scores[i]) + ' → attn=' + fmt(a) + '</span></div>';
    }).join('');
    document.getElementById('attn-output').innerHTML = '<div style="margin-bottom:.8rem;color:#8b949e;">Pontuações brutas: Q·Kᵢᵀ / sqrt(d_k)</div><div style="margin-bottom:1rem;color:#c9d1d9;">S₁=' + fmt(scores[0]) + ', S₂=' + fmt(scores[1]) + ', S₃=' + fmt(scores[2]) + '</div><div style="margin-bottom:.8rem;color:#8b949e;">Após softmax (pesos de atenção):</div>' + bars + '<div style="margin-top:.8rem;color:#3fb950;font-size:.8rem;">∑ pesos = ' + fmt(attn.reduce((a,b)=>a+b,0)) + '</div>';
  }

  ['q1','q2','dk','k1a','k1b','k2a','k2b','k3a','k3b'].forEach(id => document.getElementById(id).addEventListener('input', compute));
  compute();
})();
</script>

---

## Atenção Multi-Cabeça

Uma única cabeça de atenção captura um tipo de relacionamento entre tokens. A **Atenção Multi-Cabeça** executa $h$ cabeças em paralelo, cada uma com projeções independentes $W_Q^i, W_K^i, W_V^i$:

$$
\text{MultiHead}(Q,K,V) = \text{Concat}(\text{cabeça}_1, \ldots, \text{cabeça}_h)\, W^O
$$

$$
\text{cabeça}_i = \text{Attention}(Q W_Q^i,\; K W_K^i,\; V W_V^i)
$$

Cada cabeça pode especializar-se: uma captura dependências sintáticas, outra correferências, outra padrões posicionais.

<div style="background:#0d1117;border-radius:12px;padding:1.5rem;margin:2rem 0;">
<canvas id="mha-canvas" style="width:100%;display:block;"></canvas>
</div>

<script>
(function() {
  const canvas = document.getElementById('mha-canvas');
  const ctx = canvas.getContext('2d');
  const W = 700, H = 220; canvas.width = W; canvas.height = H; canvas.style.height = H + 'px';
  const heads = 4, colors = ['#58a6ff','#3fb950','#f0883e','#bc8cff'];
  const labels = ['Sintático','Semântico','Posicional','Correferência'];
  const tokens = ['the','cat','sat','on','mat'];
  ctx.fillStyle = '#0d1117'; ctx.fillRect(0,0,W,H);
  const tokenSpacing = (W - 100) / tokens.length;
  const tokenY = 30;
  tokens.forEach((t, i) => {
    const x = 50 + i * tokenSpacing + tokenSpacing/2;
    ctx.fillStyle = '#21262d'; ctx.beginPath(); ctx.roundRect(x-22, tokenY-12, 44, 24, 5); ctx.fill();
    ctx.fillStyle = '#c9d1d9'; ctx.font = '12px monospace'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle'; ctx.fillText(t, x, tokenY);
  });
  for (let h = 0; h < heads; h++) {
    const y = 80 + h * 34;
    ctx.fillStyle = colors[h]; ctx.font = 'bold 10px Inter,sans-serif'; ctx.textAlign = 'left';
    ctx.fillText('Cabeça ' + (h+1) + ': ' + labels[h], 4, y + 4);
    tokens.forEach((_, i) => {
      const x = 50 + i * tokenSpacing + tokenSpacing/2;
      ctx.fillStyle = colors[h] + '33'; ctx.beginPath(); ctx.arc(x, y, 8, 0, 2*Math.PI); ctx.fill();
      ctx.strokeStyle = colors[h]; ctx.lineWidth = 1.5; ctx.stroke();
    });
    const pairs = [[[1,3],[2,4],[0,2]],[[0,4],[1,4],[2,3],[3,4]],[[4,0],[3,1],[2,2]],[[0,1],[2,3],[1,4]]][h];
    pairs.forEach(([from, to]) => {
      if (to >= tokens.length) return;
      const x1 = 50+from*tokenSpacing+tokenSpacing/2, x2 = 50+to*tokenSpacing+tokenSpacing/2;
      ctx.beginPath(); ctx.moveTo(x1,y); ctx.bezierCurveTo(x1,y-15,x2,y-15,x2,y);
      ctx.strokeStyle = colors[h] + 'aa'; ctx.lineWidth = 1.5; ctx.stroke();
    });
  }
  ctx.fillStyle = '#8b949e'; ctx.font = '11px Inter,sans-serif'; ctx.textAlign = 'center';
  ctx.fillText('→  Concat + Linear (W^O)  →  Saída', W/2, H - 12);
})();
</script>

---

## Self-Attention vs. Cross-Attention

| Tipo | Q de | K, V de | Uso típico |
|------|--------|-----------|-------------|
| **Self-Attention** | mesma sequência | mesma sequência | Encoder Transformer, BERT |
| **Cross-Attention** | sequência alvo | sequência fonte | Decoder Transformer, CLIP |
| **Causal Self-Attention** | mesma seq (mascarada) | mesma seq | GPT, decoders autorregressivos |

Na **causal self-attention**, uma máscara triangular inferior é aplicada antes do softmax, impedindo que o token $i$ veja tokens futuros $j > i$:

$$
M_{ij} = \begin{cases} 0 & \text{se } j \leq i \\ -\infty & \text{se } j > i \end{cases}
$$

---

## Codificação Posicional

A atenção é **invariante a permutações** — embaralhar tokens não afeta as pontuações. Para injetar informação de posição, o Transformer original usa codificações senoidais:

$$
PE_{(pos, 2i)} = \sin\!\left(\frac{pos}{10000^{2i/d}}\right), \quad
PE_{(pos, 2i+1)} = \cos\!\left(\frac{pos}{10000^{2i/d}}\right)
$$

Modelos modernos (LLaMA, GPT-4) usam **RoPE (Rotary Position Embedding)**, que aplica rotações aos vetores Q e K, capturando posição relativa de forma mais eficaz.

---

## Complexidade e Eficiência

A atenção padrão é $O(n^2 d)$ em tempo e memória — cara para sequências longas. Variantes eficientes:

| Método | Complexidade | Ideia |
|--------|-----------|------|
| **Softmax Attention** (padrão) | $O(n^2)$ | Matriz de atenção completa |
| **Sparse Attention** | $O(n\sqrt{n})$ | Atenção local + global |
| **Linear Attention** | $O(n)$ | Decomposição por kernel |
| **FlashAttention** | $O(n^2)$ tempo, $O(n)$ memória | Tiling eficiente em SRAM |

FlashAttention[^3] é o padrão moderno: matematicamente idêntico, mas reordena cálculos para minimizar transferências HBM↔SRAM na GPU.

---

[^1]: Bahdanau, D., Cho, K., & Bengio, Y. (2015). [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473){:target="_blank"}.
[^2]: Vaswani, A. et al. (2017). [Attention Is All You Need](https://arxiv.org/abs/1706.03762){:target="_blank"}.
[^3]: Dao, T. et al. (2022). [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135){:target="_blank"}.

---

--8<-- "docs/2026.2/classes/attention/quiz.pt.md"
