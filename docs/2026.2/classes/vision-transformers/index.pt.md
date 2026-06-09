## Vision Transformers (ViT)

Na aula de [Transformers](../transformers/index.md) construímos uma arquitetura para sequências de *tokens de texto*. Em 2021, Dosovitskiy et al.[^1] fizeram uma pergunta provocadora: e se alimentássemos uma **imagem** nesse mesmo encoder? A resposta — "An Image is Worth 16×16 Words" — mostrou que, com dados suficientes, um Transformer quase sem modificações pode **superar** as [Redes Neurais Convolucionais](../convolutional-neural-networks/index.md) na classificação de imagens, sem nenhuma convolução.

O ViT é a ponte entre o mundo convolucional (aula 11) e o mundo dos Transformers (aula 13). Uma vez que você o entende, os codificadores de imagem dentro do [CLIP](../clip/index.md), do [Stable Diffusion](../stable-diffusion/index.md) e dos [Diffusion Transformers](../diffusion-transformers/index.md) deixam de ser caixas-pretas.

---

## O trade-off do viés indutivo

Uma CNN traz dois fortes *vieses indutivos* embutidos na arquitetura:

- **Localidade** — um kernel convolucional olha apenas para uma pequena vizinhança de pixels.
- **Equivariância à translação** — o mesmo filtro desliza por toda a imagem, então uma feature é detectada independentemente de *onde* ela aparece.

Esses *priors* são exatamente o motivo pelo qual CNNs aprendem com tanta eficiência a partir de datasets *pequenos*: a arquitetura já "sabe" que pixels próximos importam e que objetos podem se deslocar.

Uma camada de self-attention pura **não** tem nenhum desses *priors*. Cada patch pode atender a todos os outros patches desde a camada 1, então o modelo precisa *aprender* as relações espaciais a partir dos dados. Isso é uma faca de dois gumes:

<div style="display:grid;grid-template-columns:1fr 1fr;gap:1rem;margin:1.5rem 0;">

<div style="background:#0d1117;border-left:3px solid #f0883e;padding:1rem;border-radius:8px;">
<strong style="color:#f0883e;">Fraqueza</strong><br><br>
<ul style="color:#c9d1d9;font-size:.9rem;margin:0;padding-left:1.2rem;">
<li>Precisa de <strong>muitos</strong> dados para aprender o que a CNN ganha de graça</li>
<li>Só no ImageNet-1k, uma CNN de tamanho similar vence</li>
</ul>
</div>

<div style="background:#0d1117;border-left:3px solid #3fb950;padding:1rem;border-radius:8px;">
<strong style="color:#3fb950;">Força</strong><br><br>
<ul style="color:#c9d1d9;font-size:.9rem;margin:0;padding-left:1.2rem;">
<li>Campo receptivo <strong>global</strong> desde a primeira camada</li>
<li>Com dados suficientes, supera as CNNs — o viés era um teto, não só um piso</li>
</ul>
</div>

</div>

---

## O pipeline do ViT, passo a passo

O truque é transformar uma imagem em uma *sequência de tokens* para que o encoder Transformer que já conhecemos possa consumi-la. Percorra o pipeline abaixo:

<div id="vit-viz" style="background:#0d1117;border-radius:12px;padding:1.5rem;margin:2rem 0;overflow:hidden;">
<canvas id="vit-canvas" style="width:100%;display:block;"></canvas>
<div style="display:flex;gap:1rem;justify-content:center;margin-top:1rem;flex-wrap:wrap;" id="vit-controls">
  <button onclick="vitStep(-1)" style="padding:6px 20px;background:#21262d;color:#c9d1d9;border:1px solid #30363d;border-radius:6px;cursor:pointer;">← Anterior</button>
  <span id="vit-step-label" style="color:#8b949e;font-family:monospace;line-height:2;"></span>
  <button onclick="vitStep(1)" style="padding:6px 20px;background:#21262d;color:#c9d1d9;border:1px solid #30363d;border-radius:6px;cursor:pointer;">Próximo →</button>
</div>
<div id="vit-desc" style="color:#c9d1d9;font-family:Inter,sans-serif;font-size:.9rem;text-align:center;margin-top:.8rem;min-height:2.5rem;padding:0 1rem;"></div>
</div>

<script>
(function() {
  const canvas = document.getElementById('vit-canvas');
  const ctx = canvas.getContext('2d');
  let currentStep = 0;

  const steps = [
    { label: "Imagem de entrada", desc: "Começamos com uma imagem de forma H×W×C — aqui um grid 4×4 representa os pixels." },
    { label: "Dividir em patches", desc: "Cortamos a imagem em um grid de patches de tamanho fixo (ex: 16×16 px). Cada patch vira uma 'palavra'." },
    { label: "Achatar + projeção linear", desc: "Achatamos cada patch e passamos por uma única camada Linear compartilhada → um embedding de patch (token) de dimensão d." },
    { label: "Prepor [CLS] + posições", desc: "Adicionamos um token [CLS] aprendível no início e somamos positional embeddings aprendíveis, para o modelo saber de onde veio cada patch." },
    { label: "Encoder Transformer", desc: "Alimentamos a sequência de tokens no MESMO encoder da aula de Transformers: L blocos de Multi-Head Self-Attention + FFN." },
    { label: "Cabeça MLP sobre [CLS]", desc: "Pegamos a representação final do [CLS] e passamos por uma cabeça MLP → probabilidades de classe." },
  ];

  window.vitStep = function(d) {
    currentStep = Math.max(0, Math.min(steps.length - 1, currentStep + d));
    render();
  };

  const palette = ['#58a6ff','#3fb950','#f0883e','#bc8cff','#db61a2','#e3b341','#39c5cf','#ff7b72',
                   '#a5d6ff','#56d364','#ffa657','#d2a8ff','#ff9bce','#f2cc60','#76e3ea','#ffa198'];

  function render() {
    const W = canvas.parentElement.offsetWidth - 48;
    const H = 300;
    canvas.width = W; canvas.height = H;
    canvas.style.height = H + 'px';
    ctx.fillStyle = '#0d1117';
    ctx.fillRect(0,0,W,H);

    const step = steps[currentStep];
    document.getElementById('vit-step-label').textContent = 'Passo ' + (currentStep+1) + '/' + steps.length + ': ' + step.label;
    document.getElementById('vit-desc').textContent = step.desc;

    const grid = 4;            // 4x4 = 16 patches
    const n = grid * grid;
    const imgTop = 40;

    // ----- ESQUERDA: a imagem / grid de patches -----
    const imgSize = Math.min(150, H - imgTop - 80);
    const cell = imgSize / grid;
    const imgX = 30, imgY = imgTop;
    const gap = currentStep >= 1 ? 4 : 0;

    ctx.fillStyle = '#8b949e';
    ctx.font = '12px Inter,sans-serif'; ctx.textAlign = 'center'; ctx.textBaseline = 'alphabetic';
    ctx.fillText(currentStep === 0 ? 'Imagem (H×W×C)' : 'Patches', imgX + imgSize/2, imgY - 12);

    for (let r = 0; r < grid; r++) {
      for (let c = 0; c < grid; c++) {
        const i = r * grid + c;
        const px = imgX + c * cell + (currentStep >= 1 ? c * gap : 0);
        const py = imgY + r * cell + (currentStep >= 1 ? r * gap : 0);
        ctx.fillStyle = currentStep === 0 ? '#1f6feb' : palette[i];
        ctx.beginPath(); ctx.roundRect(px, py, cell - 1, cell - 1, currentStep>=1?3:1); ctx.fill();
      }
    }

    if (currentStep === 0) { render0Arrows(W); return; }

    // ----- DIREITA: sequência de tokens -----
    const seqTop = imgY + 20;
    const tokW = 26, tokH = 26, tokGap = 6;
    const hasCls = currentStep >= 3;
    const tokens = hasCls ? n + 1 : n;
    const seqX = imgX + imgSize + 70;
    const avail = W - seqX - 30;
    const perRow = Math.max(1, Math.floor(avail / (tokW + tokGap)));

    ctx.fillStyle = '#8b949e'; ctx.font = '12px Inter,sans-serif'; ctx.textAlign = 'left';
    const seqLabel = currentStep <= 2 ? 'Embeddings de patch (N tokens)' :
                     currentStep === 3 ? '[CLS] + tokens + posição' : 'Sequência de tokens';
    ctx.fillText(seqLabel, seqX, seqTop - 10);

    // seta da imagem para a sequência
    ctx.strokeStyle = '#f0883e'; ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(imgX + imgSize + 12, imgY + imgSize/2);
    ctx.lineTo(seqX - 12, seqTop + 13);
    ctx.stroke();
    drawArrowHead(seqX - 12, seqTop + 13, 0);
    ctx.fillStyle = '#f0883e'; ctx.font = '10px monospace'; ctx.textAlign = 'center';
    ctx.fillText(currentStep >= 2 ? 'Linear' : 'achatar', (imgX+imgSize+seqX)/2, imgY + imgSize/2 - 8);

    for (let t = 0; t < tokens; t++) {
      const isCls = hasCls && t === 0;
      const patchIdx = hasCls ? t - 1 : t;
      const row = Math.floor(t / perRow);
      const col = t % perRow;
      const tx = seqX + col * (tokW + tokGap);
      const ty = seqTop + row * (tokH + tokGap + (currentStep>=3?12:0));

      ctx.fillStyle = isCls ? '#ffffff' : (currentStep <= 1 ? palette[patchIdx] : '#30363d');
      if (currentStep === 2 && !isCls) ctx.fillStyle = palette[patchIdx];
      ctx.beginPath(); ctx.roundRect(tx, ty, tokW, tokH, 4); ctx.fill();
      if (currentStep >= 4 && !isCls) {
        ctx.strokeStyle = '#58a6ff'; ctx.lineWidth = 1; ctx.stroke();
      }
      if (isCls) {
        ctx.fillStyle = '#0d1117'; ctx.font = 'bold 9px monospace'; ctx.textAlign = 'center'; ctx.textBaseline='middle';
        ctx.fillText('CLS', tx + tokW/2, ty + tokH/2);
        ctx.textBaseline='alphabetic';
      }
      // marcador de positional embedding
      if (currentStep >= 3) {
        ctx.fillStyle = '#8b949e'; ctx.font = '8px monospace'; ctx.textAlign='center';
        ctx.fillText(isCls ? '0' : String(patchIdx+1), tx + tokW/2, ty + tokH + 9);
      }
    }

    const seqBottom = seqTop + Math.ceil(tokens/perRow) * (tokH + tokGap + (currentStep>=3?12:0));

    if (currentStep >= 4) {
      // barra do bloco encoder
      const by = seqBottom + 18;
      ctx.fillStyle = currentStep === 4 ? '#f0883e' : '#1f3244';
      ctx.beginPath(); ctx.roundRect(seqX, by, Math.min(avail, perRow*(tokW+tokGap)), 30, 6); ctx.fill();
      ctx.fillStyle = currentStep === 4 ? '#0d1117' : '#58a6ff';
      ctx.font = 'bold 12px Inter,sans-serif'; ctx.textAlign = 'center'; ctx.textBaseline='middle';
      ctx.fillText('Encoder Transformer  ×L  (MHSA + FFN)', seqX + Math.min(avail, perRow*(tokW+tokGap))/2, by + 15);
      ctx.textBaseline='alphabetic';

      if (currentStep === 5) {
        const hy = by + 44;
        ctx.fillStyle = '#3fb950';
        ctx.beginPath(); ctx.roundRect(seqX, hy, 160, 26, 6); ctx.fill();
        ctx.fillStyle = '#0d1117'; ctx.font = 'bold 11px Inter,sans-serif'; ctx.textAlign='center'; ctx.textBaseline='middle';
        ctx.fillText('Cabeça MLP → classes', seqX + 80, hy + 13);
        ctx.textBaseline='alphabetic';
        // destaca o CLS alimentando a cabeça
        ctx.strokeStyle = '#3fb950'; ctx.lineWidth = 1.5; ctx.setLineDash([3,3]);
        ctx.beginPath(); ctx.moveTo(seqX + 13, by + 30); ctx.lineTo(seqX + 13, hy); ctx.stroke();
        ctx.setLineDash([]);
      }
    }
  }

  function render0Arrows(W) {
    ctx.fillStyle = '#484f58'; ctx.font = '13px Inter,sans-serif'; ctx.textAlign = 'center'; ctx.textBaseline='middle';
    ctx.fillText('Clique em "Próximo →" para dividir a imagem em patches', W*0.68, 150);
    ctx.textBaseline='alphabetic';
  }

  function drawArrowHead(x, y, angle) {
    ctx.save(); ctx.translate(x, y); ctx.rotate(angle);
    ctx.fillStyle = '#f0883e';
    ctx.beginPath(); ctx.moveTo(0,0); ctx.lineTo(-8,-4); ctx.lineTo(-8,4); ctx.closePath(); ctx.fill();
    ctx.restore();
  }

  render();
  window.addEventListener('resize', render);
})();
</script>

---

## Patch embedding — a única peça realmente nova

Tudo depois do primeiro passo é o encoder que você já conhece. O único mecanismo novo é como a imagem vira tokens.

Uma imagem $x \in \mathbb{R}^{H \times W \times C}$ é remodelada em uma sequência de $N$ patches achatados e, em seguida, cada patch é projetado por uma única camada linear compartilhada:

$$
x \in \mathbb{R}^{H \times W \times C}
\;\xrightarrow{\text{patchify}}\;
x_p \in \mathbb{R}^{N \times (P^2 C)}
\;\xrightarrow{\;E\;}\;
z \in \mathbb{R}^{N \times d}
$$

onde $P$ é o tamanho do patch, $N = HW/P^2$ é o número de patches e $E \in \mathbb{R}^{(P^2 C) \times d}$ é a matriz de **patch embedding**. Para uma imagem $224 \times 224$ com $P = 16$, isso dá $N = 196$ tokens.

Um token **`[CLS]`** aprendível $z_{\text{cls}}$ é adicionado ao início e **positional embeddings** aprendíveis $E_{\text{pos}}$ são somados (a atenção sozinha é invariante a permutação, então sem posições o modelo não conseguiria distinguir um patch no canto superior esquerdo de um no canto inferior direito):

$$
z_0 = [\, z_{\text{cls}};\; x_p^1 E;\; x_p^2 E;\; \dots;\; x_p^N E \,] + E_{\text{pos}}
$$

> **Nota.** Uma projeção linear sobre patches $P \times P$ não-sobrepostos é matematicamente idêntica a uma `Conv2d` com `kernel_size = stride = P`. É exatamente assim que se implementa na prática — uma única convolução faz o patchify *e* a projeção em uma só operação.

---

## O encoder e a cabeça de classificação

A sequência $z_0$ passa por $L$ blocos **encoder** Transformer idênticos — os mesmos blocos MHSA + Add&Norm + FFN da aula de [Transformers](../transformers/index.md) (o ViT usa a variante pre-norm e GELU):

$$
z'_\ell = \text{MHSA}(\text{LN}(z_{\ell-1})) + z_{\ell-1}, \qquad
z_\ell   = \text{FFN}(\text{LN}(z'_\ell)) + z'_\ell
$$

Para classificação, apenas o estado final do token `[CLS]` é lido e passado por uma pequena cabeça MLP:

$$
y = \text{cabeça MLP}\big(\text{LN}(z_L^{0})\big)
$$

É o modelo inteiro. Sem convoluções, sem pirâmides de pooling — apenas patchify e, em seguida, um encoder Transformer padrão.

---

## Fome de dados: por que o ViT precisa de pré-treinamento

Como lhe faltam os *priors* convolucionais, o ViT só brilha em **escala**. O artigo original tornou isso concreto:

<div style="background:#161b22;border-radius:8px;padding:1.2rem;margin:1.5rem 0;color:#c9d1d9;font-size:.92rem;">
<ul style="margin:0;padding-left:1.2rem;">
<li>Treinado apenas no <strong>ImageNet-1k</strong> (~1,3M imagens), o ViT fica <em>abaixo</em> de uma ResNet comparável.</li>
<li>Pré-treinado no <strong>ImageNet-21k</strong> (~14M) ele se iguala.</li>
<li>Pré-treinado no <strong>JFT-300M</strong> (~300M) ele <em>supera</em> as melhores CNNs e transfere muito bem para tarefas posteriores.</li>
</ul>
</div>

Essa é exatamente a receita de **pré-treinar e depois ajustar (finetune)** da aula de [Transfer Learning](../transfer-learning/index.md): pré-treinar o encoder em um dataset enorme e depois ajustar a cabeça MLP barata (ou o modelo inteiro com taxa de aprendizado baixa) na sua tarefa. É também por isso que o CLIP pôde treinar um encoder de imagem ViT em 400M pares imagem-texto — nessa escala, o viés indutivo fraco vira vantagem.

---

## CNN vs. ViT em um relance

| | CNN | Vision Transformer |
|---|---|---|
| Operação central | Convolução (local) | Self-attention (global) |
| Viés indutivo | Forte (localidade, equivariância) | Fraco — aprendido dos dados |
| Campo receptivo | Cresce com a profundidade | Global desde a camada 1 |
| Eficiência em dados | Forte em datasets pequenos | Precisa de pré-treinamento em larga escala |
| Custo computacional | $O(N)$ em pixels | $O(N^2)$ em patches |
| Escala com dados | Satura mais cedo | Continua melhorando |

**Híbridos e sucessores.** Várias variantes reintroduzem *algum* viés espacial para obter o melhor dos dois mundos: o **DeiT** (treinamento eficiente em dados com destilação, sem precisar do JFT) e o **Swin Transformer** (atenção em janelas com estrutura hierárquica, tipo pirâmide, que devolve a localidade e torna o ViT prático para detecção e segmentação).

---

## Referência de implementação

=== "PyTorch (patch embedding)"
    ```python
    import torch
    import torch.nn as nn

    class PatchEmbed(nn.Module):
        """Imagem -> sequência de tokens (Conv2d faz patchify + projeção)."""
        def __init__(self, img_size=224, patch=16, in_ch=3, dim=768):
            super().__init__()
            self.n_patches = (img_size // patch) ** 2
            self.proj = nn.Conv2d(in_ch, dim, kernel_size=patch, stride=patch)

        def forward(self, x):                  # x: (B, C, H, W)
            x = self.proj(x)                   # (B, dim, H/p, W/p)
            return x.flatten(2).transpose(1, 2)  # (B, N, dim)
    ```

=== "PyTorch (forward do ViT)"
    ```python
    class ViT(nn.Module):
        def __init__(self, dim=768, depth=12, heads=12, n_classes=1000):
            super().__init__()
            self.patch_embed = PatchEmbed(dim=dim)
            n = self.patch_embed.n_patches
            self.cls = nn.Parameter(torch.zeros(1, 1, dim))
            self.pos = nn.Parameter(torch.zeros(1, n + 1, dim))
            layer = nn.TransformerEncoderLayer(dim, heads, dim * 4,
                                               activation='gelu', norm_first=True,
                                               batch_first=True)
            self.encoder = nn.TransformerEncoder(layer, depth)
            self.head = nn.Linear(dim, n_classes)

        def forward(self, x):
            B = x.size(0)
            x = self.patch_embed(x)                       # (B, N, dim)
            cls = self.cls.expand(B, -1, -1)
            x = torch.cat([cls, x], dim=1) + self.pos     # prepor CLS + somar posições
            x = self.encoder(x)
            return self.head(x[:, 0])                      # classifica a partir do token CLS
    ```

=== "Usar um ViT pré-treinado"
    ```python
    import timm
    # carrega pesos pré-treinados ImageNet-21k -> 1k e ajusta a cabeça
    model = timm.create_model('vit_base_patch16_224', pretrained=True,
                              num_classes=10)
    ```

---

[^1]: Dosovitskiy, A. et al. (2021). [An Image is Worth 16×16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929){:target="_blank"}. ICLR.
[^2]: Touvron, H. et al. (2021). [Training data-efficient image transformers & distillation through attention (DeiT)](https://arxiv.org/abs/2012.12877){:target="_blank"}. ICML.
[^3]: Liu, Z. et al. (2021). [Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/abs/2103.14030){:target="_blank"}. ICCV.


---

--8<-- "docs/2026.2/classes/vision-transformers/quiz.pt.md"
