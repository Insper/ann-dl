## Vision Transformers (ViT)

In the [Transformers](../transformers/index.md) class we built an architecture for sequences of *text tokens*. In 2021, Dosovitskiy et al.[^1] asked a provocative question: what if we feed an **image** to that exact same encoder? Their answer — "An Image is Worth 16×16 Words" — showed that, with enough data, a near-vanilla Transformer can **beat** [Convolutional Neural Networks](../convolutional-neural-networks/index.md) at image classification, without any convolutions at all.

The ViT is the bridge between the convolutional world (class 11) and the Transformer world (class 13). Once you understand it, the image encoders inside [CLIP](../clip/index.md), [Stable Diffusion](../stable-diffusion/index.md) and [Diffusion Transformers](../diffusion-transformers/index.md) stop being black boxes.

---

## The inductive bias trade-off

A CNN comes with two strong *inductive biases* baked into its architecture:

- **Locality** — a convolutional kernel only looks at a small neighborhood of pixels.
- **Translation equivariance** — the same filter slides across the whole image, so a feature is detected regardless of *where* it appears.

These priors are exactly why CNNs learn so efficiently from *small* datasets: the architecture already "knows" that nearby pixels matter and that objects can move around.

A pure self-attention layer has **none** of these priors. Every patch can attend to every other patch from layer 1, so the model must *learn* spatial relationships from data. This is a double-edged sword:

<div style="display:grid;grid-template-columns:1fr 1fr;gap:1rem;margin:1.5rem 0;">

<div style="background:#0d1117;border-left:3px solid #f0883e;padding:1rem;border-radius:8px;">
<strong style="color:#f0883e;">Weakness</strong><br><br>
<ul style="color:#c9d1d9;font-size:.9rem;margin:0;padding-left:1.2rem;">
<li>Needs <strong>lots</strong> of data to learn what a CNN gets for free</li>
<li>On ImageNet-1k alone, a CNN of similar size wins</li>
</ul>
</div>

<div style="background:#0d1117;border-left:3px solid #3fb950;padding:1rem;border-radius:8px;">
<strong style="color:#3fb950;">Strength</strong><br><br>
<ul style="color:#c9d1d9;font-size:.9rem;margin:0;padding-left:1.2rem;">
<li><strong>Global</strong> receptive field from the very first layer</li>
<li>Given enough data, it surpasses CNNs — the bias was a ceiling, not just a floor</li>
</ul>
</div>

</div>

---

## The ViT pipeline, step by step

The trick is to turn an image into a *sequence of tokens* so the Transformer encoder we already know can consume it. Step through the pipeline below:

<div id="vit-viz" style="background:#0d1117;border-radius:12px;padding:1.5rem;margin:2rem 0;overflow:hidden;">
<canvas id="vit-canvas" style="width:100%;display:block;"></canvas>
<div style="display:flex;gap:1rem;justify-content:center;margin-top:1rem;flex-wrap:wrap;" id="vit-controls">
  <button onclick="vitStep(-1)" style="padding:6px 20px;background:#21262d;color:#c9d1d9;border:1px solid #30363d;border-radius:6px;cursor:pointer;">← Previous</button>
  <span id="vit-step-label" style="color:#8b949e;font-family:monospace;line-height:2;"></span>
  <button onclick="vitStep(1)" style="padding:6px 20px;background:#21262d;color:#c9d1d9;border:1px solid #30363d;border-radius:6px;cursor:pointer;">Next →</button>
</div>
<div id="vit-desc" style="color:#c9d1d9;font-family:Inter,sans-serif;font-size:.9rem;text-align:center;margin-top:.8rem;min-height:2.5rem;padding:0 1rem;"></div>
</div>

<script>
(function() {
  const canvas = document.getElementById('vit-canvas');
  const ctx = canvas.getContext('2d');
  let currentStep = 0;

  const steps = [
    { label: "Input image", desc: "Start with an image of shape H×W×C — here a 4×4 grid stands in for the pixels." },
    { label: "Split into patches", desc: "Cut the image into a grid of fixed-size patches (e.g. 16×16 px). Each patch becomes one 'word'." },
    { label: "Flatten + linear projection", desc: "Flatten each patch and pass it through a single shared Linear layer → a d-dimensional patch embedding (token)." },
    { label: "Prepend [CLS] + add positions", desc: "Prepend a learnable [CLS] token and add learnable positional embeddings so the model knows where each patch came from." },
    { label: "Transformer encoder", desc: "Feed the token sequence into the SAME encoder from the Transformers class: L blocks of Multi-Head Self-Attention + FFN." },
    { label: "MLP head on [CLS]", desc: "Take the final [CLS] representation, pass it through an MLP head → class probabilities." },
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
    document.getElementById('vit-step-label').textContent = 'Step ' + (currentStep+1) + '/' + steps.length + ': ' + step.label;
    document.getElementById('vit-desc').textContent = step.desc;

    const grid = 4;            // 4x4 = 16 patches
    const n = grid * grid;
    const cellH = 110;
    const imgTop = 40;

    // ----- LEFT: the image / patch grid -----
    const imgSize = Math.min(150, H - imgTop - 80);
    const cell = imgSize / grid;
    const imgX = 30, imgY = imgTop;
    const gap = currentStep >= 1 ? 4 : 0;

    ctx.fillStyle = '#8b949e';
    ctx.font = '12px Inter,sans-serif'; ctx.textAlign = 'center'; ctx.textBaseline = 'alphabetic';
    ctx.fillText(currentStep === 0 ? 'Image (H×W×C)' : 'Patches', imgX + imgSize/2, imgY - 12);

    for (let r = 0; r < grid; r++) {
      for (let c = 0; c < grid; c++) {
        const i = r * grid + c;
        const x = imgX + c * (cell + gap/grid*grid/ (grid)) ;
        const px = imgX + c * cell + (currentStep >= 1 ? c * gap : 0);
        const py = imgY + r * cell + (currentStep >= 1 ? r * gap : 0);
        ctx.fillStyle = currentStep === 0 ? '#1f6feb' : palette[i];
        ctx.beginPath(); ctx.roundRect(px, py, cell - 1, cell - 1, currentStep>=1?3:1); ctx.fill();
      }
    }

    if (currentStep === 0) { render0Arrows(W); return; }

    // ----- RIGHT: token sequence -----
    const seqTop = imgY + 20;
    const tokW = 26, tokH = 26, tokGap = 6;
    const hasCls = currentStep >= 3;
    const tokens = hasCls ? n + 1 : n;
    const seqX = imgX + imgSize + 70;
    const avail = W - seqX - 30;
    const perRow = Math.max(1, Math.floor(avail / (tokW + tokGap)));

    ctx.fillStyle = '#8b949e'; ctx.font = '12px Inter,sans-serif'; ctx.textAlign = 'left';
    const seqLabel = currentStep <= 2 ? 'Patch embeddings (N tokens)' :
                     currentStep === 3 ? '[CLS] + patch tokens + pos' : 'Token sequence';
    ctx.fillText(seqLabel, seqX, seqTop - 10);

    // arrow from image to sequence
    ctx.strokeStyle = '#f0883e'; ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(imgX + imgSize + 12, imgY + imgSize/2);
    ctx.lineTo(seqX - 12, seqTop + 13);
    ctx.stroke();
    drawArrowHead(seqX - 12, seqTop + 13, 0);
    ctx.fillStyle = '#f0883e'; ctx.font = '10px monospace'; ctx.textAlign = 'center';
    ctx.fillText(currentStep >= 2 ? 'Linear' : 'flatten', (imgX+imgSize+seqX)/2, imgY + imgSize/2 - 8);

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
      // positional embedding marker
      if (currentStep >= 3) {
        ctx.fillStyle = '#8b949e'; ctx.font = '8px monospace'; ctx.textAlign='center';
        ctx.fillText(isCls ? '0' : String(patchIdx+1), tx + tokW/2, ty + tokH + 9);
      }
    }

    const seqBottom = seqTop + Math.ceil(tokens/perRow) * (tokH + tokGap + (currentStep>=3?12:0));

    if (currentStep >= 4) {
      // encoder block bar
      const by = seqBottom + 18;
      ctx.fillStyle = currentStep === 4 ? '#f0883e' : '#1f3244';
      ctx.beginPath(); ctx.roundRect(seqX, by, Math.min(avail, perRow*(tokW+tokGap)), 30, 6); ctx.fill();
      ctx.fillStyle = currentStep === 4 ? '#0d1117' : '#58a6ff';
      ctx.font = 'bold 12px Inter,sans-serif'; ctx.textAlign = 'center'; ctx.textBaseline='middle';
      ctx.fillText('Transformer Encoder  ×L  (MHSA + FFN)', seqX + Math.min(avail, perRow*(tokW+tokGap))/2, by + 15);
      ctx.textBaseline='alphabetic';

      if (currentStep === 5) {
        const hy = by + 44;
        ctx.fillStyle = '#3fb950';
        ctx.beginPath(); ctx.roundRect(seqX, hy, 150, 26, 6); ctx.fill();
        ctx.fillStyle = '#0d1117'; ctx.font = 'bold 11px Inter,sans-serif'; ctx.textAlign='center'; ctx.textBaseline='middle';
        ctx.fillText('MLP Head → classes', seqX + 75, hy + 13);
        ctx.textBaseline='alphabetic';
        // highlight CLS feeding the head
        ctx.strokeStyle = '#3fb950'; ctx.lineWidth = 1.5; ctx.setLineDash([3,3]);
        ctx.beginPath(); ctx.moveTo(seqX + 13, by + 30); ctx.lineTo(seqX + 13, hy); ctx.stroke();
        ctx.setLineDash([]);
      }
    }
  }

  function render0Arrows(W) {
    ctx.fillStyle = '#484f58'; ctx.font = '13px Inter,sans-serif'; ctx.textAlign = 'center'; ctx.textBaseline='middle';
    ctx.fillText('Press "Next →" to patchify the image', W*0.68, 150);
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

## Patch embedding — the only genuinely new piece

Everything after the first step is the encoder you already know. The one new mechanism is how the image becomes tokens.

An image $x \in \mathbb{R}^{H \times W \times C}$ is reshaped into a sequence of $N$ flattened patches, then each patch is projected by a single shared linear layer:

$$
x \in \mathbb{R}^{H \times W \times C}
\;\xrightarrow{\text{patchify}}\;
x_p \in \mathbb{R}^{N \times (P^2 C)}
\;\xrightarrow{\;E\;}\;
z \in \mathbb{R}^{N \times d}
$$

where $P$ is the patch size, $N = HW/P^2$ is the number of patches, and $E \in \mathbb{R}^{(P^2 C) \times d}$ is the **patch embedding** matrix. For a $224 \times 224$ image with $P = 16$, that is $N = 196$ tokens.

A learnable **`[CLS]` token** $z_{\text{cls}}$ is prepended, and learnable **positional embeddings** $E_{\text{pos}}$ are added (attention alone is permutation-invariant, so without positions the model could not tell a patch in the top-left from one in the bottom-right):

$$
z_0 = [\, z_{\text{cls}};\; x_p^1 E;\; x_p^2 E;\; \dots;\; x_p^N E \,] + E_{\text{pos}}
$$

> **Note.** A linear projection over non-overlapping $P \times P$ patches is mathematically identical to a `Conv2d` with `kernel_size = stride = P`. That is exactly how it is implemented in practice — one convolution does the patchifying *and* the projection in a single op.

---

## The encoder and the classification head

The sequence $z_0$ goes through $L$ identical Transformer **encoder** blocks — the same MHSA + Add&Norm + FFN blocks from the [Transformers](../transformers/index.md) class (ViT uses the pre-norm variant and GELU):

$$
z'_\ell = \text{MHSA}(\text{LN}(z_{\ell-1})) + z_{\ell-1}, \qquad
z_\ell   = \text{FFN}(\text{LN}(z'_\ell)) + z'_\ell
$$

For classification, only the final state of the `[CLS]` token is read out and passed through a small MLP head:

$$
y = \text{MLP head}\big(\text{LN}(z_L^{0})\big)
$$

That is the whole model. No convolutions, no pooling pyramids — just patchify, then a standard Transformer encoder.

---

## Data hunger: why ViT needs pretraining

Because it lacks the convolutional priors, ViT only shines at **scale**. The original paper made this concrete:

<div style="background:#161b22;border-radius:8px;padding:1.2rem;margin:1.5rem 0;color:#c9d1d9;font-size:.92rem;">
<ul style="margin:0;padding-left:1.2rem;">
<li>Trained on <strong>ImageNet-1k</strong> only (~1.3M images), ViT <em>underperforms</em> a comparable ResNet.</li>
<li>Pre-trained on <strong>ImageNet-21k</strong> (~14M) it draws level.</li>
<li>Pre-trained on <strong>JFT-300M</strong> (~300M) it <em>surpasses</em> the best CNNs and transfers beautifully to downstream tasks.</li>
</ul>
</div>

This is precisely the **pretrain-then-finetune** recipe of the [Transfer Learning](../transfer-learning/index.md) class: pretrain the encoder on a huge dataset, then fine-tune the cheap MLP head (or the whole model with a low learning rate) on your task. It is also why CLIP could train a ViT image encoder on 400M image-text pairs — at that scale, the weak inductive bias becomes an advantage.

---

## CNN vs. ViT at a glance

| | CNN | Vision Transformer |
|---|---|---|
| Core operation | Convolution (local) | Self-attention (global) |
| Inductive bias | Strong (locality, equivariance) | Weak — learned from data |
| Receptive field | Grows with depth | Global from layer 1 |
| Data efficiency | Strong on small datasets | Needs large-scale pretraining |
| Compute | $O(N)$ in pixels | $O(N^2)$ in patches |
| Scales with data | Saturates earlier | Keeps improving |

**Hybrids and successors.** Several variants reintroduce *some* spatial bias to get the best of both worlds: **DeiT** (data-efficient training with distillation, no JFT needed), and **Swin Transformer** (windowed attention with a hierarchical, pyramid-like structure that brings back locality and makes ViT practical for detection and segmentation).

---

## Implementation reference

=== "PyTorch (patch embedding)"
    ```python
    import torch
    import torch.nn as nn

    class PatchEmbed(nn.Module):
        """Image -> sequence of patch tokens (Conv2d does patchify + projection)."""
        def __init__(self, img_size=224, patch=16, in_ch=3, dim=768):
            super().__init__()
            self.n_patches = (img_size // patch) ** 2
            self.proj = nn.Conv2d(in_ch, dim, kernel_size=patch, stride=patch)

        def forward(self, x):                  # x: (B, C, H, W)
            x = self.proj(x)                   # (B, dim, H/p, W/p)
            return x.flatten(2).transpose(1, 2)  # (B, N, dim)
    ```

=== "PyTorch (ViT forward)"
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
            x = torch.cat([cls, x], dim=1) + self.pos     # prepend CLS + add positions
            x = self.encoder(x)
            return self.head(x[:, 0])                      # classify from CLS token
    ```

=== "Use a pretrained ViT"
    ```python
    import timm
    # load ImageNet-21k -> 1k pretrained weights and fine-tune the head
    model = timm.create_model('vit_base_patch16_224', pretrained=True,
                              num_classes=10)
    ```

---

[^1]: Dosovitskiy, A. et al. (2021). [An Image is Worth 16×16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929){:target="_blank"}. ICLR.
[^2]: Touvron, H. et al. (2021). [Training data-efficient image transformers & distillation through attention (DeiT)](https://arxiv.org/abs/2012.12877){:target="_blank"}. ICML.
[^3]: Liu, Z. et al. (2021). [Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/abs/2103.14030){:target="_blank"}. ICCV.


---

--8<-- "docs/2026.2/classes/vision-transformers/quiz.md"
