## Autoregressive Image Generation

Diffusion models have dominated image generation since 2020 — but there is a radically different approach that has gained traction: **treating images as sequences of discrete tokens and generating them the same way language models generate text**.

This is the approach behind native image generation in **Gemini**, **GPT-4o**, and models like **Chameleon** (Meta) and **LlamaGen**.

---

## The Problem: How to Tokenize an Image?

Text is naturally discrete (words, subwords). Images are continuous — pixels in $[0,255]^3$. To use autoregressive generation, we need a **visual vocabulary**.

The solution: **VQ-GAN** (Vector Quantization GAN)[^1] learns a *codebook* of $K$ vectors. The encoder maps any image patch to the nearest vector in the codebook — converting the image into a grid of integer indices.

<div id="vqvae-viz" style="background:#0d1117;border-radius:12px;padding:1.5rem;margin:2rem 0;overflow:hidden;">
<canvas id="vqvae-canvas" style="width:100%;display:block;border-radius:8px;"></canvas>
<div style="display:flex;justify-content:center;gap:.8rem;margin-top:1rem;flex-wrap:wrap;">
  <button onclick="vqStep(-1)" style="padding:5px 18px;background:#21262d;color:#c9d1d9;border:1px solid #30363d;border-radius:5px;cursor:pointer;">← Previous</button>
  <span id="vq-label" style="color:#8b949e;font-family:monospace;line-height:2.2;font-size:.85rem;"></span>
  <button onclick="vqStep(1)" style="padding:5px 18px;background:#21262d;color:#c9d1d9;border:1px solid #30363d;border-radius:5px;cursor:pointer;">Next →</button>
</div>
<div id="vq-desc" style="color:#c9d1d9;font-size:.85rem;text-align:center;margin-top:.6rem;min-height:2.5rem;padding:0 1rem;'"></div>
</div>

<script>
(function(){
  const canvas = document.getElementById('vqvae-canvas');
  const ctx = canvas.getContext('2d');
  let vqS=0;

  const vqSteps=[
    {label:'1. Original Image', desc:'256×256px input image. Each pixel has a continuous RGB value.'},
    {label:'2. CNN Encoder', desc:'CNN contracts the image to a 32×32×256 latent representation. Each position captures an 8×8 pixel patch.'},
    {label:'3. Quantization: codebook matching', desc:'Each 256-dim latent vector is mapped to the nearest vector in the codebook (K=8192 entries). The index replaces the vector.'},
    {label:'4. Discrete token map', desc:'Result: 32×32 grid of integer indices in [0, 8191]. 1024 tokens represent the entire image.'},
    {label:'5. CNN Decoder', desc:'The decoder reconstructs the image from codebook vectors. Trained with reconstruction loss + GAN discriminator.'},
  ];

  window.vqStep=function(d){ vqS=Math.max(0,Math.min(vqSteps.length-1,vqS+d)); drawVQ(); };

  function rng(seed){ let x=Math.sin(seed*31.7)*43758.5; return x-Math.floor(x); }

  function drawVQ(){
    const W=canvas.parentElement.offsetWidth-48;
    const H=220; canvas.width=W; canvas.height=H; canvas.style.height=H+'px';
    ctx.fillStyle='#161b22'; ctx.fillRect(0,0,W,H);

    const s=vqSteps[vqS];
    document.getElementById('vq-label').textContent=`Step ${vqS+1}/${vqSteps.length}: ${s.label}`;
    document.getElementById('vq-desc').textContent=s.desc;

    const boxH=160, boxY=(H-boxH)/2;
    const components=[
      {label:'Image\n256×256', w:80},
      {label:'Encoder\nCNN', w:60},
      {label:'Latent\n32×32×256', w:80},
      {label:'Codebook\nK=8192', w:80},
      {label:'Tokens\n32×32', w:80},
      {label:'Decoder\nCNN', w:60},
      {label:'Reconstructed\nImage', w:80},
    ];
    const totalW=components.reduce((s,c)=>s+c.w+24,0)-24;
    let x=(W-totalW)/2;

    const colors=['#484f58','#58a6ff','#3fb950','#bc8cff','#f0883e','#58a6ff','#484f58'];

    components.forEach((comp,i)=>{
      const isActive = i<=vqS+1 && vqS>0 ? true : i===0;
      const col=colors[i];
      const bH=Math.min(boxH, comp.w*1.4);
      const by=boxY+(boxH-bH)/2;
      const alpha= (i===0||i<=vqS) ? 1 : 0.25;

      ctx.globalAlpha=alpha;
      ctx.fillStyle=col+'22';
      ctx.beginPath(); ctx.roundRect(x,by,comp.w,bH,6); ctx.fill();
      ctx.strokeStyle=col; ctx.lineWidth=1.5;
      ctx.beginPath(); ctx.roundRect(x,by,comp.w,bH,6); ctx.stroke();

      // Inner visual
      if(i===0||i===6){ // image
        const gs=8, gx=x+6, gy=by+6, gsz=(comp.w-12)/gs;
        for(let r=0;r<gs;r++) for(let c=0;c<gs;c++){
          const v=rng(r*8+c+(i===6?100:0));
          ctx.fillStyle=`hsl(${v*360|0},40%,${30+v*30|0}%)`;
          ctx.fillRect(gx+c*gsz,gy+r*gsz,gsz-1,gsz-1);
        }
      } else if(i===2){ // latent
        const gs=6, gx=x+8, gy=by+8, gsz=(comp.w-16)/gs;
        for(let r=0;r<gs;r++) for(let c=0;c<gs;c++){
          ctx.fillStyle=`hsl(${140+rng(r*6+c)*40|0},60%,${25+rng(r*6+c+50)*25|0}%)`;
          ctx.fillRect(gx+c*gsz,gy+r*gsz,gsz-1,gsz-1);
        }
      } else if(i===3){ // codebook
        for(let j=0;j<12;j++){
          const vy=by+10+j*12, vx=x+10;
          const v=rng(j*7);
          ctx.fillStyle=`hsl(${270+v*60|0},60%,${30+v*20|0}%)`;
          ctx.fillRect(vx,vy,comp.w-20,9);
          ctx.fillStyle='#ffffff33'; ctx.font='7px monospace'; ctx.textAlign='left'; ctx.textBaseline='middle';
          ctx.fillText((j*700+341)+'',vx+2,vy+4);
        }
      } else if(i===4){ // token map
        const gs=8, gx=x+4, gy=by+4, gsz=(comp.w-8)/gs;
        for(let r=0;r<gs;r++) for(let c=0;c<gs;c++){
          const idx=Math.floor(rng(r*8+c)*8192);
          const hue=(idx/8192)*360;
          ctx.fillStyle=`hsl(${hue|0},70%,35%)`;
          ctx.fillRect(gx+c*gsz,gy+r*gsz,gsz-1,gsz-1);
        }
      }

      // Label below
      ctx.globalAlpha=alpha;
      ctx.fillStyle=col; ctx.font='9px Inter,sans-serif'; ctx.textAlign='center'; ctx.textBaseline='top';
      comp.label.split('\n').forEach((l,li)=>ctx.fillText(l,x+comp.w/2,by+bH+4+li*11));

      // Arrow to next
      if(i<components.length-1){
        const nextX=x+comp.w+24;
        const alphaArr=i<vqS?1:0.2;
        ctx.globalAlpha=alphaArr;
        ctx.strokeStyle='#484f58'; ctx.lineWidth=1.5;
        ctx.beginPath(); ctx.moveTo(x+comp.w+2,boxY+boxH/2); ctx.lineTo(nextX-2,boxY+boxH/2); ctx.stroke();
        ctx.fillStyle='#484f58';
        ctx.beginPath(); ctx.moveTo(nextX-2,boxY+boxH/2); ctx.lineTo(nextX-8,boxY+boxH/2-4); ctx.lineTo(nextX-8,boxY+boxH/2+4); ctx.fill();
      }
      ctx.globalAlpha=1;
      x+=comp.w+24;
    });
  }

  drawVQ(); window.addEventListener('resize',drawVQ);
})();
</script>

---

## Autoregressive Token Generation

With a trained codebook, we can represent any image as a sequence of $N$ integer indices. We then generate this sequence **exactly like an LLM generates text**:

$$
p(t_1, t_2, \ldots, t_N) = \prod_{i=1}^{N} p(t_i \mid t_1, \ldots, t_{i-1}, \text{prompt})
$$

Each token is generated one at a time, conditioned on all previous ones and the text prompt.

<div id="ar-viz" style="background:#0d1117;border-radius:12px;padding:1.5rem;margin:2rem 0;">
<canvas id="ar-canvas" style="width:100%;display:block;border-radius:8px;"></canvas>
<div style="display:flex;gap:.8rem;justify-content:center;margin-top:1rem;flex-wrap:wrap;">
  <button id="ar-btn" onclick="arToggle()" style="padding:6px 22px;background:#3fb950;color:#0d1117;border:none;border-radius:5px;cursor:pointer;font-weight:bold;">&#9654; Generate</button>
  <button onclick="arReset()" style="padding:6px 16px;background:#21262d;color:#c9d1d9;border:1px solid #30363d;border-radius:5px;cursor:pointer;">&#8635; Reset</button>
  <label style="color:#8b949e;font-size:.85rem;line-height:2.2;">Order: </label>
  <select id="ar-order" onchange="arReset()" style="background:#161b22;color:#c9d1d9;border:1px solid #30363d;border-radius:4px;padding:3px 8px;">
    <option value="raster">Raster (←→)</option>
    <option value="masked">Masked (random)</option>
  </select>
</div>
<div id="ar-stats" style="color:#8b949e;font-size:.8rem;text-align:center;margin-top:.6rem;'"></div>
</div>

<script>
(function(){
  const canvas=document.getElementById('ar-canvas');
  const ctx=canvas.getContext('2d');
  let animId=null, running=false, generated=new Set(), totalTokens=0;
  let order=[];

  function rng(s){let x=Math.sin(s*31.7)*43758.5;return x-Math.floor(x);}
  const G=16; // grid size

  function buildOrder(){
    const ord=document.getElementById('ar-order').value;
    order=[];
    if(ord==='raster'){
      for(let r=0;r<G;r++) for(let c=0;c<G;c++) order.push([r,c]);
    } else {
      for(let r=0;r<G;r++) for(let c=0;c<G;c++) order.push([r,c]);
      order.sort(()=>Math.random()-0.5);
    }
  }

  function draw(){
    const W=canvas.parentElement.offsetWidth-48;
    const H=Math.min(300,W*0.5);
    canvas.width=W; canvas.height=H; canvas.style.height=H+'px';
    ctx.fillStyle='#161b22'; ctx.fillRect(0,0,W,H);
    totalTokens=G*G;

    // Grid
    const gridSz=Math.min(H-20,W*0.45-10);
    const cell=gridSz/G;
    const gx=20, gy=(H-gridSz)/2;

    // Draw generated tokens
    for(let r=0;r<G;r++) for(let c=0;c<G;c++){
      const key=r*G+c;
      if(generated.has(key)){
        const hue=rng(r*G+c+42)*360;
        ctx.fillStyle=`hsl(${hue|0},55%,32%)`;
      } else {
        ctx.fillStyle='#21262d';
      }
      ctx.fillRect(gx+c*cell+0.5,gy+r*cell+0.5,cell-1,cell-1);
    }

    // Highlight current generating position
    if(generated.size<order.length && running){
      const [r,c]=order[generated.size];
      ctx.fillStyle='#f0883e88';
      ctx.fillRect(gx+c*cell,gy+r*cell,cell,cell);
      ctx.strokeStyle='#f0883e'; ctx.lineWidth=1.5;
      ctx.strokeRect(gx+c*cell,gy+r*cell,cell,cell);
    }

    // Border
    ctx.strokeStyle='#30363d'; ctx.lineWidth=1;
    ctx.strokeRect(gx,gy,gridSz,gridSz);
    ctx.fillStyle='#484f58'; ctx.font='9px Inter,sans-serif'; ctx.textAlign='center';
    ctx.fillText(`${G}×${G} = ${G*G} tokens`,gx+gridSz/2,gy+gridSz+14);

    // Right panel: probability bars
    const pw=Math.min(200,W-gx-gridSz-60);
    if(pw>80){
      const px=gx+gridSz+30, py=(H-120)/2;
      ctx.fillStyle='#8b949e'; ctx.font='9px Inter,sans-serif'; ctx.textAlign='left';
      ctx.fillText('p(next token|context)',px,py-8);

      const topK=5;
      for(let i=0;i<topK;i++){
        const seed=generated.size*7+i;
        const prob=[0.42,0.23,0.15,0.11,0.09][i]*(0.85+rng(seed)*0.3);
        const idx=Math.floor(rng(seed*3)*8192);
        const bw=Math.round((prob/0.5)*pw);
        const by=py+i*22;
        ctx.fillStyle='#21262d';
        ctx.beginPath(); ctx.roundRect(px,by,pw,16,3); ctx.fill();
        ctx.fillStyle=i===0?'#3fb950':'#58a6ff88';
        ctx.beginPath(); ctx.roundRect(px,by,Math.min(bw,pw),16,3); ctx.fill();
        ctx.fillStyle='#c9d1d9'; ctx.font='8px monospace'; ctx.textAlign='left';
        ctx.fillText(`tok_${idx}: ${(prob*100).toFixed(1)}%`,px+4,by+11);
      }
    }

    // Stats
    document.getElementById('ar-stats').textContent=
      `Generated: ${generated.size}/${G*G} tokens  (${((generated.size/(G*G))*100).toFixed(0)}%)`;
  }

  window.arToggle=function(){
    running=!running;
    document.getElementById('ar-btn').textContent=running?'⏸ Pause':'▶ Generate';
    document.getElementById('ar-btn').style.background=running?'#d29922':'#3fb950';
    if(running) animate();
    else cancelAnimationFrame(animId);
  };

  window.arReset=function(){
    running=false; cancelAnimationFrame(animId);
    document.getElementById('ar-btn').textContent='▶ Generate';
    document.getElementById('ar-btn').style.background='#3fb950';
    generated=new Set(); buildOrder(); draw();
  };

  function animate(){
    if(generated.size<order.length&&running){
      const [r,c]=order[generated.size];
      generated.add(r*G+c);
      draw();
      const delay=generated.size<10?80:generated.size<50?30:12;
      animId=setTimeout(animate,delay);
    } else {
      running=false;
      document.getElementById('ar-btn').textContent='▶ Generate';
      document.getElementById('ar-btn').style.background='#3fb950';
      draw();
    }
  }

  buildOrder(); draw();
  window.addEventListener('resize',draw);
})();
</script>

---

## MaskGIT: Parallel Generation via Masking

Purely autoregressive generation is **slow**: 1024 tokens = 1024 model passes. **MaskGIT**[^2] accelerates this with iterative parallel generation:

1. Start with **all tokens masked** `[MASK]`
2. At each iteration, predict **all tokens simultaneously** (bidirectional!)
3. "Reveal" only the tokens with highest confidence
4. Repeat with fewer masked tokens

In just **8–12 iterations**, it generates 1024 tokens — versus 1024 iterations for pure AR.

<div id="mask-viz" style="background:#0d1117;border-radius:12px;padding:1.5rem;margin:2rem 0;">
<canvas id="mask-canvas" style="width:100%;display:block;border-radius:8px;"></canvas>
<div style="display:flex;gap:.8rem;justify-content:center;margin-top:1rem;flex-wrap:wrap;">
  <button onclick="maskStep(-1)" style="padding:5px 18px;background:#21262d;color:#c9d1d9;border:1px solid #30363d;border-radius:5px;cursor:pointer;">← Previous</button>
  <span id="mask-label" style="color:#f0883e;font-family:monospace;font-size:.85rem;line-height:2;"></span>
  <button onclick="maskStep(1)" style="padding:5px 18px;background:#21262d;color:#c9d1d9;border:1px solid #30363d;border-radius:5px;cursor:pointer;">Next →</button>
</div>
</div>

<script>
(function(){
  const canvas=document.getElementById('mask-canvas');
  const ctx=canvas.getContext('2d');
  let maskS=0;
  const G=10, K=8;
  function rng(s){let x=Math.sin(s*31.7)*43758.5;return x-Math.floor(x);}

  // Pre-generate reveal schedule (MaskGIT cosine schedule)
  const totalToks=G*G;
  const schedule=[1,0.82,0.63,0.46,0.31,0.19,0.09,0.02,0];
  const tokenColors=Array.from({length:totalToks},(_,i)=>`hsl(${(rng(i*7)*320)|0},55%,38%)`);

  window.maskStep=function(d){maskS=Math.max(0,Math.min(K-1,maskS+d));drawMask();};

  function drawMask(){
    const W=canvas.parentElement.offsetWidth-48;
    const H=200; canvas.width=W; canvas.height=H; canvas.style.height=H+'px';
    ctx.fillStyle='#161b22'; ctx.fillRect(0,0,W,H);

    const frac=schedule[maskS];
    const revealed=Math.round(totalToks*(1-frac));
    document.getElementById('mask-label').textContent=
      `Iteration ${maskS+1}/${K}: ${revealed}/${totalToks} tokens revealed (${((1-frac)*100).toFixed(0)}%)`;

    // Multiple grid snapshots side by side
    const steps=Math.min(K,Math.floor((W-20)/60));
    const gSz=Math.min(50,(W-20)/steps-8);
    const cell=gSz/G;
    const startX=(W-(steps*(gSz+8)))/2;

    for(let s=0;s<steps;s++){
      const f=schedule[s];
      const rev=Math.round(totalToks*(1-f));
      const gx=startX+s*(gSz+8), gy=(H-gSz)/2;
      const isActive=s===maskS;

      // Highlight active
      if(isActive){
        ctx.fillStyle='#f0883e22';
        ctx.beginPath(); ctx.roundRect(gx-3,gy-3,gSz+6,gSz+6,5); ctx.fill();
        ctx.strokeStyle='#f0883e'; ctx.lineWidth=1.5;
        ctx.beginPath(); ctx.roundRect(gx-3,gy-3,gSz+6,gSz+6,5); ctx.stroke();
      }

      for(let r=0;r<G;r++) for(let c=0;c<G;c++){
        const idx=r*G+c;
        const isRev=idx<rev;
        ctx.fillStyle=isRev?tokenColors[idx]:'#1c2128';
        ctx.fillRect(gx+c*cell+0.3,gy+r*cell+0.3,cell-0.6,cell-0.6);
        if(!isRev&&cell>5){
          ctx.fillStyle='#30363d';
          ctx.fillRect(gx+c*cell+0.3,gy+r*cell+0.3,cell-0.6,cell-0.6);
        }
      }

      ctx.fillStyle=isActive?'#f0883e':'#484f58';
      ctx.font=`${isActive?'bold ':''}8px monospace`;
      ctx.textAlign='center'; ctx.textBaseline='top';
      ctx.fillText(`t=${s+1}`,gx+gSz/2,gy+gSz+4);
      ctx.fillText(`${rev}tok`,gx+gSz/2,gy+gSz+14);
    }
  }

  drawMask(); window.addEventListener('resize',drawMask);
})();
</script>

---

## Any-to-Any: Gemini, GPT-4o, and Chameleon

The final step is removing the distinction between text and image tokens. **Any-to-any** models treat everything as a token sequence:

```
[TEXT: "a photo of"] [IMG_TOK_3742] [IMG_TOK_891] ... [IMG_TOK_5531] [TEXT: "cat"]
```

The standard Transformer model processes this mixed sequence naturally.

<div style="background:#161b22;border-radius:8px;padding:1.2rem;margin:1.5rem 0;overflow-x:auto;">
<canvas id="anyto-canvas" style="width:100%;display:block;"></canvas>
</div>

<script>
(function(){
  const canvas=document.getElementById('anyto-canvas');
  const ctx=canvas.getContext('2d');

  function draw(){
    const W=canvas.parentElement.offsetWidth-32;
    const H=120; canvas.width=W; canvas.height=H; canvas.style.height=H+'px';
    ctx.fillStyle='#161b22'; ctx.fillRect(0,0,W,H);

    const tokens=[
      {label:'Describe',type:'text'},
      {label:'the image:',type:'text'},
      {label:'🟦🟦\n🟧🟦',type:'img'},
      {label:'img_4821',type:'img'},
      {label:'img_0331',type:'img'},
      {label:'img_7102',type:'img'},
      {label:'→ Answer:',type:'text'},
      {label:'"blue sky\nand sunset"',type:'text'},
    ];

    let x=10;
    tokens.forEach((t,i)=>{
      const isImg=t.type==='img';
      const col=isImg?'#3fb950':'#58a6ff';
      const lines=t.label.split('\n');
      const tw=Math.max(60, lines.reduce((m,l)=>Math.max(m,l.length*7),0)+16);
      const tH=isImg?56:40;
      const ty=(H-tH)/2;

      ctx.fillStyle=col+'22';
      ctx.beginPath(); ctx.roundRect(x,ty,tw,tH,6); ctx.fill();
      ctx.strokeStyle=col; ctx.lineWidth=1.5;
      ctx.beginPath(); ctx.roundRect(x,ty,tw,tH,6); ctx.stroke();

      ctx.fillStyle=col; ctx.font=`bold ${Math.min(10,tw/8)}px monospace`;
      ctx.textAlign='center'; ctx.textBaseline='middle';
      lines.forEach((l,li)=>ctx.fillText(l,x+tw/2,ty+tH/2+(li-(lines.length-1)/2)*13));

      ctx.fillStyle=col+'99'; ctx.font='7px Inter,sans-serif';
      ctx.fillText(isImg?'img_tok':'txt_tok',x+tw/2,ty+tH+10);

      if(i<tokens.length-1){
        ctx.strokeStyle='#30363d'; ctx.lineWidth=1;
        ctx.beginPath(); ctx.moveTo(x+tw+2,H/2); ctx.lineTo(x+tw+8,H/2); ctx.stroke();
      }
      x+=tw+10;
      if(x>W-20) return;
    });

    ctx.fillStyle='#58a6ff'; ctx.font='9px Inter,sans-serif'; ctx.textAlign='left';
    ctx.fillText('● Text',10,H-8);
    ctx.fillStyle='#3fb950'; ctx.fillText('● Image (VQ tokens)',60,H-8);
  }

  draw(); window.addEventListener('resize',draw);
})();
</script>

### How Each Model Implements This

| Model | Visual tokenizer | Generation | Training |
|--------|------------------|---------|------------|
| **Chameleon** (Meta) | VQ-VAE (8192 codes) | Pure autoregressive | Text + image together from the start |
| **Gemini 2.0** (Google) | Proprietary tokenizer | AR + diffusion decoder | Native multimodal |
| **GPT-4o** (OpenAI) | Discrete visual tokens | AR + diffusion decoder | Native multimodal |
| **LlamaGen** | VQGAN (16384 codes) | AR with LLaMA | Initializes from pre-trained LLaMA |

---

## AR vs. Diffusion: When to Use Each?

<div style="display:grid;grid-template-columns:1fr 1fr;gap:1rem;margin:1.5rem 0;">

<div style="background:#0d1117;border-left:3px solid #3fb950;padding:1rem;border-radius:8px;">
<strong style="color:#3fb950;">Autoregressive Generation</strong>
<ul style="color:#c9d1d9;font-size:.85rem;margin:.5rem 0;padding-left:1.2rem;">
<li>Unifies text and image in the same architecture</li>
<li>Best for multimodal any-to-any</li>
<li>Leverages the entire LLM infrastructure</li>
<li>Scales well with more data</li>
<li><strong style="color:#ff7b72;">Slow</strong>: 1 token at a time</li>
</ul>
</div>

<div style="background:#0d1117;border-left:3px solid #58a6ff;padding:1rem;border-radius:8px;">
<strong style="color:#58a6ff;">Diffusion (DDPM / Flow Matching)</strong>
<ul style="color:#c9d1d9;font-size:.85rem;margin:.5rem 0;padding-left:1.2rem;">
<li>Best standalone image quality</li>
<li>Coherent global generation</li>
<li>More control (guidance, cfg scale)</li>
<li>Faster per image than AR</li>
<li><strong style="color:#ff7b72;">Does not natively unify with text</strong></li>
</ul>
</div>

</div>

The current trend: **hybrids** — an autoregressive LLM backbone for understanding and reasoning, with a diffusion decoder to render the final image at high quality. This is exactly what GPT-4o does.

---

## Implementation: VQ-GAN + Autoregressive Transformer

```python
import torch
import torch.nn as nn

# 1. Vector quantizer
class VectorQuantizer(nn.Module):
    def __init__(self, n_codes, d_code):
        super().__init__()
        self.codebook = nn.Embedding(n_codes, d_code)

    def forward(self, z):
        # z: (B, H, W, d_code) — encoder latents
        flat = z.view(-1, z.shape[-1])
        # Distances to codebook
        dists = torch.cdist(flat, self.codebook.weight)
        indices = dists.argmin(dim=-1)            # index of nearest code
        quantized = self.codebook(indices).view_as(z)
        # Straight-through estimator for backprop
        quantized_st = z + (quantized - z).detach()
        return quantized_st, indices.view(z.shape[:3])

# 2. Autoregressive generation with GPT-like model
class ImageGPT(nn.Module):
    def __init__(self, n_codes, seq_len, d_model, n_heads, n_layers):
        super().__init__()
        self.tok_emb = nn.Embedding(n_codes + 1, d_model)  # +1 for BOS token
        self.pos_emb = nn.Embedding(seq_len + 1, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, n_heads, d_model*4, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, n_layers)
        self.head = nn.Linear(d_model, n_codes)

    def forward(self, tokens):
        B, T = tokens.shape
        pos = torch.arange(T, device=tokens.device).unsqueeze(0)
        x = self.tok_emb(tokens) + self.pos_emb(pos)
        # Causal mask
        mask = torch.triu(torch.ones(T, T, device=tokens.device), diagonal=1).bool()
        x = self.transformer(x, mask=mask)
        return self.head(x)  # logits over n_codes

    @torch.no_grad()
    def generate(self, prompt_tokens, n_new, temperature=1.0, top_k=2048):
        tokens = prompt_tokens.clone()
        for _ in range(n_new):
            logits = self(tokens)[:, -1, :] / temperature
            if top_k: logits[logits < logits.topk(top_k)[0][:,-1:]] = -float('inf')
            probs = logits.softmax(-1)
            next_tok = torch.multinomial(probs, 1)
            tokens = torch.cat([tokens, next_tok], dim=1)
        return tokens
```

---

[^1]: Esser, P. et al. (2021). [Taming Transformers for High-Resolution Image Synthesis (VQ-GAN)](https://arxiv.org/abs/2012.09841){:target="_blank"}.
[^2]: Chang, H. et al. (2022). [MaskGIT: Masked Generative Image Transformer](https://arxiv.org/abs/2202.04200){:target="_blank"}.
[^3]: Yu, L. et al. (2023). [LlamaGen: Autoregressive Image Generation without Vector Quantization](https://arxiv.org/abs/2406.06525){:target="_blank"}.
[^4]: Team, C. et al. (2024). [Chameleon: Mixed-Modal Early-Fusion Foundation Models](https://arxiv.org/abs/2405.09818){:target="_blank"}.
[^5]: Li, J. et al. (2024). [MAR: Autoregressive Image Generation without Vector Quantization](https://arxiv.org/abs/2406.11838){:target="_blank"}.


---

--8<-- "docs/2026.2/classes/autoregressive-generation/quiz.md"
