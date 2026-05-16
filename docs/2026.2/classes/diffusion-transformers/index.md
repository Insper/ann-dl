## Diffusion Transformers (DiT)

In 2023, Peebles & Xie[^1] demonstrated something simple and impactful: **the U-Net is not necessary for diffusion models**. By replacing it with pure Transformer blocks, the model not only maintained quality — it started to **scale predictably** with more parameters and data, exactly like language models.

Today, all state-of-the-art image and video generation uses DiT:

| Model | Architecture | Objective |
|--------|------------|---------|
| **FLUX.1** | DiT (dual-stream) | Flow Matching |
| **Stable Diffusion 3** | MMDiT | Flow Matching |
| **Sora** (OpenAI) | Spacetime DiT | Diffusion |
| **Movie Gen** (Meta) | DiT | Flow Matching |
| **CogVideoX** | DiT 3D | Flow Matching |

---

## From U-Net to Transformer

The classic U-Net uses convolutions with hierarchical skip connections — good for capturing local details, but difficult to scale. DiT replaces all of this with global attention blocks.

<div id="arch-compare" style="background:#0d1117;border-radius:12px;padding:1.5rem;margin:2rem 0;">
<canvas id="arch-canvas" style="width:100%;display:block;"></canvas>
<div style="display:flex;justify-content:center;gap:1rem;margin-top:.8rem;flex-wrap:wrap;">
  <button onclick="showArch('unet')" id="btn-unet" style="padding:5px 20px;background:#58a6ff;color:#0d1117;border:none;border-radius:5px;cursor:pointer;font-weight:bold;">U-Net</button>
  <button onclick="showArch('dit')" id="btn-dit" style="padding:5px 20px;background:#21262d;color:#c9d1d9;border:1px solid #30363d;border-radius:5px;cursor:pointer;">DiT</button>
  <button onclick="showArch('mmdit')" id="btn-mmdit" style="padding:5px 20px;background:#21262d;color:#c9d1d9;border:1px solid #30363d;border-radius:5px;cursor:pointer;">MMDiT (FLUX)</button>
</div>
<div id="arch-desc" style="color:#c9d1d9;font-size:.85rem;text-align:center;margin-top:.8rem;min-height:2rem;padding:0 1rem;"></div>
</div>

<script>
(function(){
  const canvas = document.getElementById('arch-canvas');
  const ctx = canvas.getContext('2d');
  let currentArch = 'unet';

  window.showArch = function(arch) {
    currentArch = arch;
    ['unet','dit','mmdit'].forEach(a => {
      const btn = document.getElementById('btn-'+a);
      btn.style.background = a===arch ? '#58a6ff' : '#21262d';
      btn.style.color = a===arch ? '#0d1117' : '#c9d1d9';
      btn.style.border = a===arch ? 'none' : '1px solid #30363d';
    });
    draw();
  };

  const descs = {
    unet: 'U-Net: encoder contracts (convolutions + pooling), decoder expands (upsample + conv). Skip connections transfer features between levels.',
    dit: 'DiT: latent divided into patches → token sequence → N Transformer blocks → depatchify. Simple and scalable.',
    mmdit: 'MMDiT (FLUX/SD3): text and image tokens flow in separate streams but communicate via shared bidirectional attention.'
  };

  function box(x,y,w,h,color,label,sub,alpha=1) {
    ctx.globalAlpha = alpha;
    ctx.fillStyle = color+'33';
    ctx.beginPath(); ctx.roundRect(x,y,w,h,5); ctx.fill();
    ctx.strokeStyle = color; ctx.lineWidth=1.5;
    ctx.beginPath(); ctx.roundRect(x,y,w,h,5); ctx.stroke();
    ctx.globalAlpha = 1;
    ctx.fillStyle = color; ctx.font='bold 11px Inter,sans-serif'; ctx.textAlign='center'; ctx.textBaseline='middle';
    ctx.fillText(label, x+w/2, y+h/2 - (sub?5:0));
    if(sub){ctx.font='9px monospace'; ctx.fillStyle=color+'aa'; ctx.fillText(sub,x+w/2,y+h/2+8);}
  }

  function arrow(x1,y1,x2,y2,color='#30363d'){
    ctx.strokeStyle=color; ctx.lineWidth=1.5;
    ctx.beginPath(); ctx.moveTo(x1,y1); ctx.lineTo(x2,y2); ctx.stroke();
    const angle=Math.atan2(y2-y1,x2-x1);
    ctx.fillStyle=color;
    ctx.beginPath(); ctx.moveTo(x2,y2);
    ctx.lineTo(x2-8*Math.cos(angle-0.4),y2-8*Math.sin(angle-0.4));
    ctx.lineTo(x2-8*Math.cos(angle+0.4),y2-8*Math.sin(angle+0.4));
    ctx.fill();
  }

  function drawUNet(W,H){
    const cx=W/2, cols=['#58a6ff','#3fb950','#f0883e','#bc8cff','#d29922'];
    const levels=5, bW=70, bH=28, gap=12;
    const totalW = levels*2*bW + (levels*2-1)*gap;
    const startX = cx - totalW/2;
    const midY = H/2;

    // Encoder (contracting)
    for(let i=0;i<levels;i++){
      const x=startX+i*(bW+gap);
      const y=midY - (i+1)*18;
      const h=bH+(levels-i)*6;
      box(x,y,bW,h,cols[i],'Conv '+i,'enc');
      if(i>0) arrow(x-gap+2,y+h/2,x,y+h/2,'#ffffff33');
      // skip lines
      if(i<levels-1){
        const rx=startX+(levels*2-2-i)*(bW+gap);
        ctx.strokeStyle='#ffffff22'; ctx.lineWidth=1; ctx.setLineDash([3,3]);
        ctx.beginPath(); ctx.moveTo(x+bW,y+h/2); ctx.lineTo(rx,y+h/2); ctx.stroke();
        ctx.setLineDash([]);
      }
    }
    // Bottleneck
    const btx=startX+levels*(bW+gap)-gap/2; const bty=midY-levels*18-10;
    box(btx-bW/2,bty,bW*1.1,bH+levels*6+20,'#ff7b72','Bottleneck','attn+conv');
    // Decoder (expanding)
    for(let i=levels-2;i>=0;i--){
      const j=levels-2-i;
      const x=startX+(levels+j+1)*(bW+gap);
      const y=midY-(i+1)*18;
      const h=bH+(levels-i)*6;
      box(x,y,bW,h,cols[i],'Conv '+i,'dec');
      arrow(x-gap+2,y+h/2,x,y+h/2,'#ffffff33');
    }
    // Output
    ctx.fillStyle='#8b949e'; ctx.font='10px Inter,sans-serif'; ctx.textAlign='center';
    ctx.fillText('Input Noisy Latent',startX+bW/2,midY+60);
    ctx.fillText('Output Clean Latent',startX+totalW-bW/2,midY+60);
    ctx.fillStyle='#30363d'; ctx.font='10px monospace'; ctx.textAlign='center';
    ctx.fillText('← Skip Connections →',cx,midY+40);
  }

  function drawDiT(W,H){
    const cx=W/2;
    const steps=[
      {label:'Noisy Latent',sub:'z_t ∈ R^{h×w×c}',color:'#484f58'},
      {label:'Patchify',sub:'→ N×d_model patches',color:'#58a6ff'},
      {label:'+ Pos. Emb.',sub:'2D sinusoidal',color:'#58a6ff'},
      {label:'DiT Block ×N',sub:'AdaLN + Attn + FFN',color:'#3fb950'},
      {label:'Depatchify',sub:'N×d → h×w×c',color:'#58a6ff'},
      {label:'Predicted ε / v',sub:'loss target',color:'#f0883e'},
    ];
    const bH=36, gap=8, bW=180;
    const totalH=steps.length*(bH+gap)-gap;
    let y=(H-totalH)/2;
    const x=cx-bW/2;

    // Conditioning input
    box(cx+100,y+3*(bH+gap),90,bH,'#bc8cff','Timestep t','+class/text');
    arrow(cx+100,y+3*(bH+gap)+bH/2,x+bW,y+3*(bH+gap)+bH/2,'#bc8cff');

    steps.forEach((s,i)=>{
      box(x,y,bW,bH,s.color,s.label,s.sub);
      if(i<steps.length-1) arrow(cx,y+bH+1,cx,y+bH+gap,'#30363d');
      y+=bH+gap;
    });
  }

  function drawMMDiT(W,H){
    const cx=W/2;
    const imgColor='#3fb950', txtColor='#58a6ff', sharedColor='#bc8cff';

    // Two streams
    const streamW=140, streamH=38, gap=10;
    const imgX=cx-streamW-30, txtX=cx+30;
    const blocks=['Input','Linear Proj.','+ Pos. Emb.','MM-Attn Block ×N','Linear Out'];
    const blockColors=[['#21262d','#21262d'],['#3fb95055','#58a6ff55'],['#3fb95055','#58a6ff55'],[sharedColor+'55',sharedColor+'55'],['#3fb95055','#58a6ff55']];
    const blockBorder=[[imgColor,txtColor],[imgColor,txtColor],[imgColor,txtColor],[sharedColor,sharedColor],[imgColor,txtColor]];

    let y=20;
    // Stream labels
    ctx.fillStyle=imgColor; ctx.font='bold 12px Inter,sans-serif'; ctx.textAlign='center';
    ctx.fillText('Image Tokens', imgX+streamW/2, y+14);
    ctx.fillStyle=txtColor; ctx.fillText('Text Tokens', txtX+streamW/2, y+14);
    y+=32;

    blocks.forEach((label,i)=>{
      box(imgX,y,streamW,streamH,blockBorder[i][0],label,'',1);
      box(txtX,y,streamW,streamH,blockBorder[i][1],label,'',1);
      if(i===3){
        // Bidirectional attention arc
        const midY=y+streamH/2;
        ctx.strokeStyle=sharedColor+'88'; ctx.lineWidth=2; ctx.setLineDash([4,3]);
        ctx.beginPath(); ctx.moveTo(imgX+streamW,midY); ctx.lineTo(txtX,midY); ctx.stroke();
        ctx.setLineDash([]);
        ctx.fillStyle=sharedColor; ctx.font='bold 9px Inter,sans-serif'; ctx.textAlign='center';
        ctx.fillText('⟺ Shared Attention', cx, midY-6);
        ctx.font='8px monospace'; ctx.fillStyle=sharedColor+'aa';
        ctx.fillText('img sees txt & txt sees img', cx, midY+8);
      }
      if(i<blocks.length-1){
        arrow(imgX+streamW/2,y+streamH+1,imgX+streamW/2,y+streamH+gap,imgColor);
        arrow(txtX+streamW/2,y+streamH+1,txtX+streamW/2,y+streamH+gap,txtColor);
      }
      y+=streamH+gap;
    });

    ctx.fillStyle='#8b949e'; ctx.font='10px Inter,sans-serif'; ctx.textAlign='center';
    ctx.fillText('Noisy Latent Patches', imgX+streamW/2, y+12);
    ctx.fillText('Text Embeddings (T5)', txtX+streamW/2, y+12);
  }

  function draw(){
    const W=canvas.parentElement.offsetWidth-48;
    const H=340; canvas.width=W; canvas.height=H; canvas.style.height=H+'px';
    ctx.fillStyle='#0d1117'; ctx.fillRect(0,0,W,H);
    document.getElementById('arch-desc').textContent=descs[currentArch];
    if(currentArch==='unet') drawUNet(W,H);
    else if(currentArch==='dit') drawDiT(W,H);
    else drawMMDiT(W,H);
  }

  draw(); window.addEventListener('resize',draw);
})();
</script>

---

## Step 1 — Patchify: Images as Token Sequences

Just like ViT divides images into patches, DiT operates in the **latent space** (after the VAE encoder). A latent of shape $H \times W \times C$ is divided into patches of size $p \times p$:

$$
\text{Number of tokens: } N = \frac{H}{p} \times \frac{W}{p}
$$

Each patch is flattened and projected to dimension $d_{\text{model}}$ — becoming a "visual token".

<div id="patch-viz" style="background:#0d1117;border-radius:12px;padding:1.5rem;margin:2rem 0;">
<div style="display:flex;gap:1rem;justify-content:center;margin-bottom:1rem;flex-wrap:wrap;">
  <label style="color:#8b949e;font-size:.85rem;line-height:2.2;">Latent: </label>
  <select id="latent-size" onchange="drawPatches()" style="background:#161b22;color:#c9d1d9;border:1px solid #30363d;border-radius:4px;padding:3px 8px;">
    <option value="8">8×8 (tiny)</option>
    <option value="16" selected>16×16 (SD normal)</option>
    <option value="32">32×32 (HD)</option>
  </select>
  <label style="color:#8b949e;font-size:.85rem;line-height:2.2;">Patch size: </label>
  <select id="patch-size" onchange="drawPatches()" style="background:#161b22;color:#c9d1d9;border:1px solid #30363d;border-radius:4px;padding:3px 8px;">
    <option value="1">1×1</option>
    <option value="2" selected>2×2</option>
    <option value="4">4×4</option>
  </select>
</div>
<canvas id="patch-canvas" style="width:100%;display:block;border-radius:8px;"></canvas>
<div id="patch-stats" style="color:#8b949e;font-size:.85rem;text-align:center;margin-top:.8rem;"></div>
</div>

<script>
window.drawPatches = function() {
  const canvas = document.getElementById('patch-canvas');
  const ctx = canvas.getContext('2d');
  const L = +document.getElementById('latent-size').value;
  const P = +document.getElementById('patch-size').value;

  const W = canvas.parentElement.offsetWidth - 48;
  const H = 280; canvas.width=W; canvas.height=H; canvas.style.height=H+'px';
  ctx.fillStyle='#161b22'; ctx.fillRect(0,0,W,H);

  const N = (L/P)*(L/P);
  const gridSz = Math.min(220, W*0.45);
  const cell = gridSz / L;
  const gx = 24, gy = (H-gridSz)/2;

  // Color map (simulate latent features)
  const rng = (seed,i)=>{ let x=Math.sin(seed*31+i*17)*43758.5453; return x-Math.floor(x); };

  // Draw latent grid
  for(let r=0;r<L;r++) for(let c=0;c<L;c++){
    const v = rng(r*100+c, 7);
    ctx.fillStyle = `hsl(${200+v*80},${40+v*40}%,${20+v*30}%)`;
    ctx.fillRect(gx+c*cell, gy+r*cell, cell-0.5, cell-0.5);
  }

  // Draw patch borders
  ctx.strokeStyle='#f0883e'; ctx.lineWidth=1.5;
  for(let r=0;r<=L;r+=P){
    ctx.beginPath(); ctx.moveTo(gx,gy+r*cell); ctx.lineTo(gx+gridSz,gy+r*cell); ctx.stroke();
  }
  for(let c=0;c<=L;c+=P){
    ctx.beginPath(); ctx.moveTo(gx+c*cell,gy); ctx.lineTo(gx+c*cell,gy+gridSz); ctx.stroke();
  }

  // Label
  ctx.fillStyle='#8b949e'; ctx.font='10px Inter,sans-serif'; ctx.textAlign='center';
  ctx.fillText(`Latent ${L}×${L}`, gx+gridSz/2, gy-8);
  ctx.fillStyle='#f0883e'; ctx.fillText(`Patches ${P}×${P}`, gx+gridSz/2, gy+gridSz+14);

  // Arrow
  const arrowX = gx+gridSz+20;
  ctx.strokeStyle='#484f58'; ctx.lineWidth=2;
  ctx.beginPath(); ctx.moveTo(arrowX,H/2); ctx.lineTo(arrowX+30,H/2); ctx.stroke();
  ctx.fillStyle='#484f58';
  ctx.beginPath(); ctx.moveTo(arrowX+30,H/2); ctx.lineTo(arrowX+22,H/2-5); ctx.lineTo(arrowX+22,H/2+5); ctx.fill();
  ctx.fillStyle='#484f58'; ctx.font='9px monospace'; ctx.textAlign='center';
  ctx.fillText('flatten', arrowX+15, H/2-8);
  ctx.fillText('+proj', arrowX+15, H/2+16);

  // Token sequence
  const tokW = Math.min(28, (W - arrowX - 60) / Math.min(N, 16) - 3);
  const tokH = 22;
  const seqX = arrowX + 45;
  const maxVis = Math.min(N, Math.floor((W - seqX - 10) / (tokW+3)));
  const seqY = H/2 - tokH/2;

  for(let i=0;i<maxVis;i++){
    const v=rng(i,3);
    ctx.fillStyle=`hsl(${200+v*80},50%,35%)`;
    ctx.beginPath(); ctx.roundRect(seqX+i*(tokW+3),seqY,tokW,tokH,4); ctx.fill();
    ctx.strokeStyle='#3fb950'; ctx.lineWidth=1;
    ctx.beginPath(); ctx.roundRect(seqX+i*(tokW+3),seqY,tokW,tokH,4); ctx.stroke();
    if(tokW>=16){ ctx.fillStyle='#c9d1d9'; ctx.font='8px monospace'; ctx.textAlign='center';
      ctx.fillText('t'+(i+1),seqX+i*(tokW+3)+tokW/2,seqY+tokH/2+3); }
  }
  if(maxVis<N){
    ctx.fillStyle='#484f58'; ctx.font='12px monospace'; ctx.textAlign='left';
    ctx.fillText('...', seqX+maxVis*(tokW+3)+4, seqY+tokH/2+4);
  }

  ctx.fillStyle='#3fb950'; ctx.font='10px Inter,sans-serif'; ctx.textAlign='center';
  const seqCx=seqX+(Math.min(maxVis,N))*(tokW+3)/2;
  ctx.fillText(`${N} tokens`, seqCx, seqY+tokH+14);

  document.getElementById('patch-stats').textContent =
    `Latent ${L}×${L}  |  Patch ${P}×${P}  |  Tokens: ${N}  |  d_model: 1152 (FLUX) or 1536 (SD3)`;
};
drawPatches();
</script>

---

## Step 2 — DiT Block with AdaLN

DiT uses **Adaptive Layer Normalization** (AdaLN) to inject *timestep* and *class/text* information directly into the normalization parameters:

$$
\text{AdaLN}(h, c) = \gamma(c) \cdot \frac{h - \mu}{\sigma} + \beta(c)
$$

where $c = \text{MLP}(\text{emb}(t) + \text{emb}(\text{class}))$ is the conditioning vector.

The parameters $\gamma$ and $\beta$ are **predicted** — not statically learned — making normalization sensitive to the diffusion step and the prompt.

<div id="block-viz" style="background:#0d1117;border-radius:12px;padding:1.5rem;margin:2rem 0;">
<canvas id="block-canvas" style="width:100%;display:block;"></canvas>
</div>

<script>
(function(){
  const canvas = document.getElementById('block-canvas');
  const ctx = canvas.getContext('2d');

  function draw(){
    const W=canvas.parentElement.offsetWidth-48;
    const H=300; canvas.width=W; canvas.height=H; canvas.style.height=H+'px';
    ctx.fillStyle='#0d1117'; ctx.fillRect(0,0,W,H);

    const bW=150, bH=30, cx=W/2, gap=8;
    let y=20;

    function blk(label,color,sub){
      ctx.fillStyle=color+'33'; ctx.beginPath(); ctx.roundRect(cx-bW/2,y,bW,bH,5); ctx.fill();
      ctx.strokeStyle=color; ctx.lineWidth=1.5; ctx.beginPath(); ctx.roundRect(cx-bW/2,y,bW,bH,5); ctx.stroke();
      ctx.fillStyle=color; ctx.font='bold 11px Inter,sans-serif'; ctx.textAlign='center'; ctx.textBaseline='middle';
      ctx.fillText(label,cx,y+bH/2-(sub?5:0));
      if(sub){ctx.font='8px monospace'; ctx.fillStyle=color+'99'; ctx.fillText(sub,cx,y+bH/2+8);}
      const yy=y; y+=bH+gap;
      return yy;
    }
    function arr(){
      ctx.strokeStyle='#30363d'; ctx.lineWidth=1.5;
      ctx.beginPath(); ctx.moveTo(cx,y-gap); ctx.lineTo(cx,y); ctx.stroke();
    }

    // Input
    blk('Image Tokens x','#484f58','(N, d_model)');
    // Residual start
    const resStart=y-gap/2;
    arr();
    const y1=blk('AdaLN (scale+shift)','#bc8cff','γ(c)·norm(x)+β(c)');
    arr();
    blk('Multi-Head Self-Attention','#58a6ff','Q,K,V ← x');
    arr();
    blk('+ Residual','#484f58');
    const resEnd=y-gap/2;
    arr();
    blk('AdaLN (scale+shift)','#bc8cff','γ\'(c)·norm(x)+β\'(c)');
    arr();
    blk('Feed-Forward (MLP)','#3fb950','2-layer, GELU');
    arr();
    blk('+ Residual','#484f58');
    arr();
    blk('Output Tokens x\'','#f0883e','(N, d_model)');

    // Conditioning input arrow
    const condY=y1+bH/2-gap;
    ctx.strokeStyle='#bc8cff'; ctx.lineWidth=2;
    ctx.beginPath(); ctx.moveTo(cx+bW/2+5,condY); ctx.lineTo(cx+bW+40,condY); ctx.stroke();
    ctx.fillStyle='#bc8cff22'; ctx.beginPath(); ctx.roundRect(cx+bW+42,condY-20,110,40,6); ctx.fill();
    ctx.strokeStyle='#bc8cff'; ctx.lineWidth=1; ctx.beginPath(); ctx.roundRect(cx+bW+42,condY-20,110,40,6); ctx.stroke();
    ctx.fillStyle='#bc8cff'; ctx.font='bold 9px Inter,sans-serif'; ctx.textAlign='center'; ctx.textBaseline='middle';
    ctx.fillText('Conditioning c', cx+bW+97, condY-8);
    ctx.font='8px monospace'; ctx.fillStyle='#bc8cff99';
    ctx.fillText('t_emb + class_emb', cx+bW+97, condY+6);
    ctx.fillText('→ MLP → γ,β,γ\',β\'', cx+bW+97, condY+18);

    // Residual skip lines
    ctx.strokeStyle='#ffffff15'; ctx.lineWidth=1.5; ctx.setLineDash([4,3]);
    ctx.beginPath(); ctx.moveTo(cx-bW/2-20,resStart); ctx.lineTo(cx-bW/2-20,resEnd); ctx.stroke();
    ctx.setLineDash([]);
  }

  draw(); window.addEventListener('resize',draw);
})();
</script>

---

## Step 3 — MMDiT: Multi-Modal Bidirectional Attention

**MMDiT** (SD3, FLUX) goes beyond cross-attention conditioning. Text and image **participate in the same attention operation**:

$$
[Q_{img} \| Q_{txt}] \cdot [K_{img} \| K_{txt}]^\top
$$

Image tokens see text tokens and vice versa — much richer conditioning than injecting text only via cross-attention.

FLUX uses a "dual stream" design: **separate** weights for image and text in Q/K/V/FFN blocks, but **shared** attention:

```
Img stream:  x_img → W_q^img·x  ─┐
                                   ├─→ concat → Attention(Q,K,V) → split
Txt stream:  x_txt → W_q^txt·x  ─┘
```

---

## Visualization: Complete Generation Process

<div id="gen-viz" style="background:#0d1117;border-radius:12px;padding:1.5rem;margin:2rem 0;">
<canvas id="gen-canvas" style="width:100%;display:block;"></canvas>
<div style="display:flex;justify-content:center;gap:.8rem;margin-top:1rem;flex-wrap:wrap;">
  <button onclick="genStep(-1)" style="padding:5px 18px;background:#21262d;color:#c9d1d9;border:1px solid #30363d;border-radius:5px;cursor:pointer;">← Previous</button>
  <span id="gen-label" style="color:#8b949e;font-family:monospace;line-height:2;"></span>
  <button onclick="genStep(1)" style="padding:5px 18px;background:#21262d;color:#c9d1d9;border:1px solid #30363d;border-radius:5px;cursor:pointer;">Next →</button>
</div>
<div id="gen-desc" style="color:#c9d1d9;font-size:.85rem;text-align:center;margin-top:.6rem;min-height:2.5rem;padding:0 1rem;"></div>
</div>

<script>
(function(){
  const canvas = document.getElementById('gen-canvas');
  const ctx = canvas.getContext('2d');
  let step=0;
  const steps=[
    {label:'1. Text → Embeddings', desc:'Prompt encoded by T5-XXL (4096-dim) and CLIP. Text embedding guides the entire process.'},
    {label:'2. Sample Noise z_T', desc:'Latent z_T ~ N(0,I) with the same shape as the target latent (e.g., 128×128×16 for 1024×1024px).'},
    {label:'3. Patchify → Tokens', desc:'z_T divided into 2×2 patches → 4096 tokens of dim 64. 2D positional embedding added.'},
    {label:'4. MMDiT Forward (×28 blocks)', desc:'28 MMDiT blocks: bidirectional attention between image tokens (4096) and text tokens (256). Produces velocity field v_θ.'},
    {label:'5. ODE Step (Flow Matching)', desc:'z_{t-Δt} = z_t - Δt·v_θ. Repeated 20-50× until z_0 (clean latent).'},
    {label:'6. VAE Decode → Image', desc:'z_0 decoded by the VAE: 128×128×16 → 1024×1024×3. Final image in pixels.'},
  ];

  window.genStep=function(d){ step=Math.max(0,Math.min(steps.length-1,step+d)); draw(); };

  function draw(){
    const W=canvas.parentElement.offsetWidth-48;
    const H=200; canvas.width=W; canvas.height=H; canvas.style.height=H+'px';
    ctx.fillStyle='#0d1117'; ctx.fillRect(0,0,W,H);

    const s=steps[step];
    document.getElementById('gen-label').textContent=`Step ${step+1}/${steps.length}: ${s.label}`;
    document.getElementById('gen-desc').textContent=s.desc;

    const cx=W/2, cy=H/2;
    const N=steps.length, r=Math.min(cx-40, cy-20);
    const radius=Math.min(r*0.85,120);

    steps.forEach((st,i)=>{
      const angle=(i/N)*2*Math.PI - Math.PI/2;
      const x=cx+radius*Math.cos(angle), y=cy+radius*Math.sin(angle);
      const active=i===step, past=i<step;
      const col=active?'#f0883e':past?'#3fb950':'#21262d';
      const r2=active?22:past?18:14;

      if(i>0){
        const prevAngle=((i-1)/N)*2*Math.PI-Math.PI/2;
        const px=cx+radius*Math.cos(prevAngle), py=cy+radius*Math.sin(prevAngle);
        ctx.strokeStyle=past||active?'#3fb95066':'#21262d';
        ctx.lineWidth=2; ctx.setLineDash(active?[]:[4,3]);
        ctx.beginPath(); ctx.moveTo(px,py); ctx.lineTo(x,y); ctx.stroke();
        ctx.setLineDash([]);
      }

      ctx.fillStyle=col+(active?'':'44');
      ctx.beginPath(); ctx.arc(x,y,r2,0,2*Math.PI); ctx.fill();
      ctx.strokeStyle=col; ctx.lineWidth=active?2.5:1;
      ctx.beginPath(); ctx.arc(x,y,r2,0,2*Math.PI); ctx.stroke();
      ctx.fillStyle=active?'#0d1117':col;
      ctx.font=`bold ${active?12:10}px Inter,sans-serif`;
      ctx.textAlign='center'; ctx.textBaseline='middle';
      ctx.fillText(i+1,x,y);

      // Label outside circle
      const lx=cx+(radius+38)*Math.cos(angle), ly=cy+(radius+38)*Math.sin(angle);
      ctx.fillStyle=active?'#f0883e':past?'#3fb95099':'#484f58';
      ctx.font=`${active?'bold ':''} 9px Inter,sans-serif`;
      ctx.textAlign='center'; ctx.textBaseline='middle';
      const words=st.label.split(': ')[0];
      ctx.fillText(words,lx,ly);
    });
  }

  draw(); window.addEventListener('resize',draw);
})();
</script>

---

## Why Does DiT Scale Better?

U-Nets have pooling and upsample operations that **destroy global information**. Skip connections help, but there is still a hierarchical bottleneck. In DiTs, each token sees all other tokens from the very first block — $O(N^2)$ attention, but with full global access.

This means that when doubling the parameters (more blocks, more dimensions), the DiT makes full use of the extra capacity, while the U-Net hits diminishing returns faster.

| Model | Parameters | Tokens | d_model | Blocks |
|--------|-----------|--------|---------|--------|
| DiT-XL/2 (original) | 675M | 256 | 1152 | 28 |
| SD3 MMDiT | 2B | 1024 | 1536 | 38 |
| FLUX.1-dev | 12B | 4096 | 3072 | 57 |

---

## Simplified Implementation

```python
import torch
import torch.nn as nn

class AdaLN(nn.Module):
    def __init__(self, d_model, d_cond):
        super().__init__()
        self.norm = nn.LayerNorm(d_model, elementwise_affine=False)
        self.proj = nn.Linear(d_cond, 2 * d_model)  # → γ, β

    def forward(self, x, c):
        gamma, beta = self.proj(c).chunk(2, dim=-1)
        return (1 + gamma.unsqueeze(1)) * self.norm(x) + beta.unsqueeze(1)

class DiTBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, d_cond):
        super().__init__()
        self.adaln1 = AdaLN(d_model, d_cond)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.adaln2 = AdaLN(d_model, d_cond)
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))

    def forward(self, x, c):
        h = self.adaln1(x, c)
        x = x + self.attn(h, h, h)[0]      # self-attention
        x = x + self.ff(self.adaln2(x, c)) # FFN
        return x

class DiT(nn.Module):
    def __init__(self, in_channels, patch_size, d_model, n_heads, d_ff, n_layers, d_cond):
        super().__init__()
        self.patch_size = patch_size
        p = patch_size
        self.patchify = nn.Conv2d(in_channels, d_model, p, stride=p)
        self.blocks = nn.ModuleList([DiTBlock(d_model, n_heads, d_ff, d_cond) for _ in range(n_layers)])
        self.norm_out = nn.LayerNorm(d_model)
        self.depatchify = nn.Linear(d_model, p*p*in_channels)

    def forward(self, x, t_emb, cond):
        # x: (B, C, H, W) noisy latent
        B, C, H, W = x.shape
        tokens = self.patchify(x)                       # (B, d, H/p, W/p)
        tokens = tokens.flatten(2).transpose(1, 2)      # (B, N, d)
        c = t_emb + cond                                # combine conditioning
        for block in self.blocks:
            tokens = block(tokens, c)
        tokens = self.norm_out(tokens)
        patches = self.depatchify(tokens)               # (B, N, p*p*C)
        # reshape back to (B, C, H, W)
        p = self.patch_size
        patches = patches.view(B, H//p, W//p, p, p, C).permute(0,5,1,3,2,4).reshape(B,C,H,W)
        return patches  # predicted velocity field v_θ(x_t, t)
```

---

[^1]: Peebles, W., & Xie, S. (2023). [Scalable Diffusion Models with Transformers](https://arxiv.org/abs/2212.09748){:target="_blank"}. ICCV 2023.
[^2]: Esser, P. et al. (2024). [Scaling Rectified Flow Transformers for High-Resolution Image Synthesis (SD3)](https://arxiv.org/abs/2403.03206){:target="_blank"}.
[^3]: Black Forest Labs. (2024). [FLUX.1](https://github.com/black-forest-labs/flux){:target="_blank"}.
[^4]: Dosovitskiy, A. et al. (2021). [An Image is Worth 16×16 Words (ViT)](https://arxiv.org/abs/2010.11929){:target="_blank"}.


---

--8<-- "docs/2026.2/classes/diffusion-transformers/quiz.md"
