## Geração Autorregressiva de Imagens

Modelos de difusão dominam a geração de imagens desde 2020 — mas existe uma abordagem radicalmente diferente que ganhou força: **tratar imagens como sequências de tokens discretos e gerá-las da mesma forma que modelos de linguagem geram texto**.

Essa é a abordagem por trás da geração de imagens nativa do **Gemini**, do **GPT-4o** e de modelos como **Chameleon** (Meta) e **LlamaGen**.

---

## O Problema: Como Tokenizar uma Imagem?

Texto é naturalmente discreto (palavras, subpalavras). Imagens são contínuas — pixels em $[0,255]^3$. Para usar geração autorregressiva, precisamos de um **vocabulário visual**.

A solução: **VQ-GAN** (Vector Quantization GAN)[^1] aprende um *codebook* de $K$ vetores. O encoder mapeia qualquer patch de imagem para o vetor mais próximo do codebook — convertendo a imagem em uma grade de índices inteiros.

<div id="vqvae-viz" style="background:#0d1117;border-radius:12px;padding:1.5rem;margin:2rem 0;overflow:hidden;">
<canvas id="vqvae-canvas" style="width:100%;display:block;border-radius:8px;"></canvas>
<div style="display:flex;justify-content:center;gap:.8rem;margin-top:1rem;flex-wrap:wrap;">
  <button onclick="vqStep(-1)" style="padding:5px 18px;background:#21262d;color:#c9d1d9;border:1px solid #30363d;border-radius:5px;cursor:pointer;">← Anterior</button>
  <span id="vq-label" style="color:#8b949e;font-family:monospace;line-height:2.2;font-size:.85rem;"></span>
  <button onclick="vqStep(1)" style="padding:5px 18px;background:#21262d;color:#c9d1d9;border:1px solid #30363d;border-radius:5px;cursor:pointer;">Próximo →</button>
</div>
<div id="vq-desc" style="color:#c9d1d9;font-size:.85rem;text-align:center;margin-top:.6rem;min-height:2.5rem;padding:0 1rem;'"></div>
</div>

<script>
(function(){
  const canvas = document.getElementById('vqvae-canvas');
  const ctx = canvas.getContext('2d');
  let vqS=0;

  const vqSteps=[
    {label:'1. Imagem original', desc:'Imagem 256×256px de entrada. Cada pixel tem valor contínuo RGB.'},
    {label:'2. Encoder CNN', desc:'A CNN contrai a imagem para representação latente 32×32×256. Cada posição captura um patch de 8×8 pixels.'},
    {label:'3. Quantização: correspondência ao codebook', desc:'Cada vetor latente de 256 dimensões é mapeado ao vetor mais próximo no codebook (K=8192 entradas). O índice substitui o vetor.'},
    {label:'4. Mapa de tokens discretos', desc:'Resultado: grade 32×32 de índices inteiros em [0, 8191]. 1024 tokens representam a imagem inteira.'},
    {label:'5. Decoder CNN', desc:'O decoder reconstrói a imagem a partir dos vetores do codebook. Treinado com perda de reconstrução + discriminador GAN.'},
  ];

  window.vqStep=function(d){ vqS=Math.max(0,Math.min(vqSteps.length-1,vqS+d)); drawVQ(); };

  function rng(seed){ let x=Math.sin(seed*31.7)*43758.5; return x-Math.floor(x); }

  function drawVQ(){
    const W=canvas.parentElement.offsetWidth-48;
    const H=220; canvas.width=W; canvas.height=H; canvas.style.height=H+'px';
    ctx.fillStyle='#161b22'; ctx.fillRect(0,0,W,H);

    const s=vqSteps[vqS];
    document.getElementById('vq-label').textContent=`Passo ${vqS+1}/${vqSteps.length}: ${s.label}`;
    document.getElementById('vq-desc').textContent=s.desc;

    const boxH=160, boxY=(H-boxH)/2;
    const components=[
      {label:'Imagem\n256×256', w:80},
      {label:'Encoder\nCNN', w:60},
      {label:'Latente\n32×32×256', w:80},
      {label:'Codebook\nK=8192', w:80},
      {label:'Tokens\n32×32', w:80},
      {label:'Decoder\nCNN', w:60},
      {label:'Imagem\nReconstruída', w:80},
    ];
    const totalW=components.reduce((s,c)=>s+c.w+24,0)-24;
    let x=(W-totalW)/2;
    const colors=['#484f58','#58a6ff','#3fb950','#bc8cff','#f0883e','#58a6ff','#484f58'];

    components.forEach((comp,i)=>{
      const col=colors[i];
      const bH=Math.min(boxH, comp.w*1.4);
      const by=boxY+(boxH-bH)/2;
      const alpha=(i===0||i<=vqS)?1:0.25;
      ctx.globalAlpha=alpha;
      ctx.fillStyle=col+'22';
      ctx.beginPath(); ctx.roundRect(x,by,comp.w,bH,6); ctx.fill();
      ctx.strokeStyle=col; ctx.lineWidth=1.5;
      ctx.beginPath(); ctx.roundRect(x,by,comp.w,bH,6); ctx.stroke();

      if(i===0||i===6){
        const gs=8, gx=x+6, gy=by+6, gsz=(comp.w-12)/gs;
        for(let r=0;r<gs;r++) for(let c=0;c<gs;c++){
          const v=rng(r*8+c+(i===6?100:0));
          ctx.fillStyle=`hsl(${v*360|0},40%,${30+v*30|0}%)`;
          ctx.fillRect(gx+c*gsz,gy+r*gsz,gsz-1,gsz-1);
        }
      } else if(i===2){
        const gs=6, gx=x+8, gy=by+8, gsz=(comp.w-16)/gs;
        for(let r=0;r<gs;r++) for(let c=0;c<gs;c++){
          ctx.fillStyle=`hsl(${140+rng(r*6+c)*40|0},60%,${25+rng(r*6+c+50)*25|0}%)`;
          ctx.fillRect(gx+c*gsz,gy+r*gsz,gsz-1,gsz-1);
        }
      } else if(i===3){
        for(let j=0;j<12;j++){
          const vy=by+10+j*12, vx=x+10;
          const v=rng(j*7);
          ctx.fillStyle=`hsl(${270+v*60|0},60%,${30+v*20|0}%)`;
          ctx.fillRect(vx,vy,comp.w-20,9);
          ctx.fillStyle='#ffffff33'; ctx.font='7px monospace'; ctx.textAlign='left'; ctx.textBaseline='middle';
          ctx.fillText((j*700+341)+'',vx+2,vy+4);
        }
      } else if(i===4){
        const gs=8, gx=x+4, gy=by+4, gsz=(comp.w-8)/gs;
        for(let r=0;r<gs;r++) for(let c=0;c<gs;c++){
          const idx=Math.floor(rng(r*8+c)*8192);
          ctx.fillStyle=`hsl(${((idx/8192)*360)|0},70%,35%)`;
          ctx.fillRect(gx+c*gsz,gy+r*gsz,gsz-1,gsz-1);
        }
      }

      ctx.globalAlpha=alpha;
      ctx.fillStyle=col; ctx.font='9px Inter,sans-serif'; ctx.textAlign='center'; ctx.textBaseline='top';
      comp.label.split('\n').forEach((l,li)=>ctx.fillText(l,x+comp.w/2,by+bH+4+li*11));

      if(i<components.length-1){
        const nextX=x+comp.w+24;
        ctx.globalAlpha=i<vqS?1:0.2;
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

## Geração Autorregressiva de Tokens

Com um codebook treinado, podemos representar qualquer imagem como uma sequência de $N$ índices inteiros. Geramos então essa sequência **exatamente como um LLM gera texto**:

$$
p(t_1, t_2, \ldots, t_N) = \prod_{i=1}^{N} p(t_i \mid t_1, \ldots, t_{i-1}, \text{prompt})
$$

Cada token é gerado um de cada vez, condicionado em todos os anteriores e no prompt de texto.

<div id="ar-viz" style="background:#0d1117;border-radius:12px;padding:1.5rem;margin:2rem 0;">
<canvas id="ar-canvas" style="width:100%;display:block;border-radius:8px;"></canvas>
<div style="display:flex;gap:.8rem;justify-content:center;margin-top:1rem;flex-wrap:wrap;">
  <button id="ar-btn" onclick="arToggle()" style="padding:6px 22px;background:#3fb950;color:#0d1117;border:none;border-radius:5px;cursor:pointer;font-weight:bold;">&#9654; Gerar</button>
  <button onclick="arReset()" style="padding:6px 16px;background:#21262d;color:#c9d1d9;border:1px solid #30363d;border-radius:5px;cursor:pointer;">&#8635; Resetar</button>
  <label style="color:#8b949e;font-size:.85rem;line-height:2.2;">Ordem: </label>
  <select id="ar-order" onchange="arReset()" style="background:#161b22;color:#c9d1d9;border:1px solid #30363d;border-radius:4px;padding:3px 8px;">
    <option value="raster">Raster (←→)</option>
    <option value="masked">Mascarado (aleatório)</option>
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
  const G=16;

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
    const gridSz=Math.min(H-20,W*0.45-10);
    const cell=gridSz/G;
    const gx=20, gy=(H-gridSz)/2;

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

    if(generated.size<order.length && running){
      const [r,c]=order[generated.size];
      ctx.fillStyle='#f0883e88';
      ctx.fillRect(gx+c*cell,gy+r*cell,cell,cell);
      ctx.strokeStyle='#f0883e'; ctx.lineWidth=1.5;
      ctx.strokeRect(gx+c*cell,gy+r*cell,cell,cell);
    }

    ctx.strokeStyle='#30363d'; ctx.lineWidth=1;
    ctx.strokeRect(gx,gy,gridSz,gridSz);
    ctx.fillStyle='#484f58'; ctx.font='9px Inter,sans-serif'; ctx.textAlign='center';
    ctx.fillText(`${G}×${G} = ${G*G} tokens`,gx+gridSz/2,gy+gridSz+14);

    const pw=Math.min(200,W-gx-gridSz-60);
    if(pw>80){
      const px=gx+gridSz+30, py=(H-120)/2;
      ctx.fillStyle='#8b949e'; ctx.font='9px Inter,sans-serif'; ctx.textAlign='left';
      ctx.fillText('p(próximo token|contexto)',px,py-8);
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

    document.getElementById('ar-stats').textContent=
      `Gerados: ${generated.size}/${G*G} tokens  (${((generated.size/(G*G))*100).toFixed(0)}%)`;
  }

  window.arToggle=function(){
    running=!running;
    document.getElementById('ar-btn').textContent=running?'⏸ Pausar':'▶ Gerar';
    document.getElementById('ar-btn').style.background=running?'#d29922':'#3fb950';
    if(running) animate();
    else cancelAnimationFrame(animId);
  };

  window.arReset=function(){
    running=false; cancelAnimationFrame(animId);
    document.getElementById('ar-btn').textContent='▶ Gerar';
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
      document.getElementById('ar-btn').textContent='▶ Gerar';
      document.getElementById('ar-btn').style.background='#3fb950';
      draw();
    }
  }

  buildOrder(); draw();
  window.addEventListener('resize',draw);
})();
</script>

---

## MaskGIT: Geração Paralela por Mascaramento

A geração puramente autorregressiva é **lenta**: 1024 tokens = 1024 passes de modelo. O **MaskGIT**[^2] acelera isso com geração paralela iterativa:

1. Começa com **todos os tokens mascarados** `[MASK]`
2. A cada iteração, prediz **todos os tokens simultaneamente** (bidirecional!)
3. "Revela" apenas os tokens com maior confiança
4. Repete com menos tokens mascarados

Em apenas **8–12 iterações**, gera 1024 tokens — contra 1024 iterações do AR puro.

<div id="mask-viz" style="background:#0d1117;border-radius:12px;padding:1.5rem;margin:2rem 0;">
<canvas id="mask-canvas" style="width:100%;display:block;border-radius:8px;"></canvas>
<div style="display:flex;gap:.8rem;justify-content:center;margin-top:1rem;flex-wrap:wrap;">
  <button onclick="maskStep(-1)" style="padding:5px 18px;background:#21262d;color:#c9d1d9;border:1px solid #30363d;border-radius:5px;cursor:pointer;">← Anterior</button>
  <span id="mask-label" style="color:#f0883e;font-family:monospace;font-size:.85rem;line-height:2;"></span>
  <button onclick="maskStep(1)" style="padding:5px 18px;background:#21262d;color:#c9d1d9;border:1px solid #30363d;border-radius:5px;cursor:pointer;">Próximo →</button>
</div>
</div>

<script>
(function(){
  const canvas=document.getElementById('mask-canvas');
  const ctx=canvas.getContext('2d');
  let maskS=0;
  const G=10, K=8;
  function rng(s){let x=Math.sin(s*31.7)*43758.5;return x-Math.floor(x);}
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
      `Iteração ${maskS+1}/${K}: ${revealed}/${totalToks} tokens revelados (${((1-frac)*100).toFixed(0)}%)`;
    const steps=Math.min(K,Math.floor((W-20)/60));
    const gSz=Math.min(50,(W-20)/steps-8);
    const cell=gSz/G;
    const startX=(W-(steps*(gSz+8)))/2;
    for(let s=0;s<steps;s++){
      const f=schedule[s];
      const rev=Math.round(totalToks*(1-f));
      const gx=startX+s*(gSz+8), gy=(H-gSz)/2;
      const isActive=s===maskS;
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

## Any-to-Any: Gemini, GPT-4o e Chameleon

O passo final é remover a distinção entre tokens de texto e tokens de imagem. Modelos **any-to-any** tratam tudo como uma sequência de tokens:

```
[TEXTO: "uma foto de"] [IMG_TOK_3742] [IMG_TOK_891] ... [IMG_TOK_5531] [TEXTO: "gato"]
```

O modelo Transformer padrão processa essa sequência misturada naturalmente.

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
      {label:'Descreva',type:'text'},
      {label:'a imagem:',type:'text'},
      {label:'🟦🟦\n🟧🟦',type:'img'},
      {label:'img_4821',type:'img'},
      {label:'img_0331',type:'img'},
      {label:'img_7102',type:'img'},
      {label:'→ Resposta:',type:'text'},
      {label:'"céu azul\ne pôr do sol"',type:'text'},
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
    ctx.fillText('● Texto',10,H-8);
    ctx.fillStyle='#3fb950'; ctx.fillText('● Imagem (tokens VQ)',60,H-8);
  }
  draw(); window.addEventListener('resize',draw);
})();
</script>

### Como Cada Modelo Implementa Isso

| Modelo | Tokenizador visual | Geração | Treinamento |
|--------|------------------|---------|------------|
| **Chameleon** (Meta) | VQ-VAE (8192 códigos) | Autorregressivo puro | Texto + imagem juntos desde o início |
| **Gemini 2.0** (Google) | Tokenizador proprietário | AR + decoder de difusão | Multimodal nativo |
| **GPT-4o** (OpenAI) | Tokens visuais discretos | AR + decoder de difusão | Multimodal nativo |
| **LlamaGen** | VQGAN (16384 códigos) | AR com LLaMA | Inicializa de LLaMA pré-treinado |

---

## AR vs. Difusão: Quando Usar Cada Um?

<div style="display:grid;grid-template-columns:1fr 1fr;gap:1rem;margin:1.5rem 0;">

<div style="background:#0d1117;border-left:3px solid #3fb950;padding:1rem;border-radius:8px;">
<strong style="color:#3fb950;">Geração Autorregressiva</strong>
<ul style="color:#c9d1d9;font-size:.85rem;margin:.5rem 0;padding-left:1.2rem;">
<li>Unifica texto e imagem na mesma arquitetura</li>
<li>Melhor para multimodal any-to-any</li>
<li>Aproveita toda a infraestrutura de LLMs</li>
<li>Escala bem com mais dados</li>
<li><strong style="color:#ff7b72;">Lento</strong>: 1 token por vez</li>
</ul>
</div>

<div style="background:#0d1117;border-left:3px solid #58a6ff;padding:1rem;border-radius:8px;">
<strong style="color:#58a6ff;">Difusão (DDPM / Flow Matching)</strong>
<ul style="color:#c9d1d9;font-size:.85rem;margin:.5rem 0;padding-left:1.2rem;">
<li>Melhor qualidade de imagem isolada</li>
<li>Geração global coerente</li>
<li>Mais controle (guidance, cfg scale)</li>
<li>Mais rápido por imagem que AR</li>
<li><strong style="color:#ff7b72;">Não unifica com texto nativamente</strong></li>
</ul>
</div>

</div>

A tendência atual: **híbridos** — um backbone LLM autorregressivo para entendimento e raciocínio, com um decoder de difusão para renderizar a imagem final com alta qualidade. É exatamente o que o GPT-4o faz.

---

## Implementação: VQ-GAN + Transformer AR

```python
import torch
import torch.nn as nn

# 1. Quantizador vetorial
class VectorQuantizer(nn.Module):
    def __init__(self, n_codes, d_code):
        super().__init__()
        self.codebook = nn.Embedding(n_codes, d_code)

    def forward(self, z):
        # z: (B, H, W, d_code) — latentes do encoder
        flat = z.view(-1, z.shape[-1])
        # Distâncias ao codebook
        dists = torch.cdist(flat, self.codebook.weight)
        indices = dists.argmin(dim=-1)            # índice do código mais próximo
        quantized = self.codebook(indices).view_as(z)
        # Straight-through estimator para backprop
        quantized_st = z + (quantized - z).detach()
        return quantized_st, indices.view(z.shape[:3])

# 2. Geração autorregressiva com modelo estilo GPT
class ImageGPT(nn.Module):
    def __init__(self, n_codes, seq_len, d_model, n_heads, n_layers):
        super().__init__()
        self.tok_emb = nn.Embedding(n_codes + 1, d_model)  # +1 para token BOS
        self.pos_emb = nn.Embedding(seq_len + 1, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, n_heads, d_model*4, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, n_layers)
        self.head = nn.Linear(d_model, n_codes)

    def forward(self, tokens):
        B, T = tokens.shape
        pos = torch.arange(T, device=tokens.device).unsqueeze(0)
        x = self.tok_emb(tokens) + self.pos_emb(pos)
        # Máscara causal
        mask = torch.triu(torch.ones(T, T, device=tokens.device), diagonal=1).bool()
        x = self.transformer(x, mask=mask)
        return self.head(x)  # logits sobre n_codes

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

--8<-- "docs/2026.2/classes/autoregressive-generation/quiz.pt.md"
