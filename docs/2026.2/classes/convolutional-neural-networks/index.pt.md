### Introdução às Redes Neurais Convolucionais (CNNs)

Redes Neurais Convolucionais (CNNs) são uma classe de redes neurais profundas comumente usadas para reconhecimento de imagens, análise de vídeo e outras tarefas envolvendo dados em grade (ex: pixels em imagens). Ao contrário de redes totalmente conectadas, as CNNs exploram hierarquias espaciais por meio de **convoluções**, que aplicam filtros (kernels) aprendíveis a regiões locais da entrada. Isso reduz parâmetros, permite invariância à translação e captura features como bordas, texturas ou objetos.

Uma arquitetura típica de CNN inclui:

- **Camadas convolucionais**: Extraem features via convolução.
- **Funções de ativação** (ex: ReLU): Introduzem não-linearidade.
- **Camadas de pooling** (ex: max pooling): Reduzem dimensões espaciais e a carga computacional.
- **Camadas totalmente conectadas**: Para classificação ou regressão final.
- **Camada de saída**: Frequentemente softmax para classificação.

O treinamento envolve a **passagem direta** (computando previsões) e a **passagem reversa** (retropropagação para atualizar pesos via gradientes). A seguir, focamos na matemática para a camada convolucional, que é o núcleo das CNNs. Assumimos convolução 2D para imagens (formato de entrada: batch_size × canais × altura × largura).


### Passagem Direta em uma Camada Convolucional

A passagem direta computa o mapa de features de saída deslizando um kernel sobre a entrada.

#### Notações Principais:

- Entrada: \( X \in \mathbb{R}^{B \times C_{in} \times H_{in} \times W_{in}} \) (tamanho do batch \( B \), canais de entrada \( C_{in} \), altura \( H_{in} \), largura \( W_{in} \)).
- Kernel (pesos): \( W \in \mathbb{R}^{C_{out} \times C_{in} \times K_h \times K_w} \) (canais de saída \( C_{out} \), altura do kernel \( K_h \), largura \( K_w \)).
- Bias: \( b \in \mathbb{R}^{C_{out}} \) (um por canal de saída).
- Stride: \( s \) (tamanho do passo para deslizar o kernel).
- Padding: \( p \) (zeros adicionados às bordas para controlar o tamanho da saída).
- Saída: \( Y \in \mathbb{R}^{B \times C_{out} \times H_{out} \times W_{out}} \), onde \( H_{out} = \lfloor \frac{H_{in} + 2p - K_h}{s} \rfloor + 1 \).

#### Operação de Convolução:

Para cada posição \( (i, j) \) no mapa de features de saída, para o item de batch \( b \) e canal de saída \( c_{out} \):

\[
Y_{b, c_{out}, i, j} = \sum_{c_{in}=0}^{C_{in}-1} \sum_{m=0}^{K_h-1} \sum_{n=0}^{K_w-1} W_{c_{out}, c_{in}, m, n} \cdot X_{b, c_{in}, s \cdot i + m - p, s \cdot j + n - p} + b_{c_{out}}
\]

Este é essencialmente um produto escalar entre o kernel e um patch local da entrada, somado sobre os canais de entrada, mais o bias.

Após a convolução, aplica-se ativação: \( Y' = f(Y) \), ex: ReLU: \( f(x) = \max(0, x) \).

O pooling (ex: max pooling) sobre uma janela (tamanho \( k \), stride \( s \)) retém o valor máximo em cada patch, reduzindo dimensões.

### Passagem Reversa em uma Camada Convolucional

A passagem reversa computa gradientes para pesos, biases e entradas usando a regra da cadeia, para minimizar a perda \( L \) via gradiente descendente.

#### Gradiente do Bias:
Soma simples sobre dimensões espaciais e batch:

\[
\frac{\partial L}{\partial b_{c_{out}}} = \sum_{b=0}^{B-1} \sum_{i=0}^{H_{out}-1} \sum_{j=0}^{W_{out}-1} \frac{\partial L}{\partial Y_{b, c_{out}, i, j}}
\]

#### Gradiente dos Pesos:
Correlaciona a entrada com o gradiente de saída:

\[
\frac{\partial L}{\partial W_{c_{out}, c_{in}, m, n}} = \sum_{b=0}^{B-1} \sum_{i=0}^{H_{out}-1} \sum_{j=0}^{W_{out}-1} \frac{\partial L}{\partial Y_{b, c_{out}, i, j}} \cdot X_{b, c_{in}, s \cdot i + m - p, s \cdot j + n - p}
\]

#### Gradiente da Entrada:
Convolução "completa" do kernel rotacionado com o gradiente de saída (para propagar o erro de volta):

\[
\frac{\partial L}{\partial X_{b, c_{in}, k, l}} = \sum_{c_{out}=0}^{C_{out}-1} \sum_{m=0}^{K_h-1} \sum_{n=0}^{K_w-1} \frac{\partial L}{\partial Y_{b, c_{out}, i, j}} \cdot W_{c_{out}, c_{in}, m, n}
\]

Para pooling, a passagem reversa realiza upsampling do gradiente (ex: para max pooling, coloca o gradiente apenas na posição máxima).


## Adicional

### Quando uma Rede Neural é Considerada "Profunda"?

O termo "profundo" no contexto de redes neurais refere-se à profundidade da arquitetura, especificamente o número de camadas (particularmente camadas ocultas) que permitem à rede aprender representações hierárquicas e abstratas dos dados. Não há um limiar mínimo universalmente acordado que divida estritamente redes "rasas" de "profundas", pois pode depender do contexto, da tarefa e do uso histórico na pesquisa. No entanto, com base em definições estabelecidas e consenso de especialistas em aprendizado de máquina, uma rede neural é geralmente considerada profunda se tiver pelo menos duas camadas ocultas (além das camadas de entrada e saída).

<div class="grid cards" markdown>

- **Redes Rasas vs. Profundas**

    ---

    - Uma rede neural rasa tipicamente tem 0 ou 1 camada oculta. São suficientes para tarefas simples, mas têm dificuldades com padrões de dados complexos e hierárquicos.
    - Redes profundas, por contraste, empilham múltiplas camadas ocultas para capturar features progressivamente mais abstratas (ex: bordas nas camadas iniciais para reconhecimento de imagem, evoluindo para objetos nas camadas mais profundas).

- **Base Histórica e Teórica**

    ---

    - Modelos iniciais de aprendizado profundo, como os do grupo de Geoffrey Hinton nos anos 2000, apresentavam três camadas ocultas.
    - O teorema da aproximação universal mostra que mesmo uma única camada oculta pode teoricamente aproximar qualquer função, mas na prática, redes mais profundas são mais eficientes para tarefas complexas.

- **Limiares Comuns**

    ---

    - A maioria dos pesquisadores e livros didáticos concorda em **pelo menos 2 camadas ocultas** como mínimo para "profundo".
    - Na prática, redes profundas modernas (ex: CNNs ou transformers) têm dezenas ou centenas de camadas.

</div>

---

## Interativo: Visualizador de Filtro Convolucional

Selecione um filtro e veja-o deslizar sobre uma imagem sintética, produzindo o mapa de features.

<div id="conv-viz" style="background:#0d1117;border-radius:12px;padding:1.5rem;margin:2rem 0;font-family:Inter,sans-serif;color:#e6edf3;">
<div style="display:flex;gap:.8rem;flex-wrap:wrap;margin-bottom:1rem;align-items:center;">
  <div>
    <label style="color:#8b949e;font-size:.85rem;">Filtro: </label>
    <select id="conv-filter" onchange="convDraw()" style="background:#161b22;color:#c9d1d9;border:1px solid #30363d;border-radius:4px;padding:3px 8px;">
      <option value="edge-h">Borda Horizontal</option>
      <option value="edge-v">Borda Vertical</option>
      <option value="blur">Blur (média)</option>
      <option value="sharpen">Nitidez</option>
      <option value="emboss">Relevo</option>
    </select>
  </div>
  <div>
    <label style="color:#8b949e;font-size:.85rem;">Stride: </label>
    <select id="conv-stride" onchange="convReset()" style="background:#161b22;color:#c9d1d9;border:1px solid #30363d;border-radius:4px;padding:3px 8px;">
      <option value="1">1</option>
      <option value="2">2</option>
    </select>
  </div>
  <button onclick="convAnimate()" id="conv-btn" style="padding:5px 18px;background:#3fb950;color:#0d1117;border:none;border-radius:5px;cursor:pointer;font-weight:bold;">&#9654; Animar</button>
  <button onclick="convReset()" style="padding:5px 14px;background:#21262d;color:#c9d1d9;border:1px solid #30363d;border-radius:5px;cursor:pointer;">&#8635; Resetar</button>
</div>
<canvas id="conv-canvas" style="width:100%;display:block;border-radius:8px;"></canvas>
<div id="conv-formula" style="margin-top:.8rem;font-family:monospace;font-size:.8rem;color:#8b949e;background:#161b22;border-radius:5px;padding:.7rem;"></div>
</div>

<script>
(function(){
  const filters = {
    'edge-h':{kernel:[[-1,-1,-1],[0,0,0],[1,1,1]],label:'Detector de Borda Horizontal'},
    'edge-v':{kernel:[[-1,0,1],[-1,0,1],[-1,0,1]],label:'Detector de Borda Vertical'},
    'blur':{kernel:[[1/9,1/9,1/9],[1/9,1/9,1/9],[1/9,1/9,1/9]],label:'Blur por Média'},
    'sharpen':{kernel:[[0,-1,0],[-1,5,-1],[0,-1,0]],label:'Nitidez'},
    'emboss':{kernel:[[-2,-1,0],[-1,1,1],[0,1,2]],label:'Relevo'},
  };
  const img=[[0,0,0,0,1,1,1,1],[0,0,0,0,1,1,1,1],[0,0,0,0,1,1,1,1],[0,0,0,0,1,1,1,1],[1,1,1,1,0,0,0,0],[1,1,1,1,0,0,0,0],[1,1,1,1,0,0,0,0],[1,1,1,1,0,0,0,0]];
  const N=8,K=3;
  let animId=null,animPos=0,running=false;
  function getStride(){return +document.getElementById('conv-stride').value;}
  function getFilter(){return filters[document.getElementById('conv-filter').value];}
  function convolve(kernel,stride){const out=[];for(let r=0;r<=N-K;r+=stride){const row=[];for(let c=0;c<=N-K;c+=stride){let v=0;for(let kr=0;kr<K;kr++)for(let kc=0;kc<K;kc++)v+=kernel[kr][kc]*img[r+kr][c+kc];row.push(v);}out.push(row);}return out;}
  function normalize(feat){let mn=Infinity,mx=-Infinity;feat.forEach(r=>r.forEach(v=>{mn=Math.min(mn,v);mx=Math.max(mx,v);}));return feat.map(r=>r.map(v=>mx===mn?0.5:(v-mn)/(mx-mn)));}

  window.convDraw=function(pos){
    const canvas=document.getElementById('conv-canvas');
    const ctx=canvas.getContext('2d');
    const W=canvas.parentElement.offsetWidth-48,H=260;
    canvas.width=W;canvas.height=H;canvas.style.height=H+'px';
    ctx.fillStyle='#161b22';ctx.fillRect(0,0,W,H);
    const stride=getStride(),flt=getFilter(),feat=convolve(flt.kernel,stride),featN=normalize(feat),outSz=feat.length;
    const cellIn=Math.min(26,Math.floor((W*0.35)/N)),cellOut=Math.min(32,Math.floor((W*0.32)/outSz));
    const kArea=K*24+10,inX=8,inY=(H-N*cellIn)/2,kx=inX+N*cellIn+12,ky=(H-K*24)/2,outX=kx+kArea+30,outY=(H-outSz*cellOut)/2;
    for(let r=0;r<N;r++)for(let c=0;c<N;c++){ctx.fillStyle=img[r][c]?'#c9d1d9':'#21262d';ctx.fillRect(inX+c*cellIn,inY+r*cellIn,cellIn-1,cellIn-1);}
    ctx.fillStyle='#484f58';ctx.font='9px Inter,sans-serif';ctx.textAlign='center';ctx.fillText('Entrada ('+N+'×'+N+')',inX+N*cellIn/2,inY+N*cellIn+14);
    ctx.fillStyle='#8b949e';ctx.font='bold 9px Inter,sans-serif';ctx.textAlign='center';ctx.fillText('Kernel',kx+K*24/2,ky-10);
    for(let r=0;r<K;r++)for(let c=0;c<K;c++){const v=flt.kernel[r][c];ctx.fillStyle=v>0?'#1f3d1f':v<0?'#3d1a1a':'#21262d';ctx.beginPath();ctx.roundRect(kx+c*24,ky+r*24,22,22,3);ctx.fill();ctx.fillStyle=v>0?'#3fb950':v<0?'#ff7b72':'#484f58';ctx.font='bold 9px monospace';ctx.textAlign='center';ctx.textBaseline='middle';ctx.fillText(typeof v==='number'&&v%1!==0?v.toFixed(2):v,kx+c*24+11,ky+r*24+11);}
    const ax=outX-16,ay=H/2;ctx.strokeStyle='#484f58';ctx.lineWidth=1.5;ctx.beginPath();ctx.moveTo(ax-10,ay);ctx.lineTo(ax,ay);ctx.stroke();ctx.fillStyle='#484f58';ctx.beginPath();ctx.moveTo(ax,ay);ctx.lineTo(ax-6,ay-4);ctx.lineTo(ax-6,ay+4);ctx.fill();
    for(let r=0;r<outSz;r++)for(let c=0;c<outSz;c++){const v=featN[r][c],hue=v>0.5?(30+v*60)|0:(220);ctx.fillStyle='hsl('+hue+','+(v>0.5?80:v*60|0)+'%,'+(25+v*35|0)+'%)';ctx.fillRect(outX+c*cellOut,outY+r*cellOut,cellOut-1,cellOut-1);if(cellOut>=16){ctx.fillStyle='rgba(0,0,0,0.6)';ctx.font='7px monospace';ctx.textAlign='center';ctx.textBaseline='middle';ctx.fillText(feat[r][c].toFixed(1),outX+c*cellOut+cellOut/2,outY+r*cellOut+cellOut/2);}}
    ctx.fillStyle='#3fb950';ctx.font='9px Inter,sans-serif';ctx.textAlign='center';ctx.textBaseline='alphabetic';ctx.fillText('Mapa de Features ('+outSz+'×'+outSz+')',outX+outSz*cellOut/2,outY+outSz*cellOut+14);
    if(pos!==undefined){const rs=Math.floor(pos/outSz),cs=pos%outSz,r0=rs*stride,c0=cs*stride;ctx.strokeStyle='#f0883e';ctx.lineWidth=2;ctx.beginPath();ctx.roundRect(inX+c0*cellIn-1,inY+r0*cellIn-1,K*cellIn+2,K*cellIn+2,3);ctx.stroke();ctx.beginPath();ctx.roundRect(outX+cs*cellOut-1,outY+rs*cellOut-1,cellOut+2,cellOut+2,3);ctx.stroke();}
    document.getElementById('conv-formula').innerHTML='Filtro: <span style="color:#3fb950;">'+flt.label+'</span> &nbsp;|&nbsp; Tamanho saída: <span style="color:#58a6ff;">('+N+'-'+K+')/'+stride+'+1 = '+outSz+'</span> &nbsp;|&nbsp; Params: <span style="color:#f0883e;">'+K*K+'</span> compartilhados sobre todas as '+(N*N)+' posições';
  };
  window.convAnimate=function(){running=!running;document.getElementById('conv-btn').textContent=running?'⏸ Pausar':'▶ Animar';document.getElementById('conv-btn').style.background=running?'#d29922':'#3fb950';if(running)animLoop();else cancelAnimationFrame(animId);};
  window.convReset=function(){running=false;animPos=0;cancelAnimationFrame(animId);document.getElementById('conv-btn').textContent='▶ Animar';document.getElementById('conv-btn').style.background='#3fb950';convDraw();};
  function animLoop(){const stride=getStride(),outSz=Math.floor((N-K)/stride)+1;convDraw(animPos);animPos=(animPos+1)%(outSz*outSz);if(running)animId=setTimeout(animLoop,90);}
  convDraw();window.addEventListener('resize',convDraw);
})();
</script>

---

--8<-- "docs/2026.2/classes/convolutional-neural-networks/quiz.pt.md"
