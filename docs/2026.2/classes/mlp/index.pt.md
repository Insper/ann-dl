

## Apresentação: Perceptrons Multi-Camada (MLPs)

<center>
``` mermaid
flowchart LR
    classDef default fill:transparent,stroke:#333,stroke-width:1px;
    classDef others fill:transparent,stroke:transparent,stroke-width:0px;
    subgraph in[" "]
        in1@{ shape: circle, label: " " }
        in2@{ shape: circle, label: " " }
        in3@{ shape: circle, label: " " }
        inn@{ shape: circle, label: " " }
    end
    subgraph entrada
        x1(["x<sub>1</sub>"])
        x2(["x<sub>2</sub>"])
        x3(["x<sub>3</sub>"])
        xd(["..."]):::others
        xn(["x<sub>n</sub>"])
        xb(["1"])
    end
    subgraph oculta
        direction TB
        h1(["h<sub>1</sub>"])
        h2(["h<sub>2</sub>"])
        hd(["..."]):::others
        hm(["h<sub>m</sub>"])
        hb(["1"])
    end
    subgraph saida
        y1(["y<sub>1</sub>"])
        yd(["..."]):::others
        yk(["y<sub>k</sub>"])
    end
    in1@{ shape: circle, label: " " } --> x1
    in2@{ shape: circle, label: " " } --> x2
    in3@{ shape: circle, label: " " } --> x3
    inn@{ shape: circle, label: " " } --> xn

    x1 -->|"w<sub>11</sub>"|h1
    x1 -->|"w<sub>12</sub>"|h2
    x1 -->|"w<sub>1n</sub>"|hm
    x2 -->|"w<sub>21</sub>"|h1
    x2 -->|"w<sub>22</sub>"|h2
    x2 -->|"w<sub>2n</sub>"|hm
    x3 -->|"w<sub>31</sub>"|h1
    x3 -->|"w<sub>32</sub>"|h2
    x3 -->|"w<sub>3n</sub>"|hm
    xn -->|"w<sub>i1</sub>"|h1
    xn -->|"w<sub>i2</sub>"|h2
    xn -->|"w<sub>in</sub>"|hm
    xb -->|"b<sup>i</sup><sub>1</sub>"|h1
    xb -->|"b<sup>i</sup><sub>2</sub>"|h2
    xb -->|"b<sup>i</sup><sub>n</sub>"|hm

    h1 -->|"v<sub>11</sub>"|y1
    h1 -->|"v<sub>1k</sub>"|yk
    h2 -->|"v<sub>21</sub>"|y1
    h2 -->|"v<sub>2k</sub>"|yk
    hm -->|"v<sub>m1</sub>"|y1
    hm -->|"v<sub>mk</sub>"|yk
    hb -->|"b<sup>h</sup><sub>1</sub>"|y1
    hb -->|"b<sup>h</sup><sub>k</sub>"|yk

    y1 --> out1@{ shape: dbl-circ, label: " " }
    yk --> outn@{ shape: dbl-circ, label: " " }

    style in fill:#fff,stroke:#666,stroke-width:0px
    style entrada fill:#fff,stroke:#666,stroke-width:1px
    style oculta fill:#fff,stroke:#666,stroke-width:1px
    style saida fill:#fff,stroke:#666,stroke-width:1px
```
<i>Arquitetura do Perceptron Multi-Camada (MLP).</i>
</center>

$$
y_k = \sigma \left( \sum_{j=1}^{m} \sigma \left( \sum_{i=1}^{n} x_i w_{ij} + b^{h}_{i} \right) v_{jk} + b^{y}_{j} \right)
$$

onde:

- $y_k$ é a saída para o $k$-ésimo neurônio de saída.
- $x_i$ são as features de entrada.
- $w_{ij}$ são os pesos conectando a $i$-ésima entrada ao $j$-ésimo neurônio oculto.
- $v_{jk}$ são os pesos conectando o $j$-ésimo neurônio oculto ao $k$-ésimo neurônio de saída.
- $b^{h}_{i}$ é o bias para o $i$-ésimo neurônio oculto.
- $b^{y}_{j}$ é o bias para o $j$-ésimo neurônio de saída.
- $m$ é o número de neurônios ocultos.
- $n$ é o número de features de entrada.
- $\sigma$ é a função de ativação aplicada às somas ponderadas em cada camada, como sigmoid, tanh ou ReLU.


Representação matricial da arquitetura MLP:

$$
\begin{align*}
\text{Camada de Entrada:} & \quad \mathbf{x} = [x_1, x_2, \ldots, x_n]^T \\
\text{Camada Oculta:} & \quad \mathbf{h} = \sigma (\mathbf{W} \mathbf{x} + \mathbf{b}^h) \\
\text{Camada de Saída:} & \quad \mathbf{y} = \sigma (\mathbf{V} \mathbf{h} + \mathbf{b}^y)
\end{align*}
$$


| Sigmoid | Tanh    | ReLU  |
|---------|---------|-------|
| $\sigma(x) = \displaystyle \frac{1}{1 + e^{-x}}$ | $\tanh(x) = \displaystyle \frac{e^{2x} - 1}{e^{2x} + 1}$ | $\text{ReLU}(x) = \max(0, x)$ |
| $\sigma'(x) = \sigma(x)(1 - \sigma(x))$ | $\tanh'(x) = 1 - \tanh^2(x)$ | $\text{ReLU}'(x) = \begin{cases} 1 & \text{se } x > 0 \\ 0 & \text{se } x \leq 0 \end{cases}$ |
| ![Sigmoid](https://upload.wikimedia.org/wikipedia/commons/thumb/5/5b/Activation_logistic.svg/2560px-Activation_logistic.svg.png) | ![Tanh](https://upload.wikimedia.org/wikipedia/commons/thumb/c/cb/Activation_tanh.svg/2560px-Activation_tanh.svg.png) | ![ReLU](https://upload.wikimedia.org/wikipedia/commons/thumb/f/fe/Activation_rectified_linear.svg/2560px-Activation_rectified_linear.svg.png) |
| Sigmoid é uma curva S suave que produz valores entre 0 e 1, sendo adequada para classificação binária. | Tanh é uma curva suave que produz valores entre -1 e 1, centralizando os dados em zero, o que pode ajudar na convergência. | ReLU é uma função linear por partes que produz zero para entradas negativas e a própria entrada para positivas, permitindo treinamento mais rápido e reduzindo o gradiente desvanecente. |

A retropropagação é o algoritmo usado para treinar MLPs ajustando pesos e biases com base no erro entre a saída prevista e o alvo real. O processo envolve dois passos principais:

1. **Passagem Direta**: Os dados de entrada passam pela rede, camada por camada, para calcular a saída. A saída é comparada ao valor alvo para calcular a perda.
2. **Cálculo da Perda**: Calcula-se a perda entre a saída prevista e o alvo real usando uma função de perda, como erro quadrático médio ou cross-entropy.
3. **Passagem Reversa**: O erro é propagado de volta pela rede para calcular os gradientes da perda em relação a cada peso e bias. Esses gradientes são usados para atualizar os pesos e biases usando um algoritmo de otimização, como SGD ou Adam.

## Passagem Direta (Feedforward)

Considere um MLP com:

- 2 neurônios de entrada: $x_1$ e $x_2$
- 1 camada oculta com 2 neurônios: $h_1$ e $h_2$
- 1 neurônio de saída: $y$

Assumimos funções de ativação sigmoid para as camadas oculta e de saída:

$$\displaystyle \sigma(z) = \frac{1}{1 + e^{-z}}$$

com derivada $\sigma'(z) = \sigma(z)(1 - \sigma(z))$.

Em termos matemáticos, o processo de passagem direta é descrito como:

$$
\begin{align*}
\text{Camada de Entrada:} & \quad \mathbf{x} = [x_1, x_2]^T \\
\text{Camada Oculta:} & \quad \mathbf{h} = \sigma (\mathbf{W} \mathbf{x} + \mathbf{b}^h) \\
\text{Camada de Saída:} & \quad \mathbf{y} = \sigma (\mathbf{V} \mathbf{h} + \mathbf{b}^y)
\end{align*}
$$

ou, mais canonicamente:

$$
\hat{y} = \sigma \left(
v_{11}
    \sigma \left(w_{11} x_1 + w_{21} x_2 + b^h_1\right)
    + v_{21}
    \sigma \left(w_{12} x_1 + w_{22} x_2 + b^h_2\right) + b^y_1
\right)
$$

1. Pré-ativação da camada oculta:

$$z_1 = w_{11} x_1 + w_{21} x_2 + b^h_1, \quad z_2 = w_{12} x_1 + w_{22} x_2 + b^h_2$$

2. Ativações da camada oculta:

$$h_1 = \sigma(z_1), \quad h_2 = \sigma(z_2)$$

3. Pré-ativação da camada de saída:

$$u = v_{11} h_1 + v_{21} h_2 + b^y_1$$

4. Ativação da camada de saída:

$$\hat{y} = \sigma(u)$$

## Cálculo da Perda

A função de perda quantifica a diferença entre a saída prevista e o alvo real. Para tarefas de regressão, o Erro Quadrático Médio (MSE) é comum:

$$L = \text{MSE} = \frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2$$

## Retropropagação: Calculando Gradientes

O algoritmo de retropropagação calcula as derivadas parciais de $L$ em relação a cada parâmetro usando a regra da cadeia, partindo da saída e propagando os erros de volta.

### Regra de Atualização

Para atualizar parâmetros (ex: via gradiente descendente com taxa de aprendizado $\eta$):

$$p \leftarrow p - \eta \cdot \frac{\partial L}{\partial p}$$

### Passo 1: Erro da Camada de Saída

$$\sigma_y = \frac{\partial L}{\partial u} = \frac{2}{N}(y - \hat{y}) \cdot \hat{y}(1 - \hat{y})$$

### Passo 2: Gradientes para Pesos e Bias de Saída

$$\frac{\partial L}{\partial v_{11}} = \sigma_y \cdot h_1, \quad \frac{\partial L}{\partial v_{21}} = \sigma_y \cdot h_2, \quad \frac{\partial L}{\partial b^y_1} = \sigma_y$$

### Passo 3: Erros da Camada Oculta

$$\sigma_{h_1} = (\sigma_y \cdot v_{11}) \cdot h_1(1 - h_1), \quad \sigma_{h_2} = (\sigma_y \cdot v_{21}) \cdot h_2(1 - h_2)$$

### Passo 4: Gradientes para Pesos e Biases Ocultos

$$\frac{\partial L}{\partial w_{11}} = \sigma_{h_1} \cdot x_1, \quad \frac{\partial L}{\partial w_{21}} = \sigma_{h_1} \cdot x_2$$

$$\frac{\partial L}{\partial w_{12}} = \sigma_{h_2} \cdot x_1, \quad \frac{\partial L}{\partial w_{22}} = \sigma_{h_2} \cdot x_2$$

$$\frac{\partial L}{\partial b^h_1} = \sigma_{h_1}, \quad \frac{\partial L}{\partial b^h_2} = \sigma_{h_2}$$

### Passo 5: Atualizar Pesos e Biases

Usando os gradientes calculados e a taxa de aprendizado $\eta$:

$$\mathbf{W} \leftarrow \mathbf{W} - \eta \frac{\partial L}{\partial \mathbf{W}}, \quad \mathbf{V} \leftarrow \mathbf{V} - \eta \frac{\partial L}{\partial \mathbf{V}}$$

---

## Simulação Numérica

Com base na arquitetura MLP e nos passos de retropropagação descritos, podemos implementar uma simulação numérica simples para demonstrar o processo de treinamento.

### Inicialização

As matrizes de pesos e vetores de bias são inicializados aleatoriamente em $[0,1]$:

$$\mathbf{W} = \begin{bmatrix} 0.2 & 0.4 \\ 0.6 & 0.8 \end{bmatrix}, \quad \mathbf{b}^h = [0.1, 0.2]^T$$

$$\mathbf{V} = \begin{bmatrix} 0.3 & 0.5 \end{bmatrix}, \quad b^y = 0.4, \quad \eta = 0.7$$

### Passagem Direta

Para a amostra $\mathbf{x} = [0.5, 0.8]^T$, $y = 0$:

1. Pré-ativação oculta: $\mathbf{z} = [0.52, 1.14]^T$
2. Ativações ocultas: $\mathbf{h} \approx [0.627, 0.758]^T$
3. Pré-ativação de saída: $u \approx 0.967$
4. Ativação de saída: $\hat{y} \approx 0.725$

### Cálculo da Perda

$$L = (0 - 0.725)^2 \approx 0.5249$$

### Passagem Reversa

1. $\partial L / \partial u \approx -0.289$
2. Gradientes da camada oculta: $\approx [-0.020, -0.026]^T$
3. Pesos atualizados: $\mathbf{W} \leftarrow \begin{bmatrix} 0.207 & 0.409 \\ 0.611 & 0.815 \end{bmatrix}$

Repita o processo de treinamento para cada amostra ou múltiplas épocas.

- **Aprendizado online**: atualiza o modelo após cada exemplo de treinamento.
- **Aprendizado em batch**: atualiza o modelo após processar um batch de exemplos.

## Recursos Adicionais

Para uma compreensão mais intuitiva de redes neurais, recomendo a série de vídeos do 3Blue1Brown: [https://www.3blue1brown.com/lessons/neural-networks](https://www.3blue1brown.com/lessons/neural-networks){target="_blank"}

<iframe width="100%" height="470" src="https://www.youtube.com/embed/aircAruvnKk" title="But what is a neural network? | Deep learning chapter 1" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>


[^1]: Haykin, S. (1994). Neural Networks: A Comprehensive Foundation. Prentice Hall.
[^2]: Bishop, C. M. (2006). Pattern Recognition and Machine Learning. Springer.
[^3]: Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep Learning. MIT Press.
[:octicons-download-24:](https://www.deeplearningbook.org/){target="_blank"}

---

## Interativo: Visualizador de Passagem Direta do MLP

Observe as ativações se propagando por uma rede 2-entradas → 3-ocultos → 2-saídas. Ajuste os sliders de entrada e veja os valores fluírem camada por camada.

<div id="mlp-viz" style="background:#0d1117;border-radius:12px;padding:1.5rem;margin:2rem 0;font-family:Inter,sans-serif;">
<div style="display:flex;gap:1rem;flex-wrap:wrap;margin-bottom:1rem;">
  <div style="flex:1;min-width:160px;">
    <div style="color:#8b949e;font-size:.85rem;margin-bottom:.3rem;">Entrada x₁</div>
    <input id="mlp-x1" type="range" min="-2" max="2" step="0.05" value="0.8" style="width:100%;accent-color:#58a6ff;" oninput="mlpForward()">
    <div id="mlp-x1-val" style="color:#58a6ff;font-size:.8rem;font-family:monospace;"></div>
  </div>
  <div style="flex:1;min-width:160px;">
    <div style="color:#8b949e;font-size:.85rem;margin-bottom:.3rem;">Entrada x₂</div>
    <input id="mlp-x2" type="range" min="-2" max="2" step="0.05" value="-0.5" style="width:100%;accent-color:#58a6ff;" oninput="mlpForward()">
    <div id="mlp-x2-val" style="color:#58a6ff;font-size:.8rem;font-family:monospace;"></div>
  </div>
  <div style="flex:1;min-width:120px;">
    <div style="color:#8b949e;font-size:.85rem;margin-bottom:.3rem;">Ativação</div>
    <select id="mlp-act" onchange="mlpForward()" style="background:#161b22;color:#c9d1d9;border:1px solid #30363d;border-radius:4px;padding:4px 8px;width:100%;">
      <option value="relu">ReLU</option>
      <option value="sigmoid">Sigmoid</option>
      <option value="tanh">Tanh</option>
    </select>
  </div>
</div>
<canvas id="mlp-canvas" style="width:100%;display:block;border-radius:8px;"></canvas>
<div id="mlp-output" style="margin-top:.8rem;font-family:monospace;font-size:.82rem;color:#8b949e;background:#161b22;border-radius:6px;padding:.8rem;"></div>
</div>

<script>
(function(){
  const W1 = [[0.5,-0.3,0.8],[-0.4,0.7,0.2],[0.3,-0.6,-0.5],[0.1,0.4,-0.3]];
  const W2 = [[0.6,-0.4],[-0.5,0.8],[0.3,0.2],[0.2,-0.3]];

  function relu(x){return Math.max(0,x);}
  function sigmoid(x){return 1/(1+Math.exp(-x));}
  function tanh(x){return Math.tanh(x);}
  function activate(x, fn){ return fn==='relu'?relu(x):fn==='sigmoid'?sigmoid(x):tanh(x); }

  window.mlpForward = function() {
    const x1=+document.getElementById('mlp-x1').value;
    const x2=+document.getElementById('mlp-x2').value;
    const fn=document.getElementById('mlp-act').value;
    document.getElementById('mlp-x1-val').textContent='x₁ = '+x1.toFixed(2);
    document.getElementById('mlp-x2-val').textContent='x₂ = '+x2.toFixed(2);

    const inp=[x1,x2,1];
    const hidden_pre=[0,0,0];
    for(let j=0;j<3;j++) for(let i=0;i<3;i++) hidden_pre[j]+=W1[i][j]*(i<2?inp[i]:1);
    const hidden=hidden_pre.map(v=>activate(v,fn));

    const hbias=[...hidden,1];
    const out_pre=[0,0];
    for(let k=0;k<2;k++) for(let j=0;j<4;j++) out_pre[k]+=W2[j][k]*(j<3?hbias[j]:1);
    const out=out_pre.map(v=>sigmoid(v));

    drawMLP(x1,x2,hidden_pre,hidden,out_pre,out,fn);

    document.getElementById('mlp-output').innerHTML=
      '<span style="color:#484f58;">Pré-ativ. oculta: </span>'+hidden_pre.map(v=>'<span style="color:#d29922;">'+v.toFixed(3)+'</span>').join(', ')+
      '&nbsp;&nbsp;<span style="color:#484f58;">Pós-'+fn+': </span>'+hidden.map(v=>'<span style="color:#3fb950;">'+v.toFixed(3)+'</span>').join(', ')+
      '<br><span style="color:#484f58;">Pré-ativ. saída: </span>'+out_pre.map(v=>'<span style="color:#d29922;">'+v.toFixed(3)+'</span>').join(', ')+
      '&nbsp;&nbsp;<span style="color:#484f58;">Saída (sigmoid): </span>'+out.map(v=>'<span style="color:#f0883e;font-weight:bold;">'+v.toFixed(4)+'</span>').join(', ');
  };

  function neuronColor(v, maxAbs) {
    const t=Math.max(0,Math.min(1,(v+maxAbs)/(2*maxAbs)));
    const r=Math.round(40+t*215),g=Math.round(80+t*86*0.5),b=Math.round(200-t*160);
    return 'rgb('+r+','+g+','+b+')';
  }

  function drawMLP(x1,x2,hpre,hact,opre,oact,fn) {
    const canvas=document.getElementById('mlp-canvas');
    const ctx=canvas.getContext('2d');
    const W=canvas.parentElement.offsetWidth-48,H=240;
    canvas.width=W;canvas.height=H;canvas.style.height=H+'px';
    ctx.fillStyle='#161b22';ctx.fillRect(0,0,W,H);

    const layers=[
      {label:'Entrada',nodes:[x1,x2],y:H/2,color:'#58a6ff',x:W*0.1},
      {label:'Oculta ('+fn+')',nodes:hact,pre:hpre,y:H/2,color:'#3fb950',x:W*0.5},
      {label:'Saída (sigmoid)',nodes:oact,pre:opre,y:H/2,color:'#f0883e',x:W*0.9},
    ];
    const R=20;

    layers.forEach((l,li)=>{
      if(li===layers.length-1)return;
      const next=layers[li+1];
      l.nodes.forEach((v,i)=>{
        const y1=l.y+(i-(l.nodes.length-1)/2)*60;
        next.nodes.forEach((v2,j)=>{
          const y2=next.y+(j-(next.nodes.length-1)/2)*60;
          const alpha=Math.abs(v)*0.6+0.1;
          ctx.strokeStyle='rgba(255,255,255,'+alpha.toFixed(2)+')';
          ctx.lineWidth=1.2;
          ctx.beginPath();ctx.moveTo(l.x,y1);ctx.lineTo(next.x,y2);ctx.stroke();
        });
      });
    });

    layers.forEach(l=>{
      l.nodes.forEach((v,i)=>{
        const y=l.y+(i-(l.nodes.length-1)/2)*60;
        const maxV=l===layers[0]?2:1;
        const col=neuronColor(v,maxV);
        ctx.fillStyle=col+'44';
        ctx.beginPath();ctx.arc(l.x,y,R,0,2*Math.PI);ctx.fill();
        ctx.strokeStyle=l.color;ctx.lineWidth=2;
        ctx.beginPath();ctx.arc(l.x,y,R,0,2*Math.PI);ctx.stroke();
        ctx.fillStyle='#e6edf3';ctx.font='bold 10px monospace';ctx.textAlign='center';ctx.textBaseline='middle';
        ctx.fillText(v.toFixed(2),l.x,y);
        if(l.pre){
          ctx.fillStyle='#484f58';ctx.font='8px monospace';
          ctx.fillText('z='+l.pre[i].toFixed(2),l.x,y+R+10);
        }
      });
      ctx.fillStyle=l.color;ctx.font='bold 10px Inter,sans-serif';ctx.textAlign='center';
      ctx.fillText(l.label,l.x,H-12);
    });
  }

  window.mlpForward();
  window.addEventListener('resize',window.mlpForward);
})();
</script>

---

--8<-- "docs/2026.2/classes/mlp/quiz.pt.md"
