## Distribuições de Dados e Visualização

Antes de construir qualquer modelo, você deve **entender seus dados visualmente**. A distribuição das features informa qual pré-processamento é necessário, quais problemas esperar e se os dados podem suportar uma determinada tarefa.

---

## Por que a Distribuição Importa

O mesmo algoritmo aplicado à mesma tarefa pode falhar ou ter sucesso dependendo da distribuição dos dados. Um classificador linear funciona perfeitamente em dados linearmente separáveis, mas não consegue aprender XOR. A normalização é crítica para aprendizado baseado em gradiente, mas irrelevante para árvores de decisão.

---

## O Problema do Salmão vs Robalo

Um dataset introdutório clássico: classificar peixes em uma esteira como "salmão" ou "robalo" com base em dois sensores — tamanho (cm) e brilho (0–10).

$$
\mathbf{x} = \begin{bmatrix} x_1 \text{ (tamanho)} \\ x_2 \text{ (brilho)} \end{bmatrix} \longrightarrow f(\mathbf{x}) \in \{\text{salmão}, \text{robalo}\}
$$

```python exec="1" html="1"
--8<-- "docs/2026.2/classes/data/salmon_vs_seabass_1.py"
```
/// caption
Visão unidimensional: cada feature individualmente. Note que nem o tamanho sozinho nem o brilho sozinho separa perfeitamente as espécies.
///

```python exec="1" html="1"
--8<-- "docs/2026.2/classes/data/salmon_vs_seabass_2.py"
```
/// caption
Visão bidimensional: combinar ambas as features permite que uma fronteira de decisão linear separe a maioria das amostras.
///

!!! tip "Lição"
    Mais features = espaço de features mais rico = maior potencial de separação. Mas adicionar features irrelevantes pode prejudicar. **A seleção de features importa.**

---

## O Dataset Iris

[UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/53/iris){:target="_blank"}: introduzido por Ronald A. Fisher em 1936, este dataset de 150 amostras de três espécies de Íris é um benchmark fundamental de AM.

![Partes da flor Iris](../iris_dataset.png)

| Feature | Unidade | Intervalo |
|---------|------|-------|
| Comprimento da sépala | cm | 4,3–7,9 |
| Largura da sépala | cm | 2,0–4,4 |
| Comprimento da pétala | cm | 1,0–6,9 |
| Largura da pétala | cm | 0,1–2,5 |

```pyodide install="pandas,scikit-learn" exec="on" html="1"
--8<-- "docs/2026.2/classes/data/iris_data.py"
```

```python exec="1" html="1"
--8<-- "docs/2026.2/classes/data/iris_visualization.py"
```
/// caption
Pairplot do dataset Iris. Note: comprimento da pétala vs. largura da pétala separa claramente as três espécies. Comprimento da sépala vs. largura da sépala mostra sobreposição — nem todos os pares de features são igualmente discriminativos.
///

---

## Formas Comuns de Distribuição

```python exec="1" html="1"
--8<-- "docs/2026.2/classes/data/distributions.py"
```
/// caption
Quatro distribuições 2D comuns de dados. A fronteira de decisão que um modelo precisa aprender depende inteiramente de como os dados estão distribuídos.
///

| Distribuição | Características | Modelos adequados |
|---|---|---|
| **Linear** | Classes separadas por um hiperplano | Regressão Logística, SVM Linear, Perceptron |
| **Circular / radial** | Estrutura concêntrica não-linear | SVM com RBF, Redes Neurais, KNN |
| **Clusters** | Grupos em múltiplas localizações | GMM, Redes Neurais, CNN |
| **Espiral / complexa** | Altamente não-linear | Redes Neurais Profundas, SVM com kernel |

---

## Interativo: Explore uma Distribuição

Ajuste os parâmetros abaixo para ver como diferentes distribuições gaussianas se parecem e se sobrepõem.

<div id="dist-explorer" style="background:#0d1117;border-radius:12px;padding:1.5rem;margin:2rem 0;font-family:Inter,sans-serif;color:#e6edf3;">
<div style="display:grid;grid-template-columns:1fr 1fr;gap:1rem;margin-bottom:1rem;">
  <div>
    <div style="color:#58a6ff;font-weight:bold;font-size:.85rem;margin-bottom:.4rem;">Classe A (azul)</div>
    <div style="font-size:.8rem;color:#8b949e;">Média X: <input id="ax" type="range" min="-3" max="3" step="0.1" value="-1" style="accent-color:#58a6ff;" oninput="distDraw()"> <span id="ax-val"></span></div>
    <div style="font-size:.8rem;color:#8b949e;">Média Y: <input id="ay" type="range" min="-3" max="3" step="0.1" value="0" style="accent-color:#58a6ff;" oninput="distDraw()"> <span id="ay-val"></span></div>
    <div style="font-size:.8rem;color:#8b949e;">Desvio: <input id="as" type="range" min="0.3" max="2" step="0.1" value="0.8" style="accent-color:#58a6ff;" oninput="distDraw()"> <span id="as-val"></span></div>
  </div>
  <div>
    <div style="color:#f0883e;font-weight:bold;font-size:.85rem;margin-bottom:.4rem;">Classe B (laranja)</div>
    <div style="font-size:.8rem;color:#8b949e;">Média X: <input id="bx" type="range" min="-3" max="3" step="0.1" value="1" style="accent-color:#f0883e;" oninput="distDraw()"> <span id="bx-val"></span></div>
    <div style="font-size:.8rem;color:#8b949e;">Média Y: <input id="by" type="range" min="-3" max="3" step="0.1" value="0" style="accent-color:#f0883e;" oninput="distDraw()"> <span id="by-val"></span></div>
    <div style="font-size:.8rem;color:#8b949e;">Desvio: <input id="bs" type="range" min="0.3" max="2" step="0.1" value="0.8" style="accent-color:#f0883e;" oninput="distDraw()"> <span id="bs-val"></span></div>
  </div>
</div>
<canvas id="dist-canvas" style="width:100%;display:block;border-radius:8px;"></canvas>
<div id="dist-info" style="margin-top:.6rem;font-size:.8rem;color:#8b949e;font-family:monospace;"></div>
</div>

<script>
(function(){
  function randn(seed){ let x=Math.sin(seed*31.7)*43758.5; return x-Math.floor(x); }
  const N=120;
  const baseA=Array.from({length:N},(_,i)=>[randn(i*7)*2-1, randn(i*7+1)*2-1]);
  const baseB=Array.from({length:N},(_,i)=>[randn(i*7+1000)*2-1, randn(i*7+1001)*2-1]);

  function normalSample(base, mx, my, std){
    return base.map(([bx,by])=>{ return [mx+bx*std, my+by*std]; });
  }

  window.distDraw=function(){
    const ax=+document.getElementById('ax').value, ay=+document.getElementById('ay').value, as_=+document.getElementById('as').value;
    const bx=+document.getElementById('bx').value, by=+document.getElementById('by').value, bs=+document.getElementById('bs').value;
    ['ax','ay','as','bx','by','bs'].forEach(id=>{const el=document.getElementById(id+'-val');if(el)el.textContent=document.getElementById(id).value;});

    const sampA=normalSample(baseA,ax,ay,as_);
    const sampB=normalSample(baseB,bx,by,bs);

    const canvas=document.getElementById('dist-canvas');
    const ctx=canvas.getContext('2d');
    const W=canvas.parentElement.offsetWidth-48,H=220;
    canvas.width=W;canvas.height=H;canvas.style.height=H+'px';
    ctx.fillStyle='#161b22';ctx.fillRect(0,0,W,H);

    const scale=Math.min(W,H)/6.5;
    const cx=W/2,cy=H/2;

    ctx.strokeStyle='#21262d';ctx.lineWidth=0.5;
    for(let i=-5;i<=5;i++){ctx.beginPath();ctx.moveTo(cx+i*scale/2,0);ctx.lineTo(cx+i*scale/2,H);ctx.stroke();ctx.beginPath();ctx.moveTo(0,cy+i*scale/2);ctx.lineTo(W,cy+i*scale/2);ctx.stroke();}

    [[sampA,'#58a6ff'],[sampB,'#f0883e']].forEach(([pts,col])=>{
      pts.forEach(([px,py])=>{
        ctx.fillStyle=col+'aa';
        ctx.beginPath();ctx.arc(cx+px*scale,cy-py*scale,4,0,2*Math.PI);ctx.fill();
      });
    });

    const dist=Math.sqrt((ax-bx)**2+(ay-by)**2);
    const overlap=Math.max(0,1-dist/(as_+bs));
    const sep=dist>0?dist/(as_+bs):0;

    document.getElementById('dist-info').innerHTML=
      'Distância entre classes: <span style="color:#c9d1d9;">'+dist.toFixed(2)+'</span> &nbsp;|&nbsp; '+
      'Sobreposição estimada: <span style="color:'+(overlap>0.5?'#ff7b72':overlap>0.2?'#d29922':'#3fb950')+';">'+(overlap*100).toFixed(0)+'%</span> &nbsp;|&nbsp; '+
      'Separabilidade: <span style="color:'+(sep>1.5?'#3fb950':sep>0.8?'#d29922':'#ff7b72')+';">'+(sep>1.5?'Alta':sep>0.8?'Média':'Baixa')+'</span>';
  };
  distDraw();
  window.addEventListener('resize',distDraw);
})();
</script>

---

## Técnicas Principais de Visualização

| Técnica | Melhor para | Biblioteca |
|---|---|---|
| **Scatter plot** | Relações de features 2D | `matplotlib`, `seaborn` |
| **Pairplot** | Todos os pares de features de uma vez | `seaborn.pairplot` |
| **Histograma** | Distribuição de uma feature | `matplotlib.hist` |
| **Box plot** | Distribuição + outliers | `seaborn.boxplot` |
| **Heatmap (correlação)** | Correlações entre features | `seaborn.heatmap` |
| **t-SNE / UMAP** | Dados de alta dimensão em 2D | `sklearn`, `umap-learn` |
| **Violin plot** | Distribuição por classe | `seaborn.violinplot` |

```python
import seaborn as sns
import matplotlib.pyplot as plt

# Heatmap de correlação
corr = df.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', center=0)
plt.title('Matriz de Correlação de Features')
plt.show()

# t-SNE para dados de alta dimensão
from sklearn.manifold import TSNE
X_2d = TSNE(n_components=2, random_state=42).fit_transform(X_scaled)
plt.scatter(X_2d[:,0], X_2d[:,1], c=y, cmap='tab10')
plt.title('Visualização t-SNE')
plt.show()
```

[^1]: Fisher, R. A. (1936). [Iris](https://doi.org/10.24432/C56C76){:target="_blank"}. UCI Machine Learning Repository.
[^2]: Duda, R. O., Hart, P. E., & Stork, D. G. (2000). [Pattern Classification](https://dl.acm.org/doi/book/10.5555/954544){:target="_blank"}, 2ª Edição. Wiley.
