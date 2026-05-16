## Qualidade dos Dados

Dados do mundo real são bagunçados. Antes de treinar qualquer modelo, você deve entender a qualidade dos seus dados, identificar problemas e decidir como abordá-los. Má qualidade de dados frequentemente é mais difícil de corrigir do que uma má escolha de modelo.

> "Cientistas de dados passam 60–80% do tempo limpando e preparando dados." — Estimativa comum da indústria

---

## Valores Faltantes

Dados faltantes (NaN, NULL, None) são o problema de qualidade de dados mais comum. Mas **nem todos os dados faltantes são iguais** — o *mecanismo* do valor faltante determina a estratégia correta.

| Mecanismo | Definição | Exemplo | Estratégia |
|-----------|-----------|---------|---------|
| **MCAR** — Faltando Completamente ao Acaso | Faltando independentemente do valor | Falha aleatória do sensor | Qualquer imputação ou deleção é segura |
| **MAR** — Faltando ao Acaso | Faltando depende de outras variáveis observadas | Renda não reportada com mais frequência por pessoas mais jovens | Imputar usando outras features |
| **MNAR** — Faltando Não ao Acaso | Faltando depende do próprio valor faltante | Pessoas de alta renda menos propensas a reportar renda | Muito difícil; pode precisar de conhecimento de domínio |

### Detectando Valores Faltantes

```python
import pandas as pd

# Contar valores faltantes
print(df.isnull().sum())
print(df.isnull().mean() * 100)  # como percentual

# Heatmap de padrões faltantes
import seaborn as sns
import matplotlib.pyplot as plt
sns.heatmap(df.isnull(), cbar=False, yticklabels=False, cmap='viridis')
```

### Estratégias de Imputação

<div id="impute-viz" style="background:#0d1117;border-radius:12px;padding:1.5rem;margin:2rem 0;overflow:hidden;">
<canvas id="impute-canvas" style="width:100%;display:block;"></canvas>
<div style="display:flex;gap:.5rem;justify-content:center;flex-wrap:wrap;margin-top:.8rem;">
  <button onclick="imputeShow('original')" id="imp-orig" style="padding:4px 14px;background:#21262d;color:#c9d1d9;border:1px solid #30363d;border-radius:5px;cursor:pointer;font-size:.85rem;">Original</button>
  <button onclick="imputeShow('mean')" id="imp-mean" style="padding:4px 14px;background:#21262d;color:#c9d1d9;border:1px solid #30363d;border-radius:5px;cursor:pointer;font-size:.85rem;">Média</button>
  <button onclick="imputeShow('median')" id="imp-median" style="padding:4px 14px;background:#21262d;color:#c9d1d9;border:1px solid #30363d;border-radius:5px;cursor:pointer;font-size:.85rem;">Mediana</button>
  <button onclick="imputeShow('knn')" id="imp-knn" style="padding:4px 14px;background:#21262d;color:#c9d1d9;border:1px solid #30363d;border-radius:5px;cursor:pointer;font-size:.85rem;">KNN</button>
</div>
<div id="impute-desc" style="color:#8b949e;font-size:.82rem;text-align:center;margin-top:.6rem;"></div>
</div>

<script>
(function(){
  const rng=s=>{let x=Math.sin(s*31.7)*43758.5;return x-Math.floor(x);};
  const N=30;
  const allData=Array.from({length:N},(_,i)=>{
    const v=i<15?rng(i*7)*2+2:rng(i*7+500)*2+6;
    return v+(i===5?8:0);
  });
  const missing=[3,9,14,19,24];
  const observed=allData.filter((_,i)=>!missing.includes(i));
  const mean=observed.reduce((a,b)=>a+b,0)/observed.length;
  const sorted=[...observed].sort((a,b)=>a-b);
  const median=sorted[Math.floor(sorted.length/2)];

  function imputed(mode){
    return allData.map((v,i)=>{
      if(!missing.includes(i))return{v,imputed:false};
      if(mode==='mean')return{v:mean,imputed:true};
      if(mode==='median')return{v:median,imputed:true};
      if(mode==='knn'){
        const near=[];
        for(let j=i-2;j<=i+2;j++){if(j>=0&&j<N&&!missing.includes(j))near.push(allData[j]);}
        return{v:near.length?near.reduce((a,b)=>a+b,0)/near.length:mean,imputed:true};
      }
      return{v:null,imputed:true,missing:true};
    });
  }

  let mode='original';
  window.imputeShow=function(m){
    mode=m;
    ['orig','mean','median','knn'].forEach(id=>{
      const btn=document.getElementById('imp-'+id);
      btn.style.background='#21262d';btn.style.color='#c9d1d9';
    });
    const active=document.getElementById('imp-'+m.replace('original','orig'));
    if(active){active.style.background='#58a6ff';active.style.color='#0d1117';}
    draw();
  };

  const descs={
    original:'Valores faltantes mostrados como lacunas. As 5 posições faltantes (★) não podem ser usadas para treinamento.',
    mean:'Imputação por média: preencher com a média da coluna. Rápido, mas ignora a forma da distribuição; puxa valores para o centro.',
    median:'Imputação por mediana: preencher com a mediana da coluna. Mais robusta a outliers do que a média.',
    knn:'Imputação KNN: usar a média dos k vizinhos observados mais próximos. Preserva melhor a estrutura local.'
  };

  function draw(){
    const canvas=document.getElementById('impute-canvas');
    const ctx=canvas.getContext('2d');
    const W=canvas.parentElement.offsetWidth-48,H=150;
    canvas.width=W;canvas.height=H;canvas.style.height=H+'px';
    ctx.fillStyle='#161b22';ctx.fillRect(0,0,W,H);
    document.getElementById('impute-desc').textContent=descs[mode];

    const pts=imputed(mode);
    const maxV=Math.max(...allData)*1.1;
    const pad=20,pw=W-2*pad;
    const cx=(i)=>pad+i*(pw/(N-1)),cy=(v)=>H-20-v/maxV*(H-40);

    ctx.beginPath();ctx.strokeStyle='#30363d';ctx.lineWidth=1;
    let started=false;
    pts.forEach((p,i)=>{
      if(p.missing)return;
      const x=cx(i),y=cy(p.v);
      if(!started){ctx.moveTo(x,y);started=true;}else ctx.lineTo(x,y);
    });
    ctx.stroke();

    if(mode==='mean'||mode==='median'){
      const refV=mode==='mean'?mean:median;
      ctx.strokeStyle='#d29922aa';ctx.lineWidth=1;ctx.setLineDash([4,3]);
      ctx.beginPath();ctx.moveTo(pad,cy(refV));ctx.lineTo(W-pad,cy(refV));ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle='#d29922';ctx.font='8px monospace';ctx.textAlign='left';
      ctx.fillText((mode==='mean'?'média':'mediana')+'='+refV.toFixed(1),pad+4,cy(refV)-4);
    }

    pts.forEach((p,i)=>{
      const x=cx(i);
      if(p.missing){
        ctx.fillStyle='#ff7b72';ctx.font='12px serif';ctx.textAlign='center';
        ctx.fillText('★',x,H-12);
        return;
      }
      const y=cy(p.v);
      ctx.fillStyle=p.imputed?'#f0883e':'#58a6ff';
      ctx.beginPath();ctx.arc(x,y,p.imputed?6:4,0,2*Math.PI);ctx.fill();
    });

    ctx.fillStyle='#58a6ff';ctx.font='9px Inter,sans-serif';ctx.textAlign='left';ctx.fillText('● Observado',pad,12);
    if(mode!=='original'){ctx.fillStyle='#f0883e';ctx.fillText('● Imputado',pad+65,12);}
    ctx.fillStyle='#ff7b72';ctx.fillText('★ Faltante',pad+(mode!=='original'?130:65),12);
  }

  imputeShow('original');
  window.addEventListener('resize',draw);
})();
</script>

```python
from sklearn.impute import SimpleImputer, KNNImputer

# Imputação por média
imp = SimpleImputer(strategy='mean')
X_train = imp.fit_transform(X_train)
X_test  = imp.transform(X_test)  # use as estatísticas do treino!

# Imputação KNN (mais precisa, mais lenta)
imp = KNNImputer(n_neighbors=5)
X_train = imp.fit_transform(X_train)
X_test  = imp.transform(X_test)
```

---

## Outliers

Outliers são pontos de dados que se desviam significativamente do restante. Podem ser:

- **Genuínos**: observações extremas, mas válidas (ex: um bilionário em um dataset de renda)
- **Erros**: erros de medição, erros de entrada de dados

### Métodos de Detecção

```python
import numpy as np

# Método Z-score
z_scores = np.abs((df - df.mean()) / df.std())
outliers = (z_scores > 3).any(axis=1)

# Método IQR (mais robusto)
Q1, Q3 = df.quantile(0.25), df.quantile(0.75)
IQR = Q3 - Q1
outliers = ((df < Q1 - 1.5*IQR) | (df > Q3 + 1.5*IQR)).any(axis=1)

print(f"Outliers: {outliers.sum()} / {len(df)} ({outliers.mean()*100:.1f}%)")
```

### Estratégias de Tratamento

| Estratégia | Quando usar |
|----------|-------------|
| **Remover** | Erros claros de medição, pequena porcentagem |
| **Limitar/Winsorizar** | Manter valor, mas truncar para percentil |
| **Transformar** (log, raiz) | Dados assimétricos à direita com muitos outliers altos |
| **Manter** | Valores extremos genuínos relevantes para a tarefa |

---

## Duplicatas e Ruído

**Duplicatas** — linhas idênticas ou quase idênticas — inflam contagens de treinamento e podem causar overfitting:

```python
# Verificar duplicatas
print(f"Duplicatas: {df.duplicated().sum()}")
df = df.drop_duplicates()
```

**Ruído** — erros aleatórios nos valores de features ou rótulos — é mais difícil de detectar. Estratégias:

- Suavização de rótulos (alvos suaves em vez de 0/1 rígido)
- Aumento de dados
- Métodos ensemble (média sobre o ruído)
- Aprendizado confiante (identificar amostras provavelmente mal rotuladas)

---

## Checklist de Auditoria de Qualidade de Dados

```python
def relatorio_qualidade_dados(df, col_alvo=None):
    print("=== Relatório de Qualidade de Dados ===")
    print(f"Forma: {df.shape}")
    print(f"\nValores faltantes:")
    print(df.isnull().sum()[df.isnull().sum() > 0])
    print(f"\nDuplicatas: {df.duplicated().sum()}")
    print(f"\nTipos de dados:\n{df.dtypes}")
    if col_alvo:
        print(f"\nDistribuição de classes:\n{df[col_alvo].value_counts(normalize=True)}")
    print(f"\nResumo numérico:")
    print(df.describe())
```
