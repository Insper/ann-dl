## Vazamento de Dados

O vazamento de dados é **o destruidor silencioso de modelos**. Ocorre quando informações de fora do conjunto de treinamento são inadvertidamente usadas para criar o modelo, fazendo-o parecer muito melhor do que realmente é. Modelos com vazamento podem alcançar acurácia de validação quase perfeita, passar em todos os testes e falhar completamente em produção.

!!! danger "O Vazamento de Dados é a principal causa de resultados de AM irreproduziveis"
    O vazamento pode ser sutil o suficiente para que praticantes experientes o ignorem. É responsável por uma parcela significativa de artigos de aprendizado de máquina retratados e implantações em produção que falharam.

---

## Tipos de Vazamento de Dados

### 1. Vazamento de Alvo

Usar features que são **causalmente causadas pelo alvo**, não causas dele.

**Exemplo:** Prever se um paciente receberá prescrição do antibiótico X.

| Feature | Vazamento? | Motivo |
|---------|:--------:|--------|
| Idade, pressão arterial | ✅ Não | Existem antes do diagnóstico |
| Flag `tomou_antibiotico_x` | ❌ **SIM** | Causada pela prescrição |
| `data_visita_farmacia` | ❌ **SIM** | Acontece após a prescrição |
| `pontuacao_recomendacao_medico` | ❌ **SIM** | Faz parte da decisão |

O modelo aprende "se `tomou_antibiotico_x = True`, então prescrição = True" — uma regra circular e inútil.

**Prevenção:** Para cada feature, pergunte: **"Este valor existe no momento da previsão?"** Se a resposta for "às vezes" ou "depende", trate com suspeição.

---

### 2. Contaminação Treino-Teste

Permitir que **informações do conjunto de teste influenciem o processo de treinamento**.

A forma mais comum: ajustar um pré-processador (scaler, imputer, encoder) em todo o dataset antes da divisão.

<div id="leak-viz" style="background:#0d1117;border-radius:12px;padding:1.5rem;margin:2rem 0;font-family:Inter,sans-serif;">
<canvas id="leak-canvas" style="width:100%;display:block;"></canvas>
<div style="display:flex;gap:.5rem;justify-content:center;margin-top:1rem;">
  <button onclick="leakShow('wrong')" id="btn-wrong" style="padding:6px 20px;background:#ff7b72;color:#0d1117;border:none;border-radius:5px;cursor:pointer;font-weight:bold;">Pipeline Errado</button>
  <button onclick="leakShow('right')" id="btn-right" style="padding:6px 20px;background:#21262d;color:#c9d1d9;border:1px solid #30363d;border-radius:5px;cursor:pointer;">Pipeline Correto</button>
</div>
<div id="leak-desc" style="color:#c9d1d9;font-size:.85rem;text-align:center;margin-top:.8rem;min-height:2.5rem;padding:0 1rem;"></div>
</div>

<script>
(function(){
  const canvas = document.getElementById('leak-canvas');
  const ctx = canvas.getContext('2d');
  let mode = 'wrong';

  window.leakShow = function(m) {
    mode = m;
    document.getElementById('btn-wrong').style.background = m === 'wrong' ? '#ff7b72' : '#21262d';
    document.getElementById('btn-wrong').style.color = m === 'wrong' ? '#0d1117' : '#c9d1d9';
    document.getElementById('btn-right').style.background = m === 'right' ? '#3fb950' : '#21262d';
    document.getElementById('btn-right').style.color = m === 'right' ? '#0d1117' : '#c9d1d9';
    draw();
  };

  const descs = {
    wrong: '❌ ERRADO: O scaler é ajustado em TODOS os dados (treino + teste). O modelo indiretamente vê as estatísticas do teste durante o treinamento → métricas de validação infladas que não se reproduzem em produção.',
    right: '✅ CORRETO: O scaler é ajustado APENAS nos dados de treino. Os dados de teste são transformados usando estatísticas de treino, como se fossem dados verdadeiramente não vistos. As métricas refletem a generalização real.'
  };

  function box(x, y, w, h, color, text, sub) {
    ctx.fillStyle = color + '22'; ctx.beginPath(); ctx.roundRect(x,y,w,h,6); ctx.fill();
    ctx.strokeStyle = color; ctx.lineWidth = 1.5; ctx.beginPath(); ctx.roundRect(x,y,w,h,6); ctx.stroke();
    ctx.fillStyle = color; ctx.font = 'bold 10px Inter,sans-serif'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    ctx.fillText(text, x+w/2, y+h/2 - (sub?5:0));
    if (sub) { ctx.font = '8px monospace'; ctx.fillStyle = color+'aa'; ctx.fillText(sub, x+w/2, y+h/2+7); }
  }

  function arrow(x1,y1,x2,y2,color,label) {
    ctx.strokeStyle = color; ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.moveTo(x1,y1); ctx.lineTo(x2,y2); ctx.stroke();
    const a = Math.atan2(y2-y1,x2-x1);
    ctx.fillStyle = color; ctx.beginPath(); ctx.moveTo(x2,y2); ctx.lineTo(x2-7*Math.cos(a-0.4),y2-7*Math.sin(a-0.4)); ctx.lineTo(x2-7*Math.cos(a+0.4),y2-7*Math.sin(a+0.4)); ctx.fill();
    if (label) { ctx.font = '8px Inter,sans-serif'; ctx.textAlign = 'center'; ctx.fillText(label, (x1+x2)/2, (y1+y2)/2 - 6); }
  }

  function draw() {
    const W = canvas.parentElement.offsetWidth - 48, H = 200;
    canvas.width = W; canvas.height = H; canvas.style.height = H + 'px';
    ctx.fillStyle = '#0d1117'; ctx.fillRect(0,0,W,H);
    document.getElementById('leak-desc').textContent = descs[mode];

    const bw = Math.min(110, (W-60)/5), bh = 44, mid = H/2;

    if (mode === 'wrong') {
      box(10, mid-bh/2, bw, bh, '#ff7b72', 'Todos os Dados', 'treino+teste');
      arrow(10+bw, mid, 10+bw+20, mid, '#ff7b72');
      box(10+bw+20, mid-bh/2, bw, bh, '#ff7b72', 'Scaler.fit()', '⚠️ vê o teste!');
      arrow(10+bw*2+40, mid, 10+bw*2+60, mid, '#ff7b72');
      box(10+bw*2+60, mid-bh*0.8, bw*0.9, bh*0.8, '#58a6ff', 'Treino', '.transform');
      box(10+bw*2+60, mid+bh*0.1, bw*0.9, bh*0.8, '#f0883e', 'Teste', '.transform');
      arrow(10+bw*3+65, mid-bh*0.4, 10+bw*3+85, mid-bh*0.4, '#58a6ff');
      box(10+bw*3+85, mid-bh/2, bw, bh, '#3fb950', 'Modelo', 'treinamento');
      ctx.strokeStyle = '#ff7b72'; ctx.lineWidth = 1.5; ctx.setLineDash([4,3]);
      ctx.beginPath(); ctx.moveTo(10+bw*2+60+bw*0.9/2, mid+bh*0.1); ctx.lineTo(10+bw+20+bw/2, mid+bh/2+12); ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = '#ff7b72'; ctx.font = '8px Inter,sans-serif'; ctx.textAlign = 'center';
      ctx.fillText('vaza stats do teste', 10+bw*1.7+30, mid+bh/2+22);
    } else {
      box(10, mid-bh-8, bw, bh, '#3fb950', 'Dados de Treino', '80%');
      box(10, mid+8, bw, bh, '#58a6ff', 'Dados de Teste', '20%');
      arrow(10+bw, mid-bh/2+4, 10+bw+20, mid-bh/2+4, '#3fb950', 'fit');
      box(10+bw+20, mid-bh*0.8, bw, bh*0.8, '#3fb950', 'Scaler', 'só stats treino');
      arrow(10+bw*2+20, mid-bh*0.4, 10+bw*2+40, mid-bh*0.4, '#3fb950', 'transform');
      box(10+bw*2+40, mid-bh, bw, bh, '#3fb950', 'Modelo', 'treinamento');
      ctx.strokeStyle = '#58a6ff'; ctx.lineWidth = 1.5; ctx.setLineDash([4,3]);
      ctx.beginPath(); ctx.moveTo(10+bw, mid+bh/2+8); ctx.lineTo(10+bw+20+bw/2, mid+bh*0.5); ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = '#58a6ff'; ctx.font = '8px Inter,sans-serif'; ctx.textAlign = 'center';
      ctx.fillText('só .transform', 10+bw*1.5+20, mid+bh*0.7+8);
      arrow(10+bw*2+20, mid+bh*0.1, 10+bw*2+40, mid+bh*0.1, '#58a6ff', 'avaliar');
      box(10+bw*2+40, mid+bh*0.1-bh*0.4, bw, bh*0.8, '#58a6ff', 'Avaliar', 'métricas reais');
      ctx.fillStyle = '#3fb950'; ctx.font = '20px serif'; ctx.textAlign = 'center';
      ctx.fillText('✓', W-24, mid);
    }
  }

  draw(); window.addEventListener('resize', draw);
})();
</script>

**Código errado:**
```python
# ❌ VAZAMENTO — scaler vê os dados de teste
scaler = StandardScaler()
X_all_scaled = scaler.fit_transform(X_all)          # usa TODOS os dados
X_train, X_test = train_test_split(X_all_scaled)
```

**Código correto:**
```python
# ✅ CORRETO — scaler vê apenas os dados de treino
X_train, X_test = train_test_split(X_all)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)             # fit + transform apenas no treino
X_test  = scaler.transform(X_test)                  # apenas transform no teste
```

**Usando Pipelines (recomendado):**
```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression())
])
pipe.fit(X_train, y_train)     # scaler.fit é chamado apenas em X_train
pipe.score(X_test, y_test)     # scaler.transform é chamado em X_test
```

Pipelines do Scikit-learn são a forma idiomática de prevenir contaminação treino-teste.

---

### 3. Vazamento Temporal

Em problemas de **séries temporais**, usar informações futuras para prever o passado.

```
         Passado                  Futuro
── [x₁, x₂, x₃] ──prever──► [x₄] ──────────►
                               ↑
                    NÃO deve ver x₄ durante treino para x₃!
```

**Errado:** Divisão aleatória em um dataset de série temporal.

**Correto:** Sempre use uma **divisão temporal** — todos os dados de treino vêm estritamente antes do período de validação/teste.

```python
# ❌ Errado para séries temporais
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ Correto para séries temporais
split_idx = int(len(X) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]
```

---

### 4. Vazamento na Engenharia de Features

Calcular features que agregam informações do dataset completo — incluindo as linhas de teste.

```python
# ❌ VAZAMENTO: estatísticas de grupo calculadas em todo o dataset
df['media_gasto_usuario'] = df.groupby('user_id')['gasto'].transform('mean')

# ✅ CORRETO: calcular no conjunto de treino, mesclar no teste
medias_treino = X_train.groupby('user_id')['gasto'].mean().rename('media_gasto_usuario')
X_train = X_train.merge(medias_treino, on='user_id', how='left')
X_test  = X_test.merge(medias_treino, on='user_id', how='left')   # usa stats do treino
```

---

## Checklist para Detecção de Vazamento

<div style="background:#0d1117;border-radius:8px;padding:1.2rem;margin:1.5rem 0;border-left:4px solid #ff7b72;">

**🔍 Desempenho suspeitamente alto?**

Se seu modelo atinge >95% de acurácia em um problema difícil, suspeite de vazamento antes de comemorar.

**Checklist:**
- [ ] Cada feature existe no **momento da previsão** em produção?
- [ ] O **scaler/imputer é ajustado apenas nos dados de treino**?
- [ ] Para séries temporais: a divisão é **estritamente temporal**?
- [ ] Alguma feature **correlaciona perfeitamente** (>0,99) com o alvo?
- [ ] Há **timestamps futuros** em features "passadas"?
- [ ] Você calculou **agregações de grupo** em todo o dataset?
- [ ] O desempenho é **bom demais** na validação, mas ruim quando implantado?

</div>

---

## Um Exemplo Real Famoso: o Heritage Health Prize

O Heritage Health Prize de 2011 (competição de $3M) teve várias equipes top desqualificadas por vazamento. Uma equipe atingiu AUC=0,98 na validação — muito acima do baseline humano — ao usar acidentalmente uma feature derivada da variável alvo. Quando o erro foi descoberto e a feature removida, o desempenho caiu para AUC=0,76.

A lição: **resultados extraordinários exigem escrutínio extraordinário dos dados**.
