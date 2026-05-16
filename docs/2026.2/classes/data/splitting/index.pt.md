## Divisão Treino / Validação / Teste

Dividir os dados corretamente não é uma tecnicidade — é o **design experimental** do aprendizado de máquina. Uma divisão errada produz um experimento quebrado: resultados que parecem bons durante o desenvolvimento, mas falham em produção.

---

## A Divisão em Três Conjuntos

Cada conjunto tem um papel distinto e não sobreponível:

<div id="split-viz" style="background:#0d1117;border-radius:12px;padding:1.5rem;margin:2rem 0;">
<canvas id="split-canvas" style="width:100%;display:block;"></canvas>
<div style="margin-top:.8rem;">
  <label style="color:#8b949e;font-size:.85rem;">% Treino: </label>
  <input id="split-train" type="range" min="50" max="85" step="5" value="70" style="accent-color:#3fb950;vertical-align:middle;" oninput="splitDraw()">
  <label style="color:#8b949e;font-size:.85rem;margin-left:1rem;">% Val: </label>
  <input id="split-val" type="range" min="5" max="30" step="5" value="15" style="accent-color:#58a6ff;vertical-align:middle;" oninput="splitDraw()">
  <span id="split-info" style="color:#8b949e;font-size:.8rem;font-family:monospace;margin-left:1rem;"></span>
</div>
</div>

<script>
window.splitDraw = function() {
  const canvas = document.getElementById('split-canvas');
  const ctx = canvas.getContext('2d');
  const W = canvas.parentElement.offsetWidth - 48, H = 120;
  canvas.width = W; canvas.height = H; canvas.style.height = H + 'px';
  ctx.fillStyle = '#0d1117'; ctx.fillRect(0,0,W,H);

  const tr = +document.getElementById('split-train').value;
  const va = +document.getElementById('split-val').value;
  const te = 100 - tr - va;
  document.getElementById('split-info').textContent = 'Treino:' + tr + '% | Val:' + va + '% | Teste:' + te + '%';

  const pad = 10, barH = 46, barY = (H - barH) / 2;
  const bw = W - 2*pad;
  const trW = bw * tr/100, vaW = bw * va/100, teW = bw * te/100;

  const segments = [
    { x: pad, w: trW, color: '#3fb950', label: 'Conjunto de Treino', sub: tr + '% — Ajusta pesos', role: 'Modelo aprende padrões aqui' },
    { x: pad+trW, w: vaW, color: '#58a6ff', label: 'Conjunto de Validação', sub: va + '% — Ajusta hiperparams', role: 'Seleciona o melhor modelo/época' },
    { x: pad+trW+vaW, w: teW, color: '#f0883e', label: 'Conjunto de Teste', sub: te + '% — Nota final', role: 'Reporte este número UMA VEZ' },
  ];

  segments.forEach((s, i) => {
    ctx.fillStyle = s.color + '33';
    ctx.fillRect(s.x + (i>0?1:0), barY, s.w - (i<2?1:0), barH);
    ctx.strokeStyle = s.color; ctx.lineWidth = 2;
    ctx.strokeRect(s.x + (i>0?1:0), barY, s.w - (i<2?1:0), barH);

    if (s.w > 60) {
      ctx.fillStyle = s.color; ctx.font = 'bold 10px Inter,sans-serif'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
      ctx.fillText(s.label, s.x + s.w/2, barY + barH*0.36);
      ctx.font = '8px monospace'; ctx.fillStyle = s.color + 'aa';
      ctx.fillText(s.sub, s.x + s.w/2, barY + barH*0.64);
    }
  });

  segments.forEach((s, i) => {
    if (s.w > 50) {
      ctx.fillStyle = s.color + '88'; ctx.font = '8px Inter,sans-serif'; ctx.textAlign = 'center';
      ctx.fillText(s.role, s.x + s.w/2, barY + barH + 14);
    }
  });
};
splitDraw();
window.addEventListener('resize', splitDraw);
</script>

| Conjunto | Usado para | Quantas vezes? |
|-----|---------|----------------|
| **Treino** | Ajustar pesos do modelo | Muitas iterações |
| **Validação** | Ajustar hiperparâmetros, early stopping, seleção de modelo | Muitas vezes |
| **Teste** | Reportar desempenho final | **Exatamente uma vez** |

!!! warning "A Regra de Ouro"
    Se você olhar o desempenho no conjunto de teste e depois mudar qualquer coisa no seu modelo, o conjunto de teste **não é mais uma medida válida de generalização**. Você passou a ajustar implicitamente com base nele.

---

## Validação Cruzada

Quando os dados são escassos, uma única divisão treino/val desperdiça dados. **Validação Cruzada K-Fold** usa todos os dados para treinamento e validação:

```
K=5 folds:
Fold 1: [VAL][TRN][TRN][TRN][TRN]
Fold 2: [TRN][VAL][TRN][TRN][TRN]
Fold 3: [TRN][TRN][VAL][TRN][TRN]
Fold 4: [TRN][TRN][TRN][VAL][TRN]
Fold 5: [TRN][TRN][TRN][TRN][VAL]
```

Métrica final = média ± desvio padrão entre os folds. Muito mais confiável do que uma única divisão.

```python
from sklearn.model_selection import cross_val_score, KFold

kf = KFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=kf, scoring='f1_macro')
print(f"F1: {scores.mean():.3f} ± {scores.std():.3f}")
```

**Nota:** Mesmo com validação cruzada, mantenha um **conjunto de teste separado** que nunca é usado durante a validação cruzada.

---

## Divisão Estratificada

Para **classificação**, uma divisão aleatória pode produzir distribuições de classe muito diferentes entre os folds — especialmente com desbalanceamento de classes. A **divisão estratificada** preserva a proporção de classes em cada fold.

```python
from sklearn.model_selection import StratifiedKFold, train_test_split

# Divisão estratificada treino/teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# K-fold estratificado
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
```

Sempre use `stratify=y` ao trabalhar com tarefas de classificação.

---

## Divisão Temporal para Séries Temporais

Divisões aleatórias são incorretas para séries temporais — permitem que informações futuras vazem para o treinamento. Sempre divida por tempo:

```python
# Para um DataFrame com índice de tempo
split_date = '2024-01-01'
train = df[df.index < split_date]
test  = df[df.index >= split_date]
```

Para validação cruzada de séries temporais, use **TimeSeriesSplit**:

```python
from sklearn.model_selection import TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5)
# Cada fold: treina no passado, valida no futuro imediato
```

---

## Quanto de Dados para Cada Conjunto?

| Tamanho do dataset | Divisão recomendada | Justificativa |
|---|---|---|
| Pequeno (< 10k) | 60/20/20 ou use VC | Mais dados de validação/teste para estimativas confiáveis |
| Médio (10k–1M) | 70/15/15 ou 80/10/10 | Mais dados de treino melhora o modelo |
| Grande (> 1M) | 98/1/1 | 1% de 1M = 10k amostras, suficiente para avaliação |
| Muito grande (> 100M) | 99/0,5/0,5 | Mesmo 0,5% resulta em 500k amostras de avaliação |
