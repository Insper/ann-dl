## Train / Validation / Test Split

Splitting data correctly is not a technicality — it is the **experimental design** of machine learning. A wrong split gives you a broken experiment: results that look good in development but fail in production.

---

## The Three-Way Split

Each set has a distinct, non-overlapping role:

<div id="split-viz" style="background:#0d1117;border-radius:12px;padding:1.5rem;margin:2rem 0;">
<canvas id="split-canvas" style="width:100%;display:block;"></canvas>
<div style="margin-top:.8rem;">
  <label style="color:#8b949e;font-size:.85rem;">Train %: </label>
  <input id="split-train" type="range" min="50" max="85" step="5" value="70" style="accent-color:#3fb950;vertical-align:middle;" oninput="splitDraw()">
  <label style="color:#8b949e;font-size:.85rem;margin-left:1rem;">Val %: </label>
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
  document.getElementById('split-info').textContent = 'Train:' + tr + '% | Val:' + va + '% | Test:' + te + '%';

  const pad = 10, barH = 46, barY = (H - barH) / 2;
  const bw = W - 2*pad;
  const trW = bw * tr/100, vaW = bw * va/100, teW = bw * te/100;

  const segments = [
    { x: pad, w: trW, color: '#3fb950', label: 'Training Set', sub: tr + '% — Adjust weights', role: 'Model learns patterns here' },
    { x: pad+trW, w: vaW, color: '#58a6ff', label: 'Validation Set', sub: va + '% — Tune hyperparams', role: 'Select best model/epoch' },
    { x: pad+trW+vaW, w: teW, color: '#f0883e', label: 'Test Set', sub: te + '% — Final score', role: 'Report this number ONCE' },
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

  // Roles below bar
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

| Set | Used for | How many times? |
|-----|---------|----------------|
| **Train** | Fitting model weights | Many iterations |
| **Validation** | Tuning hyperparameters, early stopping, model selection | Many times |
| **Test** | Reporting final performance | **Exactly once** |

!!! warning "The Golden Rule"
    If you look at test set performance and then change anything in your model, the test set is **no longer a valid measure of generalization**. You have been tuning on it implicitly.

---

## Cross-Validation

When data is scarce, a single train/val split wastes data. **K-Fold Cross-Validation** uses all data for training and validation:

```
K=5 folds:
Fold 1: [VAL][TRN][TRN][TRN][TRN]
Fold 2: [TRN][VAL][TRN][TRN][TRN]
Fold 3: [TRN][TRN][VAL][TRN][TRN]
Fold 4: [TRN][TRN][TRN][VAL][TRN]
Fold 5: [TRN][TRN][TRN][TRN][VAL]
```

Final metric = mean ± std across folds. Much more reliable than a single split.

```python
from sklearn.model_selection import cross_val_score, KFold

kf = KFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=kf, scoring='f1_macro')
print(f"F1: {scores.mean():.3f} ± {scores.std():.3f}")
```

**Note:** Even with cross-validation, keep a **held-out test set** that is never used during cross-validation.

---

## Stratified Split

For **classification**, a random split may produce very different class distributions across folds — especially with class imbalance. **Stratified split** preserves the class proportion in each fold.

```python
from sklearn.model_selection import StratifiedKFold, train_test_split

# Stratified train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Stratified k-fold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
```

Always use `stratify=y` when working with classification tasks.

---

## Temporal Split for Time Series

Random splits are incorrect for time series — they allow future information to leak into training. Always split by time:

```python
# For a time-indexed DataFrame
split_date = '2024-01-01'
train = df[df.index < split_date]
test  = df[df.index >= split_date]
```

For cross-validation of time series, use **TimeSeriesSplit**:

```python
from sklearn.model_selection import TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5)
# Each fold: train on past, validate on immediate future
```

---

## How Much Data for Each Set?

| Dataset size | Recommended split | Rationale |
|---|---|---|
| Small (< 10k) | 60/20/20 or use CV | More validation/test data for reliable estimates |
| Medium (10k–1M) | 70/15/15 or 80/10/10 | More training data improves model |
| Large (> 1M) | 98/1/1 | 1% of 1M = 10k samples, plenty for evaluation |
| Very large (> 100M) | 99/0.5/0.5 | Even 0.5% gives 500k evaluation samples |

For large datasets, the test set needs to be large enough to give **statistically reliable estimates**, not a fixed percentage.
