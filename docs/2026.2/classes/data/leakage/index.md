## Data Leakage

Data leakage is **the silent model-killer**. It occurs when information from outside the training set is inadvertently used to create the model, causing it to appear far better than it actually is. Models with leakage can achieve near-perfect validation accuracy, pass all tests, and then fail completely in production.

!!! danger "Data Leakage is the #1 cause of unreproducible ML results"
    Leakage can be subtle enough that experienced practitioners miss it. It is responsible for a significant portion of retracted machine learning papers and failed production deployments.

---

## Types of Data Leakage

### 1. Target Leakage

Using features that are **causally caused by the target**, not causes of it.

**Example:** Predicting whether a patient will be prescribed antibiotic X.

| Feature | Leakage? | Reason |
|---------|:--------:|--------|
| Age, blood pressure | ✅ No | Exist before diagnosis |
| `took_antibiotic_x` flag | ❌ **YES** | Caused by the prescription |
| `pharmacy_visit_date` | ❌ **YES** | Happens after prescription |
| `doctor_recommendation_score` | ❌ **YES** | Part of the decision |

The model learns "if `took_antibiotic_x = True`, then prescription = True" — a circular, useless rule.

**Prevention:** For each feature, ask: **"Does this value exist at prediction time?"** If the answer is "sometimes" or "it depends", treat it with suspicion.

---

### 2. Train-Test Contamination

Allowing **information from the test set to influence the training process**.

The most common form: fitting a preprocessor (scaler, imputer, encoder) on the full dataset before splitting.

<div id="leak-viz" style="background:#0d1117;border-radius:12px;padding:1.5rem;margin:2rem 0;font-family:Inter,sans-serif;">
<canvas id="leak-canvas" style="width:100%;display:block;"></canvas>
<div style="display:flex;gap:.5rem;justify-content:center;margin-top:1rem;">
  <button onclick="leakShow('wrong')" id="btn-wrong" style="padding:6px 20px;background:#ff7b72;color:#0d1117;border:none;border-radius:5px;cursor:pointer;font-weight:bold;">Wrong Pipeline</button>
  <button onclick="leakShow('right')" id="btn-right" style="padding:6px 20px;background:#21262d;color:#c9d1d9;border:1px solid #30363d;border-radius:5px;cursor:pointer;">Correct Pipeline</button>
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
    wrong: '❌ WRONG: Scaler is fit on ALL data (train + test). The model indirectly sees test statistics during training → inflated validation metrics that will not reproduce in production.',
    right: '✅ CORRECT: Scaler is fit ONLY on train data. Test data is transformed using train statistics, as if it were truly unseen data. Metrics reflect real generalization.'
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
      // All data → Scaler.fit → split → train → model // test → scaler.transform → evaluate
      box(10, mid-bh/2, bw, bh, '#ff7b72', 'All Data', 'train+test');
      arrow(10+bw, mid, 10+bw+20, mid, '#ff7b72');
      box(10+bw+20, mid-bh/2, bw, bh, '#ff7b72', 'Scaler.fit()', '⚠️ sees test!');
      arrow(10+bw*2+40, mid, 10+bw*2+60, mid, '#ff7b72');
      box(10+bw*2+60, mid-bh*0.8, bw*0.9, bh*0.8, '#58a6ff', 'Train', '.transform');
      box(10+bw*2+60, mid+bh*0.1, bw*0.9, bh*0.8, '#f0883e', 'Test', '.transform');
      arrow(10+bw*3+65, mid-bh*0.4, 10+bw*3+85, mid-bh*0.4, '#58a6ff');
      box(10+bw*3+85, mid-bh/2, bw, bh, '#3fb950', 'Model', 'training');
      // Contamination indicator
      ctx.strokeStyle = '#ff7b72'; ctx.lineWidth = 1.5; ctx.setLineDash([4,3]);
      ctx.beginPath(); ctx.moveTo(10+bw*2+60+bw*0.9/2, mid+bh*0.1); ctx.lineTo(10+bw+20+bw/2, mid+bh/2+12); ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = '#ff7b72'; ctx.font = '8px Inter,sans-serif'; ctx.textAlign = 'center';
      ctx.fillText('leaks test stats', 10+bw*1.7+30, mid+bh/2+22);
    } else {
      // Train → scaler.fit → model; Test → scaler.transform (using train stats) → evaluate
      box(10, mid-bh-8, bw, bh, '#3fb950', 'Train Data', '80%');
      box(10, mid+8, bw, bh, '#58a6ff', 'Test Data', '20%');
      arrow(10+bw, mid-bh/2+4, 10+bw+20, mid-bh/2+4, '#3fb950', 'fit');
      box(10+bw+20, mid-bh*0.8, bw, bh*0.8, '#3fb950', 'Scaler', 'train stats only');
      arrow(10+bw*2+20, mid-bh*0.4, 10+bw*2+40, mid-bh*0.4, '#3fb950', 'transform');
      box(10+bw*2+40, mid-bh, bw, bh, '#3fb950', 'Model', 'training');
      // Test path
      ctx.strokeStyle = '#58a6ff'; ctx.lineWidth = 1.5; ctx.setLineDash([4,3]);
      ctx.beginPath(); ctx.moveTo(10+bw, mid+bh/2+8); ctx.lineTo(10+bw+20+bw/2, mid+bh*0.5); ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = '#58a6ff'; ctx.font = '8px Inter,sans-serif'; ctx.textAlign = 'center';
      ctx.fillText('.transform only', 10+bw*1.5+20, mid+bh*0.7+8);
      arrow(10+bw*2+20, mid+bh*0.1, 10+bw*2+40, mid+bh*0.1, '#58a6ff', 'evaluate');
      box(10+bw*2+40, mid+bh*0.1-bh*0.4, bw, bh*0.8, '#58a6ff', 'Evaluate', 'real metrics');
      // Checkmark
      ctx.fillStyle = '#3fb950'; ctx.font = '20px serif'; ctx.textAlign = 'center';
      ctx.fillText('✓', W-24, mid);
    }
  }

  draw(); window.addEventListener('resize', draw);
})();
</script>

**Wrong code:**
```python
# ❌ LEAKAGE — scaler sees test data
scaler = StandardScaler()
X_all_scaled = scaler.fit_transform(X_all)          # uses ALL data
X_train, X_test = train_test_split(X_all_scaled)
```

**Correct code:**
```python
# ✅ CORRECT — scaler only sees training data
X_train, X_test = train_test_split(X_all)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)             # fit + transform on train only
X_test  = scaler.transform(X_test)                  # transform only on test
```

**Using Pipelines (recommended):**
```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression())
])
pipe.fit(X_train, y_train)     # scaler.fit is only called on X_train
pipe.score(X_test, y_test)     # scaler.transform is called on X_test
```

Scikit-learn Pipelines are the idiomatic way to prevent train-test contamination.

---

### 3. Temporal Leakage

In **time series** problems, using future information to predict the past.

```
         Past                    Future
── [x₁, x₂, x₃] ──predict──► [x₄] ──────────►
                               ↑
                    Must NOT see x₄ during training for x₃!
```

**Wrong:** Random split on a time series dataset. Sample at time t=100 ends up in training, but samples at t=95–99 (future-relative to t=90) are in the same training set.

**Correct:** Always use a **temporal split** — all training data comes strictly before the validation/test period.

```python
# ❌ Wrong for time series
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ Correct for time series
split_idx = int(len(X) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]
```

---

### 4. Feature Engineering Leakage

Computing features that aggregate information from the full dataset — including test rows.

```python
# ❌ LEAKAGE: group statistics computed on full dataset
df['user_avg_spend'] = df.groupby('user_id')['spend'].transform('mean')

# ✅ CORRECT: compute on training set, merge into test
train_means = X_train.groupby('user_id')['spend'].mean().rename('user_avg_spend')
X_train = X_train.merge(train_means, on='user_id', how='left')
X_test  = X_test.merge(train_means, on='user_id', how='left')   # uses train stats
```

---

## Leakage Detection Checklist

<div style="background:#0d1117;border-radius:8px;padding:1.2rem;margin:1.5rem 0;border-left:4px solid #ff7b72;">

**🔍 Suspiciously high performance?**

If your model achieves >95% accuracy on a hard problem, suspect leakage before celebrating.

**Checklist:**
- [ ] Does each feature exist at **prediction time** in production?
- [ ] Is the **scaler/imputer fit only on train data**?
- [ ] For time series: is the split **strictly temporal**?
- [ ] Do any features **correlate perfectly** (>0.99) with the target?
- [ ] Are there any **future timestamps** in "past" features?
- [ ] Did you compute **group aggregates** across the full dataset?
- [ ] Is performance **too good** on validation but poor when deployed?

</div>

---

## A Famous Real Example: the Heritage Health Prize

The 2011 Heritage Health Prize ($3M competition) had multiple top-performing teams disqualified for leakage. One team achieved AUC=0.98 on validation — far above human baseline — by accidentally using a feature derived from the target variable. When the error was discovered and the feature removed, performance dropped to AUC=0.76.

The lesson: **extraordinary results require extraordinary scrutiny of data**.
