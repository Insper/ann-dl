## Class Imbalance

Class imbalance occurs when the distribution of target classes is highly skewed — one or more classes dominate the dataset. This is the norm, not the exception, in real-world problems: fraud detection (99.9% legitimate), disease diagnosis (95% healthy), spam detection (80% legitimate email).

---

## Why Imbalance Breaks Standard Training

A model trained with cross-entropy loss on an imbalanced dataset will **maximize accuracy by predicting the majority class**. With 99% negatives, predicting always-negative gives 99% accuracy — but 0% recall on positives.

<div id="imb-viz" style="background:#0d1117;border-radius:12px;padding:1.5rem;margin:2rem 0;">
<canvas id="imb-canvas" style="width:100%;display:block;"></canvas>
<div style="margin-top:.8rem;">
  <label style="color:#8b949e;font-size:.85rem;">Minority class %: </label>
  <input id="imb-pct" type="range" min="1" max="50" step="1" value="5" style="accent-color:#ff7b72;vertical-align:middle;width:200px;" oninput="imbDraw()">
  <span id="imb-pct-val" style="color:#ff7b72;font-family:monospace;font-weight:bold;"></span>
</div>
<div id="imb-naive-acc" style="color:#8b949e;font-size:.82rem;margin-top:.5rem;font-family:monospace;"></div>
</div>

<script>
window.imbDraw = function() {
  const canvas = document.getElementById('imb-canvas');
  const ctx = canvas.getContext('2d');
  const W = canvas.parentElement.offsetWidth - 48, H = 80;
  canvas.width = W; canvas.height = H; canvas.style.height = H + 'px';
  ctx.fillStyle = '#0d1117'; ctx.fillRect(0,0,W,H);

  const pct = +document.getElementById('imb-pct').value;
  document.getElementById('imb-pct-val').textContent = pct + '%';

  const pad = 10, bw = W - 2*pad, bh = 44, by = (H-bh)/2;
  const majW = bw * (100-pct)/100, minW = bw * pct/100;

  ctx.fillStyle = '#58a6ff22'; ctx.fillRect(pad, by, majW, bh);
  ctx.strokeStyle = '#58a6ff'; ctx.lineWidth = 2; ctx.strokeRect(pad, by, majW, bh);
  ctx.fillStyle = '#ff7b7222'; ctx.fillRect(pad+majW, by, minW, bh);
  ctx.strokeStyle = '#ff7b72'; ctx.lineWidth = 2; ctx.strokeRect(pad+majW, by, minW, bh);

  ctx.textAlign = 'center'; ctx.textBaseline = 'middle'; ctx.font = 'bold 11px Inter,sans-serif';
  if (majW > 80) { ctx.fillStyle = '#58a6ff'; ctx.fillText('Majority class (' + (100-pct) + '%)', pad + majW/2, by + bh/2); }
  if (minW > 30) { ctx.fillStyle = '#ff7b72'; ctx.fillText('Minority (' + pct + '%)', pad+majW+minW/2, by+bh/2); }
  else { ctx.fillStyle = '#ff7b72'; ctx.font = '8px monospace'; ctx.fillText(pct+'%', pad+majW+minW/2, by+bh/2); }

  const naiveAcc = (100-pct).toFixed(1);
  document.getElementById('imb-naive-acc').innerHTML =
    'Always-majority classifier accuracy: <span style="color:#ff7b72;font-weight:bold;">' + naiveAcc + '%</span> &nbsp;|&nbsp; ' +
    'Minority recall: <span style="color:#ff7b72;font-weight:bold;">0%</span> &nbsp;|&nbsp; ' +
    'F1 (minority): <span style="color:#ff7b72;font-weight:bold;">0.00</span> — completely useless despite high accuracy!';
};
imbDraw();
window.addEventListener('resize', imbDraw);
</script>

---

## Strategies to Handle Imbalance

### 1. Use the Right Metric

First and most important: **stop using accuracy**. Use:

| Metric | Good when |
|--------|---------|
| **F1-Score (macro/weighted)** | Binary or multiclass, want balance |
| **Precision-Recall AUC** | When positives are rare and important |
| **ROC-AUC** | When you need threshold-agnostic evaluation |
| **Matthews Correlation Coefficient (MCC)** | Binary classification, very imbalanced |
| **G-Mean** | Geometric mean of recall per class |

### 2. Class Weights

Tell the loss function to penalize errors on the minority class more:

```python
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

# Automatic class weight computation
weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight = dict(zip(np.unique(y_train), weights))

# For sklearn models
model = LogisticRegression(class_weight='balanced')

# For PyTorch
import torch
pos_weight = torch.tensor([neg_count / pos_count])
criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
```

### 3. Oversampling — SMOTE

**SMOTE** (Synthetic Minority Over-sampling Technique) generates synthetic minority samples by interpolating between existing ones:

```python
from imblearn.over_sampling import SMOTE

smote = SMOTE(sampling_strategy='auto', k_neighbors=5, random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

print(f"Before: {dict(zip(*np.unique(y_train, return_counts=True)))}")
print(f"After:  {dict(zip(*np.unique(y_resampled, return_counts=True)))}")
```

!!! warning "Apply SMOTE only on training data"
    Never resample the validation or test set. This would give an unrealistic class distribution not representative of production.

### 4. Undersampling

Remove majority class samples. Simpler than SMOTE but loses information:

```python
from imblearn.under_sampling import RandomUnderSampler, TomekLinks

# Random undersampling
rus = RandomUnderSampler(sampling_strategy=0.5)  # minority:majority = 1:2
X_res, y_res = rus.fit_resample(X_train, y_train)

# Tomek Links: remove majority samples near the boundary
tl = TomekLinks()
X_res, y_res = tl.fit_resample(X_train, y_train)
```

### 5. Threshold Tuning

Standard classifiers output probabilities. The default threshold (0.5) may not be optimal for imbalanced data. Tune it on the validation set:

```python
from sklearn.metrics import precision_recall_curve
import numpy as np

probs = model.predict_proba(X_val)[:, 1]
precisions, recalls, thresholds = precision_recall_curve(y_val, probs)

# Find threshold maximizing F1
f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-8)
best_thresh = thresholds[np.argmax(f1_scores)]
y_pred = (probs >= best_thresh).astype(int)
```

---

## Strategy Decision Guide

```mermaid
flowchart TD
    A[Imbalanced dataset] --> B{Imbalance ratio?}
    B -->|"< 1:10"| C[Class weights\noften sufficient]
    B -->|"1:10 to 1:100"| D{Enough minority samples?}
    B -->|"> 1:100"| E[Combine: SMOTE\n+ class weights\n+ threshold tuning]
    D -->|"> 500 samples| F[SMOTE or\ncombined approach]
    D -->|"< 500 samples"| G[Class weights\n+ data collection]
    C --> H[Evaluate with\nPR-AUC / F1]
    F --> H
    E --> H
    G --> H
```

---

## Imbalance in Deep Learning

For neural networks with large datasets, **class weights** are usually the most practical and effective solution. SMOTE on millions of samples is slow and the synthetic samples may not match the learned feature distribution well.

Additional techniques specific to deep learning:

| Technique | Idea |
|-----------|------|
| **Focal Loss** | Down-weights easy examples (well-classified majority), focuses on hard minority | 
| **Label Smoothing** | Prevents over-confident predictions on majority class |
| **Balanced Batch Sampling** | Ensure each batch contains equal minority/majority samples |
| **Mixup** | Interpolate between samples of different classes |

```python
# Focal Loss (binary)
class FocalLoss(torch.nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25):
        super().__init__()
        self.gamma, self.alpha = gamma, alpha

    def forward(self, pred, target):
        bce = torch.nn.functional.binary_cross_entropy_with_logits(pred, target, reduction='none')
        pt = torch.exp(-bce)
        return (self.alpha * (1 - pt)**self.gamma * bce).mean()
```
