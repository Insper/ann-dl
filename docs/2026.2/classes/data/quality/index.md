## Data Quality

Real-world data is messy. Before training any model, you must understand the quality of your data, identify problems, and decide how to address them. Poor data quality is often harder to fix than a poor model choice.

> "Data scientists spend 60–80% of their time cleaning and preparing data." — Common industry estimate

---

## Missing Values

Missing data (NaN, NULL, None) is the most common data quality issue. But **not all missing data is the same** — the *mechanism* of missingness determines the right strategy.

| Mechanism | Definition | Example | Strategy |
|-----------|-----------|---------|---------|
| **MCAR** — Missing Completely At Random | Missing independently of the value | Sensor failure at random times | Any imputation or deletion safe |
| **MAR** — Missing At Random | Missing depends on other observed variables | Income not reported more often by younger people (age is observed) | Impute using other features |
| **MNAR** — Missing Not At Random | Missing depends on the missing value itself | High-income people less likely to report income | Very hard; may need domain knowledge |

### Detecting Missing Values

```python
import pandas as pd

# Count missing values
print(df.isnull().sum())
print(df.isnull().mean() * 100)  # as percentage

# Heatmap of missing patterns
import seaborn as sns
import matplotlib.pyplot as plt
sns.heatmap(df.isnull(), cbar=False, yticklabels=False, cmap='viridis')
```

### Imputation Strategies

<div id="impute-viz" style="background:#0d1117;border-radius:12px;padding:1.5rem;margin:2rem 0;overflow:hidden;">
<canvas id="impute-canvas" style="width:100%;display:block;"></canvas>
<div style="display:flex;gap:.5rem;justify-content:center;flex-wrap:wrap;margin-top:.8rem;">
  <button onclick="imputeShow('original')" id="imp-orig" style="padding:4px 14px;background:#21262d;color:#c9d1d9;border:1px solid #30363d;border-radius:5px;cursor:pointer;font-size:.85rem;">Original</button>
  <button onclick="imputeShow('mean')" id="imp-mean" style="padding:4px 14px;background:#21262d;color:#c9d1d9;border:1px solid #30363d;border-radius:5px;cursor:pointer;font-size:.85rem;">Mean</button>
  <button onclick="imputeShow('median')" id="imp-median" style="padding:4px 14px;background:#21262d;color:#c9d1d9;border:1px solid #30363d;border-radius:5px;cursor:pointer;font-size:.85rem;">Median</button>
  <button onclick="imputeShow('knn')" id="imp-knn" style="padding:4px 14px;background:#21262d;color:#c9d1d9;border:1px solid #30363d;border-radius:5px;cursor:pointer;font-size:.85rem;">KNN</button>
</div>
<div id="impute-desc" style="color:#8b949e;font-size:.82rem;text-align:center;margin-top:.6rem;"></div>
</div>

<script>
(function(){
  // Synthetic data: bimodal + skewed with outlier
  const rng=s=>{let x=Math.sin(s*31.7)*43758.5;return x-Math.floor(x);};
  const N=30;
  const allData=Array.from({length:N},(_,i)=>{
    const v=i<15?rng(i*7)*2+2:rng(i*7+500)*2+6;
    return v+(i===5?8:0); // outlier
  });
  const missing=[3,9,14,19,24]; // indices of missing values
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
        // nearest observed neighbors
        const dists=observed.map((ov,oi)=>({d:Math.abs(allData[i]-ov),v:ov}));
        // just use 3 neighbors by index proximity
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
    original:'Missing values shown as gaps. The 5 missing positions (★) cannot be used for training.',
    mean:'Mean imputation: fill with column mean. Fast but ignores distribution shape; pulls values toward center.',
    median:'Median imputation: fill with column median. More robust to outliers than mean imputation.',
    knn:'KNN imputation: use average of k nearest observed neighbors. Preserves local structure better.'
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

    // Draw line
    ctx.beginPath();ctx.strokeStyle='#30363d';ctx.lineWidth=1;
    let started=false;
    pts.forEach((p,i)=>{
      if(p.missing)return;
      const x=cx(i),y=cy(p.v);
      if(!started){ctx.moveTo(x,y);started=true;}else ctx.lineTo(x,y);
    });
    ctx.stroke();

    // Mean/median reference line
    if(mode==='mean'||mode==='median'){
      const refV=mode==='mean'?mean:median;
      ctx.strokeStyle='#d29922aa';ctx.lineWidth=1;ctx.setLineDash([4,3]);
      ctx.beginPath();ctx.moveTo(pad,cy(refV));ctx.lineTo(W-pad,cy(refV));ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle='#d29922';ctx.font='8px monospace';ctx.textAlign='left';
      ctx.fillText((mode==='mean'?'mean':'median')+'='+refV.toFixed(1),pad+4,cy(refV)-4);
    }

    // Draw points
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

    ctx.fillStyle='#58a6ff';ctx.font='9px Inter,sans-serif';ctx.textAlign='left';ctx.fillText('● Observed',pad,12);
    if(mode!=='original'){ctx.fillStyle='#f0883e';ctx.fillText('● Imputed',pad+65,12);}
    ctx.fillStyle='#ff7b72';ctx.fillText('★ Missing',pad+(mode!=='original'?130:65),12);
  }

  imputeShow('original');
  window.addEventListener('resize',draw);
})();
</script>

```python
from sklearn.impute import SimpleImputer, KNNImputer

# Mean imputation
imp = SimpleImputer(strategy='mean')
X_train = imp.fit_transform(X_train)
X_test  = imp.transform(X_test)  # use train statistics!

# KNN imputation (more accurate, slower)
imp = KNNImputer(n_neighbors=5)
X_train = imp.fit_transform(X_train)
X_test  = imp.transform(X_test)
```

---

## Outliers

Outliers are data points that deviate significantly from the rest. They can be:

- **Genuine**: extreme but valid observations (e.g., a billionaire in an income dataset)
- **Errors**: measurement mistakes, data entry errors

### Detection Methods

```python
import numpy as np

# Z-score method
z_scores = np.abs((df - df.mean()) / df.std())
outliers = (z_scores > 3).any(axis=1)

# IQR method (more robust)
Q1, Q3 = df.quantile(0.25), df.quantile(0.75)
IQR = Q3 - Q1
outliers = ((df < Q1 - 1.5*IQR) | (df > Q3 + 1.5*IQR)).any(axis=1)

print(f"Outliers: {outliers.sum()} / {len(df)} ({outliers.mean()*100:.1f}%)")
```

### Handling Strategies

| Strategy | When to use |
|----------|-------------|
| **Remove** | Clear measurement errors, small percentage |
| **Cap/Winsorize** | Keep value but clip to percentile | 
| **Transform** (log, sqrt) | Right-skewed data with many high outliers |
| **Keep** | Genuine extreme values relevant to the task |

---

## Duplicates and Noise

**Duplicates** — identical or near-identical rows — inflate training counts and can cause overfitting:

```python
# Check duplicates
print(f"Duplicates: {df.duplicated().sum()}")
df = df.drop_duplicates()

# Near-duplicates (fuzzy)
from sklearn.metrics.pairwise import cosine_similarity
# Compare all pairs — expensive for large datasets
```

**Noise** — random errors in feature values or labels — is harder to detect. Strategies:

- Label smoothing (soft targets instead of hard 0/1)
- Data augmentation
- Ensemble methods (averaging over noise)
- Confident learning (identify likely mislabeled samples)

---

## Data Quality Audit Checklist

```python
def data_quality_report(df, target_col=None):
    print("=== Data Quality Report ===")
    print(f"Shape: {df.shape}")
    print(f"\nMissing values:")
    print(df.isnull().sum()[df.isnull().sum() > 0])
    print(f"\nDuplicates: {df.duplicated().sum()}")
    print(f"\nData types:\n{df.dtypes}")
    if target_col:
        print(f"\nClass distribution:\n{df[target_col].value_counts(normalize=True)}")
    print(f"\nNumerical summary:")
    print(df.describe())
```
