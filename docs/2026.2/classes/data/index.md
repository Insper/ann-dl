## Data in Machine Learning

All machine learning is fundamentally a **data problem**. No matter how sophisticated the model architecture, it cannot compensate for data that is poorly collected, incorrectly labeled, or improperly split. Understanding data — its types, distributions, quality, and pitfalls — is the first and most critical step in any ML project.

> "Garbage in, garbage out." — Classic ML maxim

This section is organized into six focused topics:

<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:1rem;margin:2rem 0;">

<a href="types/" style="text-decoration:none;">
<div style="background:#0d1117;border:1px solid #30363d;border-left:4px solid #58a6ff;border-radius:8px;padding:1.2rem;transition:border-color .2s;">
<div style="color:#58a6ff;font-weight:bold;font-size:1rem;margin-bottom:.4rem;">📊 Feature Types</div>
<div style="color:#8b949e;font-size:.85rem;">Numerical, categorical, ordinal, binary, text, image, time series. How data type drives modeling choices.</div>
</div></a>

<a href="distributions/" style="text-decoration:none;">
<div style="background:#0d1117;border:1px solid #30363d;border-left:4px solid #3fb950;border-radius:8px;padding:1.2rem;transition:border-color .2s;">
<div style="color:#3fb950;font-weight:bold;font-size:1rem;margin-bottom:.4rem;">📈 Distributions & Visualization</div>
<div style="color:#8b949e;font-size:.85rem;">Gaussian, uniform, multimodal, and real datasets (Iris, Salmon/Seabass). How to explore and visualize data.</div>
</div></a>

<a href="splitting/" style="text-decoration:none;">
<div style="background:#0d1117;border:1px solid #30363d;border-left:4px solid #f0883e;border-radius:8px;padding:1.2rem;transition:border-color .2s;">
<div style="color:#f0883e;font-weight:bold;font-size:1rem;margin-bottom:.4rem;">✂️ Train / Val / Test Split</div>
<div style="color:#8b949e;font-size:.85rem;">Why the three-way split matters, cross-validation, stratified splits, and the golden rule of the test set.</div>
</div></a>

<a href="leakage/" style="text-decoration:none;">
<div style="background:#0d1117;border:1px solid #30363d;border-left:4px solid #ff7b72;border-radius:8px;padding:1.2rem;transition:border-color .2s;">
<div style="color:#ff7b72;font-weight:bold;font-size:1rem;margin-bottom:.4rem;">🚨 Data Leakage</div>
<div style="color:#8b949e;font-size:.85rem;">The silent model-killer. Target leakage, temporal leakage, train-test contamination, and how to detect them.</div>
</div></a>

<a href="quality/" style="text-decoration:none;">
<div style="background:#0d1117;border:1px solid #30363d;border-left:4px solid #d29922;border-radius:8px;padding:1.2rem;transition:border-color .2s;">
<div style="color:#d29922;font-weight:bold;font-size:1rem;margin-bottom:.4rem;">🧹 Data Quality</div>
<div style="color:#8b949e;font-size:.85rem;">Missing values (MCAR/MAR/MNAR), outliers, duplicates, noise. Cleaning strategies and imputation.</div>
</div></a>

<a href="imbalance/" style="text-decoration:none;">
<div style="background:#0d1117;border:1px solid #30363d;border-left:4px solid #bc8cff;border-radius:8px;padding:1.2rem;transition:border-color .2s;">
<div style="color:#bc8cff;font-weight:bold;font-size:1rem;margin-bottom:.4rem;">⚖️ Class Imbalance</div>
<div style="color:#8b949e;font-size:.85rem;">When one class dominates. Oversampling (SMOTE), undersampling, class weights, and proper evaluation.</div>
</div></a>

</div>

---

## The Data Pipeline

Before feeding data to any model, it passes through a series of transformations. Understanding the **full pipeline** helps you avoid bugs and leakage:

```mermaid
flowchart LR
    A["Raw Data\ncollection"] --> B["Exploratory\nData Analysis"]
    B --> C["Data Cleaning\n(quality issues)"]
    C --> D["Split\ntrain / val / test"]
    D --> E["Feature Engineering\n& Preprocessing"]
    E --> F["Model\nTraining"]
    F --> G["Evaluation\non test set"]
    
    style D fill:#1f3244,stroke:#58a6ff
    style G fill:#1f3d1f,stroke:#3fb950
```

!!! danger "Critical Rule"
    **Always split BEFORE preprocessing.** Computing statistics (mean, std, min, max) on the full dataset and then splitting is data leakage. Fit all transformers on training data only.

---

## Key Data Repositories

| Source | Domain | Format |
|--------|---------|--------|
| [UCI ML Repository](https://archive.ics.uci.edu/){:target="_blank"} | General ML | CSV, ARFF |
| [Kaggle Datasets](https://www.kaggle.com/datasets){:target="_blank"} | All domains | CSV, JSON |
| [Hugging Face Datasets](https://huggingface.co/datasets){:target="_blank"} | NLP, Vision | Arrow, Parquet |
| [OpenML](https://www.openml.org/){:target="_blank"} | Benchmarks | ARFF |
| [TensorFlow Datasets](https://www.tensorflow.org/datasets){:target="_blank"} | Vision, NLP, Audio | TFRecord |
| [Papers With Code](https://paperswithcode.com/datasets){:target="_blank"} | Research benchmarks | Various |
| [Google Dataset Search](https://datasetsearch.research.google.com/){:target="_blank"} | Web-wide | Various |

---

--8<-- "docs/2026.2/classes/data/quiz.md"
