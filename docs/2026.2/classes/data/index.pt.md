## Dados em Aprendizado de Máquina

Todo aprendizado de máquina é fundamentalmente um **problema de dados**. Independentemente de quão sofisticada seja a arquitetura do modelo, ela não pode compensar dados mal coletados, incorretamente rotulados ou impropriamente divididos. Compreender os dados — seus tipos, distribuições, qualidade e armadilhas — é o primeiro e mais crítico passo em qualquer projeto de AM.

> "Lixo entra, lixo sai." — Máxima clássica do AM

Esta seção está organizada em seis tópicos focados:

<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:1rem;margin:2rem 0;">

<a href="types/" style="text-decoration:none;">
<div style="background:#0d1117;border:1px solid #30363d;border-left:4px solid #58a6ff;border-radius:8px;padding:1.2rem;transition:border-color .2s;">
<div style="color:#58a6ff;font-weight:bold;font-size:1rem;margin-bottom:.4rem;">📊 Tipos de Features</div>
<div style="color:#8b949e;font-size:.85rem;">Numéricas, categóricas, ordinais, binárias, texto, imagem, séries temporais. Como o tipo de dado orienta as escolhas de modelagem.</div>
</div></a>

<a href="distributions/" style="text-decoration:none;">
<div style="background:#0d1117;border:1px solid #30363d;border-left:4px solid #3fb950;border-radius:8px;padding:1.2rem;transition:border-color .2s;">
<div style="color:#3fb950;font-weight:bold;font-size:1rem;margin-bottom:.4rem;">📈 Distribuições e Visualização</div>
<div style="color:#8b949e;font-size:.85rem;">Gaussiana, uniforme, multimodal e datasets reais (Iris, Salmão/Robalo). Como explorar e visualizar dados.</div>
</div></a>

<a href="splitting/" style="text-decoration:none;">
<div style="background:#0d1117;border:1px solid #30363d;border-left:4px solid #f0883e;border-radius:8px;padding:1.2rem;transition:border-color .2s;">
<div style="color:#f0883e;font-weight:bold;font-size:1rem;margin-bottom:.4rem;">✂️ Divisão Treino / Val / Teste</div>
<div style="color:#8b949e;font-size:.85rem;">Por que a divisão em três conjuntos importa, validação cruzada, divisões estratificadas e a regra de ouro do conjunto de teste.</div>
</div></a>

<a href="leakage/" style="text-decoration:none;">
<div style="background:#0d1117;border:1px solid #30363d;border-left:4px solid #ff7b72;border-radius:8px;padding:1.2rem;transition:border-color .2s;">
<div style="color:#ff7b72;font-weight:bold;font-size:1rem;margin-bottom:.4rem;">🚨 Vazamento de Dados</div>
<div style="color:#8b949e;font-size:.85rem;">O destruidor silencioso de modelos. Vazamento de alvo, vazamento temporal, contaminação treino-teste e como detectá-los.</div>
</div></a>

<a href="quality/" style="text-decoration:none;">
<div style="background:#0d1117;border:1px solid #30363d;border-left:4px solid #d29922;border-radius:8px;padding:1.2rem;transition:border-color .2s;">
<div style="color:#d29922;font-weight:bold;font-size:1rem;margin-bottom:.4rem;">🧹 Qualidade dos Dados</div>
<div style="color:#8b949e;font-size:.85rem;">Valores faltantes (MCAR/MAR/MNAR), outliers, duplicatas, ruído. Estratégias de limpeza e imputação.</div>
</div></a>

<a href="imbalance/" style="text-decoration:none;">
<div style="background:#0d1117;border:1px solid #30363d;border-left:4px solid #bc8cff;border-radius:8px;padding:1.2rem;transition:border-color .2s;">
<div style="color:#bc8cff;font-weight:bold;font-size:1rem;margin-bottom:.4rem;">⚖️ Desbalanceamento de Classes</div>
<div style="color:#8b949e;font-size:.85rem;">Quando uma classe domina. Oversampling (SMOTE), undersampling, pesos de classe e avaliação adequada.</div>
</div></a>

</div>

---

## O Pipeline de Dados

Antes de alimentar dados a qualquer modelo, eles passam por uma série de transformações. Compreender o **pipeline completo** ajuda a evitar bugs e vazamentos:

```mermaid
flowchart LR
    A["Coleta de\nDados Brutos"] --> B["Análise\nExploratória"]
    B --> C["Limpeza de Dados\n(problemas de qualidade)"]
    C --> D["Divisão\ntreino / val / teste"]
    D --> E["Engenharia de Features\ne Pré-processamento"]
    E --> F["Treinamento\ndo Modelo"]
    F --> G["Avaliação\nno conjunto de teste"]
    
    style D fill:#1f3244,stroke:#58a6ff
    style G fill:#1f3d1f,stroke:#3fb950
```

!!! danger "Regra Crítica"
    **Sempre divida ANTES do pré-processamento.** Calcular estatísticas (média, desvio padrão, mínimo, máximo) no dataset completo e depois dividir é vazamento de dados. Ajuste todos os transformadores apenas nos dados de treinamento.

---

## Principais Repositórios de Dados

| Fonte | Domínio | Formato |
|--------|---------|--------|
| [UCI ML Repository](https://archive.ics.uci.edu/){:target="_blank"} | AM geral | CSV, ARFF |
| [Kaggle Datasets](https://www.kaggle.com/datasets){:target="_blank"} | Todos os domínios | CSV, JSON |
| [Hugging Face Datasets](https://huggingface.co/datasets){:target="_blank"} | PLN, Visão | Arrow, Parquet |
| [OpenML](https://www.openml.org/){:target="_blank"} | Benchmarks | ARFF |
| [TensorFlow Datasets](https://www.tensorflow.org/datasets){:target="_blank"} | Visão, PLN, Áudio | TFRecord |
| [Papers With Code](https://paperswithcode.com/datasets){:target="_blank"} | Benchmarks de pesquisa | Vários |
| [Google Dataset Search](https://datasetsearch.research.google.com/){:target="_blank"} | Web | Vários |

---

--8<-- "docs/2026.2/classes/data/quiz.md"
