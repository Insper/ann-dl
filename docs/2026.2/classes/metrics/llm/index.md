## Métricas para Modelos de Linguagem

Avaliar Large Language Models é fundamentalmente diferente de avaliar classificadores ou modelos de regressão. A saída é texto aberto — há múltiplas respostas corretas, a fluência importa, e "certo" é muitas vezes subjetivo. Por isso, o campo desenvolveu métricas especializadas.

---

## Perplexidade (Perplexity)

A métrica mais fundamental para modelos de linguagem: mede quão "surpreendido" o modelo fica com um texto de teste.

$$
\text{PPL}(W) = \exp\!\left(-\frac{1}{N}\sum_{i=1}^{N} \log p_\theta(w_i \mid w_1, \ldots, w_{i-1})\right)
$$

**Interpretação:** PPL = 10 significa que o modelo está, em média, igualmente incerto entre 10 opções a cada token. Menor é melhor. Um modelo trivial que distribui uniformemente sobre $V$ tokens tem PPL = $V$ ≈ 32.000.

!!! warning "Limitação"
    Perplexidade mede ajuste ao conjunto de teste, não capacidade de seguir instruções, raciocinar ou ser útil. GPT-2 tem PPL baixíssima em Wikipedia mas não segue comandos.

---

## Métricas de Geração de Texto

### BLEU (Bilingual Evaluation Understudy)

Originalmente para tradução automática — compara n-gramas entre geração e referência:

$$
\text{BLEU} = \text{BP} \cdot \exp\!\left(\sum_{n=1}^{N} w_n \log p_n\right)
$$

onde $p_n$ é a precisão de n-gramas e BP é a penalidade por respostas curtas demais.

### ROUGE (Recall-Oriented Understudy for Gisting Evaluation)

Foco em recall — quanto do texto de referência aparece na geração. Comum em sumarização:

| Variante | O que mede |
|----------|-----------|
| ROUGE-1 | Overlap de unigramas |
| ROUGE-2 | Overlap de bigramas |
| ROUGE-L | Longest Common Subsequence |

### BERTScore

Usa embeddings do BERT para comparar semântica, não superfície textual:

$$
\text{BERTScore} = F_1\text{ entre embeddings de referência e hipótese}
$$

Captura paráfrases que BLEU penaliza.

---

## Benchmarks de Capacidade

Métricas modernas avaliam **tarefas**, não apenas fluência:

<div id="bench-viz" style="background:#0d1117;border-radius:12px;padding:1.5rem;margin:2rem 0;overflow:hidden;">
<canvas id="bench-canvas" style="width:100%;display:block;"></canvas>
</div>

<script>
(function(){
  const canvas = document.getElementById('bench-canvas');
  const ctx = canvas.getContext('2d');

  const benchmarks = [
    { name: 'MMLU', desc: 'Conhecimento multidisciplinar (57 áreas)', cat: 'Conhecimento' },
    { name: 'HumanEval', desc: 'Geração de código Python (164 problemas)', cat: 'Código' },
    { name: 'GSM8K', desc: 'Matemática nível primário (8500 problemas)', cat: 'Matemática' },
    { name: 'HellaSwag', desc: 'Completar frases com senso comum', cat: 'Raciocínio' },
    { name: 'TruthfulQA', desc: 'Respostas verídicas a perguntas enganosas', cat: 'Veracidade' },
    { name: 'MATH', desc: 'Matemática avançada (álgebra, cálculo...)', cat: 'Matemática' },
    { name: 'MT-Bench', desc: 'Diálogos multi-turno julgados por GPT-4', cat: 'Instrução' },
    { name: 'Arena (ELO)', desc: 'Preferências humanas via Chatbot Arena', cat: 'Humano' },
  ];

  // Model scores (approximate, illustrative)
  const models = [
    { name: 'GPT-3.5', scores: [70, 48, 57, 82, 47, 34, 7.9, 1100], color: '#58a6ff' },
    { name: 'LLaMA-3 8B', scores: [66, 62, 79, 82, 44, 30, 8.1, 1050], color: '#3fb950' },
    { name: 'GPT-4o', scores: [88, 90, 95, 95, 61, 76, 9.0, 1280], color: '#f0883e' },
    { name: 'DeepSeek R1', scores: [91, 92, 97, 93, 59, 92, 9.4, 1320], color: '#bc8cff' },
  ];

  function draw() {
    const W = canvas.parentElement.offsetWidth - 48;
    const H = 320; canvas.width = W; canvas.height = H; canvas.style.height = H + 'px';
    ctx.fillStyle = '#0d1117'; ctx.fillRect(0, 0, W, H);

    const N = benchmarks.length;
    const colW = (W - 100) / N;
    const maxH = H - 80;
    const baseY = H - 50;

    // Benchmark labels
    benchmarks.forEach((b, i) => {
      const x = 100 + i * colW + colW / 2;
      ctx.fillStyle = '#484f58'; ctx.font = 'bold 9px Inter,sans-serif'; ctx.textAlign = 'center';
      ctx.fillText(b.name, x, baseY + 14);
      ctx.fillStyle = '#30363d'; ctx.font = '8px Inter,sans-serif';
      ctx.fillText(b.cat, x, baseY + 25);

      // Grid line
      ctx.strokeStyle = '#21262d'; ctx.lineWidth = 0.5;
      ctx.beginPath(); ctx.moveTo(x, 20); ctx.lineTo(x, baseY); ctx.stroke();
    });

    // Y axis
    [0, 25, 50, 75, 100].forEach(v => {
      const y = baseY - (v / 100) * maxH;
      ctx.strokeStyle = '#21262d'; ctx.lineWidth = 0.5;
      ctx.beginPath(); ctx.moveTo(95, y); ctx.lineTo(W, y); ctx.stroke();
      ctx.fillStyle = '#484f58'; ctx.font = '9px monospace'; ctx.textAlign = 'right';
      ctx.fillText(v + '%', 90, y + 4);
    });

    // Bars per model
    const barW = Math.max(4, colW / models.length - 3);
    const groupW = barW * models.length + 3 * (models.length - 1);
    models.forEach((m, mi) => {
      benchmarks.forEach((b, bi) => {
        const rawScore = m.scores[bi];
        const normScore = bi === 7 ? (rawScore - 900) / 5 : rawScore; // ELO normalization
        const pct = Math.min(100, Math.max(0, normScore));
        const x = 100 + bi * colW + (colW - groupW) / 2 + mi * (barW + 3);
        const barH = (pct / 100) * maxH;
        const y = baseY - barH;

        ctx.fillStyle = m.color + 'bb';
        ctx.beginPath(); ctx.roundRect(x, y, barW, barH, 2); ctx.fill();
      });
    });

    // Legend
    models.forEach((m, i) => {
      ctx.fillStyle = m.color;
      ctx.fillRect(100 + i * 100, H - 14, 10, 8);
      ctx.font = '9px Inter,sans-serif'; ctx.textAlign = 'left';
      ctx.fillText(m.name, 114 + i * 100, H - 7);
    });
  }

  draw(); window.addEventListener('resize', draw);
})();
</script>

| Benchmark | Domínio | Formato | Nota |
|-----------|---------|---------|------|
| **MMLU** | 57 disciplinas | Multiple-choice | Padrão ouro para conhecimento |
| **HumanEval** | Código Python | Completar função | Pass@k: % problemas resolvidos |
| **GSM8K** | Aritmética | Passo-a-passo | Requer chain-of-thought |
| **HellaSwag** | Senso comum | Múltipla escolha | Fácil pra humanos (95%), difícil pra modelos |
| **TruthfulQA** | Veracidade | Geração livre | Mede tendência a alucinar |
| **MATH** | Matemática avançada | Resolução de problemas | Exige raciocínio simbólico |
| **MT-Bench** | Instrução | Diálogo multi-turno | Nota 1-10 por LLM-juiz |
| **Chatbot Arena** | Preferência geral | A/B humano | ELO rating, mais realista |

---

## Avaliação por LLM-juiz (LLM-as-Judge)

Uma tendência crescente: usar um LLM potente (GPT-4, Claude) para avaliar as respostas de outro LLM. Permite avaliar qualidade subjetiva em escala.

```python
def llm_judge(question, response_a, response_b, judge_model="gpt-4"):
    prompt = f"""
    Pergunta: {question}
    
    Resposta A: {response_a}
    Resposta B: {response_b}
    
    Qual resposta é melhor? Avalie em termos de precisão, clareza e utilidade.
    Responda apenas com 'A', 'B', ou 'Empate' e uma breve justificativa.
    """
    return judge_model.generate(prompt)
```

**Vieses conhecidos:** preferência por respostas longas, preferência pela primeira opção, favorecer o próprio modelo (self-preference).

---

## Métricas de Alinhamento e Segurança

| Métrica | O que avalia |
|---------|-------------|
| **Toxicidade** (Perspective API) | Conteúdo ofensivo, ódio |
| **Veracidade** (TruthfulQA) | Taxa de alucinações |
| **Instrução-following** | % de instruções seguidas corretamente |
| **Refusal rate** | Taxa de recusa em pedidos prejudiciais |
| **Calibração** | Confiança vs. acurácia real |

---

## Avaliação de RAG (Retrieval-Augmented Generation)

Sistemas RAG têm métricas adicionais:

| Métrica | Fórmula | O que mede |
|---------|---------|-----------|
| **Context Precision** | Docs relevantes / docs recuperados | Qualidade da recuperação |
| **Context Recall** | Docs relevantes encontrados / todos relevantes | Cobertura da recuperação |
| **Faithfulness** | Claims suportados / total de claims | Fidelidade ao contexto |
| **Answer Relevancy** | Similaridade resposta↔pergunta | Pertinência da resposta |

---

--8<-- "docs/2026.2/classes/metrics/llm/quiz.md"
