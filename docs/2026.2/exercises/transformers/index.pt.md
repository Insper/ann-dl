!!! success inline end "Prazo e Entrega"

    :date: A definir
    
    :clock1: Commits até 23:59

    :material-account: Individual

    :simple-github: Link do GitHub Pages via [insper.blackboard.com](https://insper.blackboard.com){:target="_blank"}.

**Atividade: Construindo Atenção e Transformers do Zero**

Esta atividade consolida seu entendimento sobre mecanismos de atenção e arquitetura Transformer implementando-os do zero usando **apenas NumPy e Python** (sem PyTorch ou TensorFlow para a lógica central).

---

## Exercício 1 — Atenção de Produto Escalar Escalado

Implemente a função de atenção de produto escalar escalado completa:

$$
\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
$$

### Instruções

1. **Implemente `softmax(x)`** — versão numericamente estável (subtraia o máximo antes de exponenciar)
2. **Implemente `scaled_dot_product_attention(Q, K, V, mask=None)`**:
   - Compute scores brutos: `scores = Q @ K.T / sqrt(d_k)`
   - Aplique a máscara se fornecida (defina posições mascaradas para `-inf` antes do softmax)
   - Aplique softmax por linha
   - Retorne a soma ponderada dos Valores: `output = attn_weights @ V`
3. **Teste com as seguintes entradas:**

```python
import numpy as np

d_k = 4
Q = np.array([[1.0, 0.0, 1.0, 0.0],   # query do token 1
              [0.0, 1.0, 0.0, 1.0]])   # query do token 2
K = np.array([[1.0, 0.0, 1.0, 0.0],   # key do token 1
              [0.0, 1.0, 0.0, 1.0],   # key do token 2
              [1.0, 1.0, 0.0, 0.0]])  # key do token 3
V = np.array([[1.0, 0.0],
              [0.0, 1.0],
              [0.5, 0.5]])
```

4. **Plote a matriz de pesos de atenção** como um heatmap (use matplotlib). Qual padrão você observa? O token 1 presta mais atenção ao token 1 ou ao token 3? Por quê?
5. **Aplique uma máscara causal** (triangular inferior) e execute novamente. Mostre como os pesos de atenção mudam e explique por que esta máscara é necessária para a geração autorregressiva.

### Saída Esperada

Reporte:

- A matriz de pesos de atenção (2×3) antes e depois da máscara
- A matriz de saída (2×2)
- Visualizações de heatmap
- Uma breve explicação de por que Q1 presta mais atenção a K1 do que a K2

---

## Exercício 2 — Atenção Multi-Cabeça do Zero

Estenda sua implementação para **Atenção Multi-Cabeça** com $h=2$ cabeças.

### Arquitetura

$$
\text{MultiHead}(Q,K,V) = \text{Concat}(\text{head}_1, \text{head}_2)\,W^O
$$

onde cada cabeça usa suas próprias matrizes de projeção.

### Instruções

1. **Implemente a classe `MultiHeadAttention`** com:
   - `__init__(d_model, num_heads)` — inicialize matrizes de pesos aleatórias $W_Q^i, W_K^i, W_V^i \in \mathbb{R}^{d_{model} \times d_k}$ e $W^O \in \mathbb{R}^{d_{model} \times d_{model}}$ para cada cabeça $i$
   - `forward(Q, K, V)` — projete, aplique atenção por cabeça, concatene, projete saída
2. **Use semente aleatória fixa** `np.random.seed(42)` para reprodutibilidade
3. **Teste com uma sequência de 5 tokens**, cada um com `d_model=8`, `num_heads=2` (então `d_k=4` por cabeça)
4. **Verifique** que o formato de saída é `(5, 8)` — mesmo que a entrada

### Perguntas para responder no relatório

- Por que usar $h=2$ cabeças com $d_k = d_{model}/h$ mantém a computação total similar a $h=1$?
- Se a cabeça 1 aprende a atender tokens próximos e a cabeça 2 a tokens distantes, como a saída concatenada se beneficia de ambos?

---

## Exercício 3 — Bloco Transformer de Camada Única

Combine sua atenção com uma Rede Feed-Forward para implementar um **Bloco Encoder Transformer**:

$$
x' = \text{LayerNorm}(x + \text{MultiHeadAttn}(x, x, x))
$$
$$
x'' = \text{LayerNorm}(x' + \text{FFN}(x'))
$$

### Instruções

1. **Implemente `layer_norm(x)`** — normalize por linha (subtraia a média, divida pelo desvio padrão + ε), com γ=1, β=0 aprendíveis
2. **Implemente `ffn(x, W1, b1, W2, b2)`** — duas camadas lineares com ReLU: $\text{FFN}(x) = \text{ReLU}(xW_1 + b_1)W_2 + b_2$, onde $d_{ff} = 4 \times d_{model}$
3. **Monte tudo**: construa `transformer_encoder_block(x, mha, W1, b1, W2, b2)` usando suas implementações
4. **Teste com 5 tokens, d_model=8**

### Visualização

Plote as **representações dos tokens antes e depois do bloco** como um heatmap (tokens × dimensões). As representações ficam mais ricas após o bloco? Compute e reporte a matriz de similaridade cosseno entre tokens antes e depois do bloco.

---

## Exercício 4 — Codificação Posicional

Implemente a codificação posicional senoidal e visualize-a.

$$
PE_{(pos, 2i)} = \sin\!\left(\frac{pos}{10000^{2i/d}}\right), \quad PE_{(pos, 2i+1)} = \cos\!\left(\frac{pos}{10000^{2i/d}}\right)
$$

### Instruções

1. Implemente `positional_encoding(max_len, d_model)` → retorna matriz de forma `(max_len, d_model)`
2. **Plote duas visualizações**:
   - Heatmap: linhas=posições (0–99), colunas=dimensões, cor=valor PE
   - Gráfico de linha: valores PE para as posições 0, 10, 50 ao longo de todas as dimensões

### Perguntas

- Quais dimensões codificam oscilações de alta frequência e quais codificam baixa frequência?
- Por que adicionar PE ao embedding do token permite ao Transformer distinguir a posição 1 da posição 50?
- O que acontece se você adicionar a mesma codificação posicional a tokens embaralhados?

---

## Critérios de Avaliação

!!! failure "Uso de Toolboxes"
    Você pode usar apenas **NumPy** para operações matriciais e **Matplotlib/Seaborn** para plots. PyTorch, TensorFlow e outros frameworks de AM são **estritamente proibidos** para a implementação central. Verifique seus resultados contra a saída do `nn.MultiheadAttention` do PyTorch apenas como verificação de sanidade.

    **O descumprimento resultará na rejeição de sua submissão.**

| Critério | Pontos |
|:---:|---|
| **3 pts** | Implementações corretas de atenção (Ex. 1) e Atenção Multi-Cabeça (Ex. 2) |
| **2 pts** | Bloco Transformer (Ex. 3): layer norm, FFN e conexões residuais corretas |
| **2 pts** | Codificação posicional (Ex. 4): implementação correta e visualizações |
| **2 pts** | Visualizações: heatmaps de atenção, plots de PE, heatmaps de representação de tokens |
| **1 pt** | Qualidade do relatório: explicações claras, notação matemática e discussão dos resultados |

**Formato de entrega:** GitHub Pages (usando o [template do curso](https://hsandmann.github.io/documentation.template/){:target="_blank"}). Nenhum outro formato aceito.

**Colaboração com IA:** Permitida, mas todo aluno deve ser capaz de explicar todo o código e análise. Provas orais podem ser realizadas.
