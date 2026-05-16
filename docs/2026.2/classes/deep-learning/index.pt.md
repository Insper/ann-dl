O aprendizado profundo é um subconjunto do aprendizado de máquina que se concentra em treinar redes neurais artificiais com múltiplas camadas para aprender e fazer previsões a partir de dados complexos. Essas redes são inspiradas na estrutura do cérebro humano, onde "neurônios" processam informações e as passam adiante.

Diferentemente dos algoritmos de AM tradicionais, que frequentemente requerem engenharia manual de features, os modelos de aprendizado profundo extraem automaticamente features dos dados brutos por meio de camadas de processamento. Isso os torna poderosos para tarefas como reconhecimento de imagem, processamento de linguagem natural, síntese de fala e muito mais.

O aprendizado profundo se destaca com grandes conjuntos de dados e alto poder computacional (ex: GPUs), mas pode ser uma "caixa-preta" — às vezes é difícil interpretar por que um modelo toma uma decisão específica.

O bloco construtivo central é a **rede neural artificial (ANN)**, que consiste em nós interconectados (neurônios) organizados em camadas. Os dados fluem da camada de entrada, através das camadas ocultas (onde está o "profundo"), até a camada de saída.

## Componentes Principais

Uma rede neural típica tem três partes principais:

- **Camada de Entrada**: O ponto de entrada onde os dados brutos (ex: valores de pixels de uma imagem) são alimentados na rede. Não realiza cálculos; apenas passa os dados adiante.
- **Camadas Ocultas**: A "profundidade" do aprendizado profundo. São onde a mágica acontece — múltiplas camadas empilhadas que transformam os dados por meio de operações matemáticas. Cada camada aprende representações progressivamente mais abstratas.
- **Camada de Saída**: A camada final que produz a previsão ou classificação.

## Diferentes Tipos de Camadas

Modelos de aprendizado profundo usam várias camadas especializadas dependendo da tarefa e arquitetura:

| Tipo de Camada | Descrição | Casos de Uso Comuns | Como Funciona |
|---|---|---|---|
| [**Densa (Totalmente Conectada)**](#a-dense-fully-connected) | Cada neurônio nesta camada é conectado a todos os neurônios da camada anterior. | Redes de propósito geral, como classificadores simples. | Aplica uma transformação linear (pesos × entradas + bias) seguida de uma função de ativação (ex: ReLU). |
| [**Convolucional**](#b-convolutional) | Usa filtros (kernels) para escanear dados de entrada, detectando padrões locais como bordas ou texturas. | Processamento de imagem e vídeo, visão computacional. | Desliza filtros sobre a entrada, calculando produtos internos para criar mapas de features. |
| [**Pooling**](#c-pooling-max-pooling) | Reduz a amostra da saída das camadas convolucionais, reduzindo a carga computacional. | Após camadas convolucionais em CNNs. | Agrega valores em pequenas regiões (ex: grade 2×2) em um único valor. |
| [**Recorrente**](#d-recurrent-lstm) (ex: RNN, LSTM, GRU) | Lida com dados sequenciais mantendo uma "memória" de entradas anteriores via loops. | Previsão de séries temporais, PLN, reconhecimento de fala. | Processa entradas um passo por vez, usando estados ocultos para transportar informações. |
| [**Embedding**](#e-embedding) | Converte dados categóricos (ex: palavras) em vetores densos de tamanho fixo. | PLN, sistemas de recomendação. | Mapeia dados esparsos de alta dimensionalidade para um espaço contínuo de menor dimensão. |
| [**Atenção**](#f-attention-scaled-dot-product) | Permite ao modelo focar em partes relevantes da entrada dinamicamente. | PLN moderno (ex: modelos GPT), tradução automática. | Usa queries, keys e values para calcular pontuações de atenção. |
| [**Normalização**](#g-normalization-batch-normalization) | Estabiliza o treinamento normalizando as ativações dentro de uma camada. | Praticamente todas as redes profundas. | Ajusta e escala ativações (ex: média para 0, variância para 1). |
| [**Dropout**](#h-dropout) | "Descarta" aleatoriamente uma fração de neurônios durante o treinamento para prevenir overfitting. | Regularização em qualquer rede. | Remove temporariamente conexões, forçando a rede a aprender representações redundantes. |
| [**Flatten**](#i-flatten) | Converte dados multidimensionais em um vetor 1D para camadas densas. | Transição de extração de features (CNN) para classificação. | Reformata tensores sem alterar os valores. |
| [**Ativação**](#j-activation) | Aplica uma função não-linear à saída de outras camadas. | Em toda parte, para adicionar não-linearidade. | Transforma saídas lineares; ex: ReLU define valores negativos como 0. |

## Arquiteturas Comuns de Aprendizado Profundo

Essas camadas são combinadas em arquiteturas adaptadas a problemas específicos:

- **Redes Neurais Feedforward (FNN)**: Pilha básica de camadas densas para tarefas simples.
- **Redes Neurais Convolucionais (CNN)**: Camadas convolucionais + pooling para dados espaciais como imagens (ex: ResNet, VGG).
- **Redes Neurais Recorrentes (RNN)**: Camadas recorrentes para sequências (ex: LSTM para geração de texto).
- **Transformers**: Camadas de atenção para lidar com dependências de longo alcance (ex: BERT para PLN, Vision Transformers para imagens).
- **Autoencoders**: Encoder + decoder para aprendizado não supervisionado como denoising.
- **GANs**: Combina redes geradoras e discriminadoras para gerar dados realistas.

## Passagem Direta e Reversa para Cada Camada

A passagem direta calcula a saída de cada camada dado o input, enquanto a passagem reversa calcula gradientes para aprendizado.

A retropropagação calcula o gradiente da perda em relação às entradas e parâmetros de cada camada para atualizá-los via otimizadores como o gradiente descendente.

---

### A. Densa (Totalmente Conectada)

--8<-- "docs/2026.2/classes/deep-learning/dense.md"

---

### B. Convolucional

--8<-- "docs/2026.2/classes/deep-learning/convolutional.md"

---

### C. Pooling (Max Pooling)

Reduz a amostra da saída das camadas convolucionais, reduzindo a carga computacional e prevenindo overfitting. Segue as camadas convolucionais em CNNs para resumir features. Agrega valores em pequenas regiões (ex: grade 2×2) em um único valor, tornando o modelo mais robusto a variações como translações.

**Passagem Direta**:

<div class="grid cards" markdown>

-   $Y[i,j] = \max(X[i:i+k, j:j+k])$ para tamanho de pool $k$.

-   $X = \begin{bmatrix} 1 & 2 & 3 & 4 \\ 5 & 6 & 7 & 8 \\ 9 & 10 & 11 & 12 \\ 13 & 14 & 15 & 16 \end{bmatrix}$, pool=2, stride=2,

    $Y = \begin{bmatrix} 6 & 8 \\ 14 & 16 \end{bmatrix}$.

</div>

**Passagem Reversa**:

<div class="grid cards" markdown>

-   Distribua o gradiente upstream $\frac{\partial L}{\partial Y}$ para a posição máxima em cada janela; 0 nos demais.

-   $\frac{\partial L}{\partial Y} = \begin{bmatrix} 0.5 & -0.5 \\ 1 & 0 \end{bmatrix}$: 0,5 para a posição de 6 (1,1), -0,5 para 8 (1,3), etc.

</div>

**Implementação**:

```pyodide install="pandas,scikit-learn" exec="on" html="1"
--8<-- "docs/2026.2/classes/deep-learning/pooling.py"
```

---

### D. Recorrente (LSTM)

--8<-- "docs/2026.2/classes/deep-learning/lstm.md"

---

### E. Embedding

**Passagem Direta**: $y = E[i]$, onde $E$ é a matriz de embedding e $i$ é o índice de entrada.

**Passagem Reversa**: Gradiente $\frac{\partial L}{\partial E[i]} += \frac{\partial L}{\partial y}$; outras linhas 0. (Atualização esparsa).

**Implementação**:

```pyodide install="pandas,scikit-learn" exec="on" html="1"
--8<-- "docs/2026.2/classes/deep-learning/embedding.py"
```

---

### F. Atenção (Produto Escalar Escalado)

**Passagem Direta**: $\text{Attention} = \text{softmax}\left(\frac{QK^T}{\sqrt{d}}\right)V$.

**Passagem Reversa**: Gradientes para Q, K, V via regra da cadeia no softmax e multiplicações de matrizes.

**Implementação**:

```pyodide install="pandas,scikit-learn" exec="on" html="1"
--8<-- "docs/2026.2/classes/deep-learning/attention.py"
```

---

### G. Normalização (Batch Normalization)

**Passagem Direta**: Normalizar, escalar, deslocar.

**Passagem Reversa**: Gradientes para entrada, gamma, beta via regra da cadeia sobre média/variância.

**Implementação**:

```pyodide install="pandas,scikit-learn" exec="on" html="1"
--8<-- "docs/2026.2/classes/deep-learning/normalization.py"
```

---

### H. Dropout

--8<-- "docs/2026.2/classes/deep-learning/dropout.md"

---

### I. Flatten

**Passagem Direta**: Reformatar para 1D.

**Passagem Reversa**: Reformatar o gradiente upstream de volta ao formato original.

**Implementação**:

```pyodide install="pandas,scikit-learn" exec="on" html="1"
--8<-- "docs/2026.2/classes/deep-learning/flatten.py"
```

### J. Ativação (ReLU)

**Passagem Direta**: $y = \max(0, x)$.

**Passagem Reversa**: $\frac{\partial L}{\partial x} = \frac{\partial L}{\partial y} \cdot (x > 0)$.

**Implementação**:

```pyodide install="pandas,scikit-learn" exec="on" html="1"
--8<-- "docs/2026.2/classes/deep-learning/activation.py"
```


[^1]: Mohd Halim Mohd Noor, Ayokunle Olalekan Ige: [A Survey on State-of-the-art Deep Learning Applications and Challenges](https://arxiv.org/pdf/2403.17561){target='_blank'}, 2025.
[^2]: Aston Zhang et al.: [Dive into Deep Learning](https://d2l.ai/){target='_blank'}, 2020.
[^3]: Ian Goodfellow, Yoshua Bengio, Aaron Courville: [Deep Learning](https://www.deeplearningbook.org/){target='_blank'}, 2016.

---

--8<-- "docs/2026.2/classes/deep-learning/quiz.pt.md"
