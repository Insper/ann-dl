O pré-processamento de dados é uma fase crítica no desenvolvimento de modelos de redes neurais, garantindo que os dados brutos sejam transformados em um formato adequado para treinamento e inferência eficazes. Redes neurais são **altamente sensíveis à qualidade e ao formato dos dados de entrada**. Dados mal preparados podem levar a convergência lenta, overfitting ou acurácia subótima. O pré-processamento mitiga esses problemas abordando ruído, inconsistências e incompatibilidades estruturais nos conjuntos de dados.

## Tarefas Típicas de Pré-processamento

| Tarefa | Descrição |
|------|-------------|
| **Limpeza de Texto** | Remover caracteres indesejados, stop words e realizar stemming/lematização. |
| **Normalização** | Padronizar formatos de texto, como formatos de data e moeda. |
| **Tokenização** | Dividir texto em palavras ou subpalavras para análise mais fácil. |
| **Extração de Features** | Converter texto em features numéricas usando técnicas como TF-IDF ou embeddings. |
| **Aumento de Dados** | Gerar dados sintéticos para aumentar o tamanho e a diversidade do conjunto de dados. |

Um dataset típico para tarefas de aprendizado de máquina pode incluir colunas de diferentes tipos de dados — numéricos, categóricos e texto:

```python exec="on" html="0"
--8<-- "docs/2026.2/classes/preprocessing/titanic-original.py"
```
/// caption
Linhas de amostra do dataset Titanic
///


## Limpeza de Dados

A limpeza de dados envolve identificar e corrigir erros, inconsistências e valores faltantes no conjunto de dados. Valores faltantes, comuns em dados do mundo real, podem ser tratados por métodos de imputação como substituição pela média, mediana ou moda, ou removendo linhas/colunas afetadas se a perda for mínima. Outliers, que podem distorcer o treinamento de redes neurais, são detectados usando métodos estatísticos como z-scores ou intervalos interquartis. Dados inconsistentes, como formatos variados em texto, requerem padronização para garantir uniformidade.

=== "Resultado"

    ```python exec="on" html="0"
    --8<-- "docs/2026.2/classes/preprocessing/titanic-fill-nan.py"
    ```
    
=== "Código"

    ```python
    --8<-- "docs/2026.2/classes/preprocessing/titanic-fill-nan.py"
    ```

## Codificação de Variáveis Categóricas

Dados categóricos, por natureza não numéricos, devem ser convertidos para a entrada de redes neurais. A codificação one-hot cria vetores binários para cada categoria, ex: transformando cores `['vermelho', 'azul', 'verde']` em `[[1,0,0], [0,1,0], [0,0,1]]`. Isso evita premissas ordinais mas aumenta a dimensionalidade. A codificação de rótulos atribui inteiros (ex: 0 para "vermelho", 1 para "azul"), adequada para categorias ordinais mas arriscada para categorias nominais devido à ordenação implícita.

=== "Resultado"

    ```python exec="on" html="0"
    --8<-- "docs/2026.2/classes/preprocessing/titanic-preprocessing.py"
    ```

=== "Código"

    ```python
    --8<-- "docs/2026.2/classes/preprocessing/titanic-preprocessing.py"
    ```


## Normalização e Padronização

A normalização escala features para um intervalo limitado, tipicamente $[0, 1]$, usando escalonamento min-max:

$$x' = \displaystyle \frac{x - \min(x)}{\max(x) - \min(x)}$$

Isso é crucial para redes neurais que empregam ativações sigmoid ou tanh, pois evita saturação.

A padronização (normalização z-score) transforma os dados para ter média $0$ e desvio padrão $1$:

$$x' = \frac{x - \mu}{\sigma}$$

onde $\mu$ é a média e $\sigma$ o desvio padrão. É preferida para redes com ativações ReLU ou quando as distribuições dos dados são similares à Gaussiana.

---

--8<-- "docs/2026.2/classes/preprocessing/quiz.pt.md"
