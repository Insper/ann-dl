
!!! success inline end "Prazo e Entrega"

    :date: 05.set (sexta)
    
    :clock1: Commits até 23:59

    :material-account: Individual

    :simple-github: Enviar o Link do GitHub Pages (sim, **apenas** o link das pages) via [insper.blackboard.com](https://insper.blackboard.com){:target="_blank"}.


**Atividade: Preparação e Análise de Dados para Redes Neurais**

Esta atividade é projetada para testar suas habilidades em geração de datasets sintéticos, tratamento de desafios de dados do mundo real e preparação de dados para serem alimentados em **redes neurais**.

***

## Exercício 1

### **Explorando a Separabilidade de Classes em 2D**

Entender como os dados estão distribuídos é o primeiro passo antes de projetar uma arquitetura de rede. Neste exercício, você irá gerar e visualizar um dataset bidimensional para explorar como a distribuição de dados afeta a complexidade das fronteiras de decisão que uma rede neural precisaria aprender.

### **Instruções**

1.  **Gere os Dados:** Crie um dataset sintético com 400 amostras no total, divididas igualmente entre 4 classes (100 amostras cada). Use uma distribuição gaussiana para gerar os pontos de cada classe com os seguintes parâmetros:
    * **Classe 0:** Média = $[2, 3]$, Desvio Padrão = $[0.8, 2.5]$
    * **Classe 1:** Média = $[5, 6]$, Desvio Padrão = $[1.2, 1.9]$
    * **Classe 2:** Média = $[8, 1]$, Desvio Padrão = $[0.9, 0.9]$
    * **Classe 3:** Média = $[15, 4]$, Desvio Padrão = $[0.5, 2.0]$
1.  **Plote os Dados:** Crie um scatter plot 2D mostrando todos os pontos. Use uma cor diferente para cada classe para torná-las distinguíveis.
1.  **Analise e Desenhe Fronteiras:**
    1. Examine o scatter plot cuidadosamente. Descreva a distribuição e a sobreposição das quatro classes.
    1. Com base em sua inspeção visual, uma fronteira linear simples poderia separar todas as classes?
    1. No seu plot, esboce as fronteiras de decisão que você acha que uma rede neural treinada poderia aprender para separar essas classes.

***

## Exercício 2

### **Não-Linearidade em Dimensões Maiores**

Redes neurais simples (como um Perceptron) só conseguem aprender fronteiras lineares. Redes profundas se destacam quando os dados não são linearmente separáveis. Este exercício desafia você a criar e visualizar tal dataset.

### **Instruções**

1.  **Gere os Dados:** Crie um dataset com 500 amostras para a Classe A e 500 amostras para a Classe B. Use uma distribuição normal multivariada com os seguintes parâmetros:

    * **Classe A:**

        Vetor médio:

        $$\mu_A = [0, 0, 0, 0, 0]$$

        Matriz de covariância:

        $$
        \Sigma_A = \begin{pmatrix}
        1.0 & 0.8 & 0.1 & 0.0 & 0.0 \\
        0.8 & 1.0 & 0.3 & 0.0 & 0.0 \\
        0.1 & 0.3 & 1.0 & 0.5 & 0.0 \\
        0.0 & 0.0 & 0.5 & 1.0 & 0.2 \\
        0.0 & 0.0 & 0.0 & 0.2 & 1.0
        \end{pmatrix}
        $$

    * **Classe B:**

        Vetor médio:
            
        $$\mu_B = [1.5, 1.5, 1.5, 1.5, 1.5]$$
        
        Matriz de covariância:

        $$
        \Sigma_B = \begin{pmatrix}
        1.5 & -0.7 & 0.2 & 0.0 & 0.0 \\
        -0.7 & 1.5 & 0.4 & 0.0 & 0.0 \\
        0.2 & 0.4 & 1.5 & 0.6 & 0.0 \\
        0.0 & 0.0 & 0.6 & 1.5 & 0.3 \\
        0.0 & 0.0 & 0.0 & 0.3 & 1.5
        \end{pmatrix}
        $$

1.  **Visualize os Dados:** Como você não pode plotar diretamente um gráfico 5D, você deve reduzir a dimensionalidade.
    * Use uma técnica como **Análise de Componentes Principais (PCA)** para projetar os dados 5D em 2 dimensões.
    * Crie um scatter plot desta representação 2D, colorindo os pontos por classe (A ou B).
1.  **Analise os Plots:**
    1. Com base em sua projeção 2D, descreva a relação entre as duas classes.
    1. Discuta a **separabilidade linear** dos dados. Explique por que este tipo de estrutura de dados representa um desafio para modelos lineares simples e provavelmente exigiria uma rede neural de múltiplas camadas com funções de ativação não-lineares para ser classificada com precisão.

***

## Exercício 3

### **Preparando Dados do Mundo Real para uma Rede Neural**

Este exercício usa um dataset real do Kaggle. Sua tarefa é realizar o pré-processamento necessário para torná-lo adequado para uma rede neural que usa a função de ativação tangente hiperbólica (`tanh`) em suas camadas ocultas.

### **Instruções**

1.  **Obtenha os Dados:** Baixe o dataset [**Spaceship Titanic**](https://www.kaggle.com/competitions/spaceship-titanic){:target="_blank"} do Kaggle.
2.  **Descreva os Dados:**
    * Descreva brevemente o objetivo do dataset (ou seja, o que a coluna `Transported` representa?).
    * Liste as features e identifique quais são **numéricas** (ex: `Age`, `RoomService`) e quais são **categóricas** (ex: `HomePlanet`, `Destination`).
    * Investigue o dataset em busca de **valores faltantes**. Quais colunas os têm, e quantos?
3.  **Pré-processe os Dados:** Seu objetivo é limpar e transformar os dados para que possam ser alimentados em uma rede neural. A função de ativação `tanh` produz saídas no intervalo `[-1, 1]`, então seus dados de entrada devem ser escalados adequadamente.
    * **Trate os Dados Faltantes:** Defina e implemente uma estratégia para tratar os valores faltantes em todas as colunas afetadas. Justifique suas escolhas.
    * **Codifique Features Categóricas:** Converta colunas categóricas como `HomePlanet`, `CryoSleep` e `Destination` em formato numérico. One-hot encoding é uma boa escolha.
    * **Normalize/Padronize Features Numéricas:** Escale as colunas numéricas (ex: `Age`, `RoomService`). Como a função `tanh` é centrada em zero, **Padronização** (média 0, desvio 1) ou **Normalização** para o intervalo `[-1, 1]` são excelentes escolhas. Implemente uma e explique por que é uma boa prática.
4.  **Visualize os Resultados:**
    * Crie histogramas para uma ou duas features numéricas (como `FoodCourt` ou `Age`) **antes** e **depois** do escalonamento para mostrar o efeito da transformação.

***

## **Critérios de Avaliação**

O entregável desta atividade consiste em um **relatório** que inclui:

1. Uma breve descrição de sua abordagem a cada exercício.
1. O código usado para gerar os datasets, pré-processar os dados e criar as visualizações, com comentários explicando cada etapa.
1. Os plots e visualizações solicitados em cada exercício.
1. Sua análise e respostas às perguntas colocadas em cada exercício.

**Notas Importantes:**

- O entregável deve ser enviado no formato especificado: **GitHub Pages**. **Nenhum outro formato será aceito.** Existe um template do curso que você pode usar — [template](https://hsandmann.github.io/documentation.template/){target='_blank'};

- Há uma **política estrita contra plágio**. Qualquer forma de plágio resultará em nota zero para a atividade;

- **O prazo para cada atividade não é estendido** — **NENHUMA EXCEÇÃO** será feita para entregas atrasadas.

- **Colaboração com IA é permitida**, mas cada aluno **DEVE ENTENDER** e ser capaz de explicar todas as partes do código e análise enviados. Qualquer uso de ferramentas de IA deve ser citado adequadamente. ^^**PROVAS ORAIS**^^ podem ser realizadas.

**Critérios de Nota:**

**Exercício 1 (3 pontos):**

| Critério | Descrição |
|:--------:|-------------|
| **1 pt** | Os dados são gerados corretamente e visualizados em um scatter plot claro com rótulos e cores adequados. |
| **2 pts** | A análise da separabilidade de classes é precisa, e as fronteiras de decisão propostas são lógicas e bem explicadas. |

**Exercício 2 (3 pontos):**

| Critério | Descrição |
|:--------:|-------------|
| **1 pt** | Os dados são gerados corretamente usando os parâmetros multivariados especificados. |
| **1 pt** | A redução de dimensionalidade é aplicada corretamente e a projeção 2D é claramente plotada. |
| **1 pt** | A análise identifica corretamente a relação não-linear e explica por que uma rede neural seria um modelo adequado. |

**Exercício 3 (4 pontos):**

| Critério | Descrição |
|:--------:|-------------|
| **1 pt** | Os dados são carregados corretamente e suas características são descritas com precisão. |
| **2 pts** | Todas as etapas de pré-processamento são implementadas corretamente e com justificativa clara. |
| **1 pt** | As visualizações demonstram efetivamente o impacto do pré-processamento dos dados. |
