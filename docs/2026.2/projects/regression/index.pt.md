
!!! success inline end "Prazo e Entrega"

    :date: 19.out (domingo)
    
    :clock1: Commits até 23:59

    :material-account-group: [Equipe (2-3 membros) - formulário](https://forms.gle/BmzmJodCiMTX8gF79){:target="_blank"}

    :simple-github: Enviar o Link do GitHub Pages via [insper.blackboard.com](https://insper.blackboard.com){:target="_blank"}.

Neste projeto, você irá abordar uma tarefa de regressão do mundo real usando uma rede neural Perceptron de Múltiplas Camadas (MLP). O objetivo é aprofundar seu entendimento sobre redes neurais lidando com preparação de dados, implementação do modelo, estratégias de treinamento e avaliação.

!!! example "Bônus de Competição"

    Pontos bônus serão concedidos se você enviar sua solução para uma competição online relevante (ex: [Kaggle](https://www.kaggle.com){:target="_blank"}, [DrivenData](https://www.drivendata.org){:target="_blank"} ou [Zindi](https://zindi.africa){:target="_blank"}).
    
    | Pontos | Descrição |
    |:--------:|-------------|
    | +0,5 | Submissão válida para uma competição reconhecida (prova necessária). |
    | +0,5 | Submissão válida com classificação no top 50% do leaderboard (prova necessária). |

!!! danger "Restrições Importantes"

    - **NÃO USE** os datasets Boston Housing, California Housing ou outros **clássicos**.
    - A tarefa deve ser de regressão (previsão de valores contínuos).
    - Você pode usar bibliotecas de alto nível, mas **TEM QUE** entender e explicar todas as partes do código.
    - O dataset deve ter pelo menos 1.000 amostras e múltiplas features (pelo menos 5).

## Etapas do Projeto

### 1. **Seleção do Dataset**

- Escolha um dataset público para um problema de regressão. Fontes incluem:
    - Kaggle (ex: previsão de preços de imóveis, ações ou consumo de energia).
    - UCI Machine Learning Repository (ex: Concrete Compressive Strength, Air Quality, ou Wine Quality).
    - Outras fontes abertas.
- No seu relatório: Forneça o nome do dataset, URL de origem, tamanho e justificativa.

### 2. **Explicação do Dataset**

- Descreva o dataset em detalhes: O que representa? Quais são as features? Qual é a variável alvo (valor contínuo)?
- Identifique possíveis problemas: Valores faltantes, outliers ou ruído.
- No seu relatório: Inclua estatísticas resumidas e visualizações.

### 3. **Limpeza e Normalização dos Dados**

- Trate valores faltantes, remova duplicatas e detecte outliers.
- Codifique variáveis categóricas e normalize/escale features numéricas.
- No seu relatório: Justifique cada escolha e mostre exemplos antes/depois.

### 4. **Implementação do MLP**

- Implemente um MLP para regressão.
- Função de perda: MSE ou MAE para regressão.
- No seu relatório: Forneça trechos de código chave e explique os hiperparâmetros.

### 5. **Treinamento do Modelo**

- Treine seu MLP com propagação direta, cálculo da perda, retropropagação e atualizações de parâmetros.
- No seu relatório: Descreva o processo de treinamento e quaisquer desafios.

### 6. **Estratégia de Treinamento e Teste**

- Divida os dados (ex: 70/15/15) ou use validação cruzada k-fold.
- Early stopping ou outras técnicas para prevenir overfitting.

### 7. **Curvas de Erro e Visualização**

- Plote curvas de perda de treino e validação ao longo das épocas.
- Analise convergência, overfitting/underfitting e ajustes feitos.

### 8. **Métricas de Avaliação**

- Aplique métricas de regressão: MAE, MAPE, MSE, RMSE e R².
- Compare com baselines (ex: preditor de média).
- No seu relatório: Apresente resultados em tabelas e visualizações (ex: plots de resíduos).

***

## **Critérios de Avaliação**

O entregável consiste em um **relatório** abrangente com seções para cada etapa (1-8), conclusão e referências.

**Notas Importantes:**

- Entregável em **GitHub Pages** — [template](https://hsandmann.github.io/documentation.template/){target='_blank'};
- **O prazo não é estendido** — **NENHUMA EXCEÇÃO** para entregas atrasadas.
- **Colaboração com IA é permitida**, mas o aluno **DEVE ENTENDER** e explicar todo o código. ^^**PROVAS ORAIS**^^ podem ser realizadas.

**Rubrica de Avaliação** (de 10 pontos):

| Critério | Descrição |
|:--------:|-------------|
| **2 pts** | Seleção e Explicação do Dataset: 1 ponto<br>Limpeza/Normalização dos Dados: 1 ponto |
| **6 pts** | Implementação do MLP: 2 pontos;<br>Treinamento e Estratégia: 1,5 pontos;<br>Curvas de Erro: 1 ponto;<br>Métricas e Análise: 1,5 pontos |
| **2 pts** | Qualidade do Relatório: 1 ponto;<br>Bônus: Até +1 por submissão em competição. |
