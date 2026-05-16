
!!! success inline end "Prazo e Entrega"

    :date: 05.out (domingo)
    
    :clock1: Commits até 23:59

    :material-account-group: [Equipe (2-3 membros) - formulário](https://forms.gle/Rrb9b3dJcHTUHbsK6){:target="_blank"}

    :simple-github: Enviar o Link do GitHub Pages via [insper.blackboard.com](https://insper.blackboard.com){:target="_blank"}.


Neste projeto, você irá abordar uma tarefa de classificação do mundo real usando uma rede neural Perceptron de Múltiplas Camadas (MLP). O objetivo é aprofundar seu entendimento sobre redes neurais lidando com preparação de dados, implementação do modelo, estratégias de treinamento e avaliação sem depender de bibliotecas de aprendizado profundo de alto nível.

!!! example "Bônus de Competição"

    Este projeto encoraja a criatividade na seleção de dataset e recompensa a ambição — pontos bônus serão concedidos se você enviar sua solução para uma competição online relevante (ex: plataformas como [Kaggle](https://www.kaggle.com){:target="_blank"}, [DrivenData](https://www.drivendata.org){:target="_blank"} ou [Zindi](https://zindi.africa){:target="_blank"}). As submissões devem ser documentadas no relatório, incluindo um link para sua entrada e qualquer posição no leaderboard. Pontos bônus:
    
    | Pontos | Descrição |
    |:--------:|-------------|
    | +0,5 | Submissão válida para uma competição reconhecida (prova necessária, ex: link, screenshot). |
    | +0,5 | Submissão válida com classificação no top 50% do leaderboard (prova necessária). |

!!! danger "Restrições Importantes"

    - **NÃO USE** os datasets Titanic, Iris, Wine ou outros datasets **clássicos**. Esses são muito usados e resultarão em nota zero para a parte de seleção do dataset.
    - A tarefa deve ser de classificação (ex: binária, multiclasse ou multi-rótulo).
    - Você pode implementar o MLP você mesmo ou usar bibliotecas de alto nível como TensorFlow, PyTorch, Keras ou os módulos de rede neural do scikit-learn — estes são **PERMITIDOS**. Mas você **TEM QUE** entender e explicar todas as partes do código e análise enviados.
    - O dataset deve ter pelo menos 1.000 amostras e múltiplas features (pelo menos 5).


## Etapas do Projeto

Siga estas etapas em seu trabalho. Seu relatório deve abordar cada uma explicitamente.

### 1. **Seleção do Dataset**

- Escolha um dataset público para um problema de classificação. Fontes incluem:
    - Kaggle (ex: datasets para reconhecimento de dígitos, detecção de spam ou diagnóstico médico).
    - UCI Machine Learning Repository (ex: Banknote Authentication, Adult Income, ou Covertype).
    - Outras fontes abertas como OpenML, Google Dataset Search, ou portais de dados governamentais.

- Garanta que o dataset tenha pelo menos 1.000 amostras e múltiplas features (pelo menos 5).
- No seu relatório: Forneça o nome do dataset, URL de origem, tamanho (linhas/colunas) e por que você o escolheu.

### 2. **Explicação do Dataset**

- Descreva o dataset em detalhes: O que ele representa? Quais são as features e seus tipos? Qual é a variável alvo?
- Discuta qualquer conhecimento de domínio relevante.
- Identifique possíveis problemas: Classes desbalanceadas, valores faltantes, outliers ou ruído.
- No seu relatório: Inclua estatísticas resumidas e visualizações (ex: histogramas, matrizes de correlação).

### 3. **Limpeza e Normalização dos Dados**

- Limpe os dados: Trate valores faltantes, remova duplicatas, detecte e trate outliers.
- Pré-processe: Codifique variáveis categóricas, normalize/escale features numéricas.
- No seu relatório: Explique cada etapa, justifique suas escolhas e mostre exemplos antes/depois.

### 4. **Implementação do MLP**

- Código de um MLP usando apenas NumPy (ou equivalente) para operações como multiplicação de matrizes, funções de ativação e gradientes.
- Arquitetura: Inclua pelo menos uma camada de entrada, uma camada oculta e uma camada de saída.
- Função de perda: Entropia cruzada para classificação.
- Otimizador: SGD ou variante como mini-batch GD.
- No seu relatório: Forneça trechos de código chave. Explique os hiperparâmetros.

### 5. **Treinamento do Modelo**

- Treine seu MLP nos dados preparados.
- Implemente o loop de treinamento: Propagação direta, cálculo da perda, retropropagação e atualizações de parâmetros.
- No seu relatório: Descreva o processo de treinamento, incluindo quaisquer desafios e como foram abordados.

### 6. **Estratégia de Treinamento e Teste**

- Divida os dados: Use conjuntos treino/validação/teste (ex: divisão 70/15/15) ou validação cruzada k-fold.
- Modo de treinamento: Escolha batch, mini-batch ou online (estocástico); explique o motivo.
- Early stopping ou outras técnicas para prevenir overfitting.

### 7. **Curvas de Erro e Visualização**

- Plote curvas de perda/acurácia de treino e validação ao longo das épocas.
- Analise: Discuta convergência, overfitting/underfitting e ajustes feitos.
- No seu relatório: Inclua pelo menos dois plots. Interprete as tendências.

### 8. **Métricas de Avaliação**

- Aplique métricas de classificação no conjunto de teste: Acurácia, precisão, recall, F1-score, matriz de confusão.
- Se desbalanceado, inclua ROC-AUC ou curvas precisão-recall.
- Compare com baselines (ex: preditor de classe majoritária).

***

## **Critérios de Avaliação**

O entregável consiste em um **relatório** abrangente que inclui:

- **Seções**: Uma para cada etapa acima (1-8).
- **Conclusão**: Descobertas gerais, limitações, melhorias futuras.
- **Referências**: Cite fontes do dataset, artigos sobre MLPs, etc.

**Notas Importantes:**

- O entregável deve ser enviado em **GitHub Pages**. Existe um template do curso — [template](https://hsandmann.github.io/documentation.template/){target='_blank'};
- **O prazo não é estendido** — **NENHUMA EXCEÇÃO** para entregas atrasadas.
- **Colaboração com IA é permitida**, mas o aluno **DEVE ENTENDER** e explicar todo o código. ^^**PROVAS ORAIS**^^ podem ser realizadas.

**Rubrica de Avaliação** (de 10 pontos):

| Critério | Descrição |
|:--------:|-------------|
| **2 pts** | Seleção e Explicação do Dataset: 1 ponto<br>Limpeza/Normalização dos Dados: 1 ponto |
| **6 pts** | Implementação do MLP: 2 pontos (correção e originalidade);<br>Treinamento e Estratégia: 1,5 pontos;<br>Curvas de Erro: 1 ponto;<br>Métricas e Análise: 1,5 pontos |
| **2 pts** | Qualidade do Relatório (clareza, estrutura, visuais): 1 ponto;<br>Bônus: Até +1 por submissão em competição. |

Este projeto testará suas habilidades de aprendizado de máquina de ponta a ponta.
