!!! success inline end "Prazo e Entrega"

    :date: 14.set (domingo)
    
    :clock1: Commits até 23:59

    :material-account: Individual

    :simple-github: Enviar o Link do GitHub Pages (sim, **apenas** o link das pages) via [insper.blackboard.com](https://insper.blackboard.com){:target="_blank"}.

**Atividade: Entendendo Perceptrons e Suas Limitações**

Esta atividade é projetada para testar suas habilidades em Perceptrons e suas limitações.

***

## Exercício 1

### **Tarefa de Geração de Dados:**

Gere duas classes de pontos de dados 2D (1000 amostras por classe) usando distribuições normais multivariadas. Use os seguintes parâmetros:

- Classe 0:

    Média = $[1.5, 1.5]$,
    
    Matriz de covariância = $[[0.5, 0], [0, 0.5]]$ (variância de $0.5$ ao longo de cada dimensão, sem covariância).

- Classe 1:

    Média = $[5, 5]$,
    
    Matriz de covariância = $[[0.5, 0], [0, 0.5]]$.

Esses parâmetros garantem que as classes sejam quase linearmente separáveis, com sobreposição mínima. Plote os pontos de dados (usando bibliotecas como matplotlib) para visualizar a separação, colorindo os pontos por classe.

### **Tarefa de Implementação do Perceptron:**

Implemente um perceptron de camada única do zero para classificar os dados gerados em duas classes. Você pode usar apenas NumPy para operações básicas de álgebra linear. Não use nenhuma biblioteca de aprendizado de máquina pré-construída (ex: sem scikit-learn).

- Inicialize os pesos (w) como um vetor 2D (mais um termo de bias b).
- Use a regra de aprendizado do perceptron: Para cada amostra mal classificada $(x, y)$, atualize $w = w + η * y * x$ e $b = b + η * y$, onde $η$ é a taxa de aprendizado (comece com $η=0.01$).
- Treine o modelo até a convergência (nenhuma atualização de peso ocorre em uma passagem completa pelo dataset) ou por no máximo 100 épocas. Acompanhe a acurácia após cada época.
- Após o treinamento, avalie a acurácia no dataset completo e plote a fronteira de decisão (linha definida por $w·x + b = 0$) sobreposta nos pontos de dados. Plote também a acurácia de treinamento ao longo das épocas.

Reporte os pesos finais, o bias, a acurácia e discuta por que a separabilidade dos dados leva à convergência rápida.

## Exercício 2

### **Tarefa de Geração de Dados:**

Gere duas classes de pontos de dados 2D (1000 amostras por classe) usando distribuições normais multivariadas com os seguintes parâmetros:

- Classe 0:

    Média = $[3, 3]$,

    Matriz de covariância = $[[1.5, 0], [0, 1.5]]$ (variância maior de 1.5 ao longo de cada dimensão).

- Classe 1:

    Média = $[4, 4]$,

    Matriz de covariância = $[[1.5, 0], [0, 1.5]]$.

Esses parâmetros criam sobreposição parcial entre as classes, tornando os dados não totalmente linearmente separáveis. Plote os pontos de dados para visualizar a sobreposição, colorindo por classe.

### **Tarefa de Implementação do Perceptron:**

Usando as mesmas diretrizes de implementação do Exercício 1, treine um perceptron neste dataset. Avalie a acurácia após o treinamento e plote a fronteira de decisão. Reporte os pesos finais, o bias, a acurácia e discuta como a sobreposição afeta o treinamento em comparação com o Exercício 1.

## **Critérios de Avaliação**

!!! failure "Uso de Toolboxes"

    Você pode usar toolboxes (ex: NumPy) ==APENAS para operações matriciais e cálculos== durante esta atividade. Todos os outros cálculos, incluindo funções de ativação, cálculos de perda, gradientes e a passagem direta, ==**DEVEM SER IMPLEMENTADOS** dentro do seu Perceptron==. O uso de ==bibliotecas de terceiros para a implementação do Perceptron **É ESTRITAMENTE PROIBIDO**==.

    **O descumprimento destas instruções resultará na rejeição de sua submissão.**

O entregável desta atividade consiste em um **relatório** que inclui:

- Uma breve descrição de sua abordagem de implementação e quaisquer desafios enfrentados;
- Os pesos e bias finais após o treinamento para ambos os exercícios;
- A acurácia alcançada nos conjuntos de treino e validação;
- Visualizações da fronteira de decisão e distribuição dos dados;
- Um plot mostrando a acurácia de treinamento ao longo das épocas para ambos os exercícios;
- Discussão sobre as diferenças entre os dois exercícios, focando no impacto da separabilidade dos dados.

**Notas Importantes:**

- O entregável deve ser enviado no formato especificado: **GitHub Pages**. Existe um template do curso — [template](https://hsandmann.github.io/documentation.template/){target='_blank'};
- **O prazo não é estendido** — **NENHUMA EXCEÇÃO** para entregas atrasadas.
- **Colaboração com IA é permitida**, mas o aluno **DEVE ENTENDER** e explicar todo o código. **PROVAS ORAIS** podem ser realizadas.

**Critérios de Nota:**

| Critério | Descrição |
|:--------:|-------------|
| **4 pts** | Correção da implementação do perceptron |
| **2 pts** | Exercício 1: Geração de dados, treinamento, avaliação e análise. |
| **2 pts** | Exercício 2: Geração de dados, treinamento, avaliação e análise. |
| **1 pt** | Visualizações: Qualidade e clareza dos plots. |
| **1 pt** | Qualidade do Relatório: Clareza, organização e completude. |
