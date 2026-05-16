
!!! success inline end "Prazo e Entrega"

    :date: 21.set (domingo)
    
    :clock1: Commits até 23:59

    :material-account: Individual

    :simple-github: Enviar o Link do GitHub Pages (sim, **apenas** o link das pages) via [insper.blackboard.com](https://insper.blackboard.com){:target="_blank"}.

**Atividade: Entendendo Perceptrons de Múltiplas Camadas (MLPs)**

Esta atividade é projetada para testar suas habilidades em Perceptrons de Múltiplas Camadas (MLPs).

***

## Exercício 1: Cálculo Manual das Etapas do MLP

Considere um MLP simples com 2 features de entrada, 1 camada oculta contendo 2 neurônios e 1 neurônio de saída. Use a tangente hiperbólica (tanh) como ativação tanto para a camada oculta quanto para a camada de saída. A função de perda é o erro quadrático médio (MSE): \( L = \frac{1}{N} (y - \hat{y})^2 \).

Para este exercício, use os seguintes valores específicos:

- Vetores de entrada e saída:

    \( \mathbf{x} = [0.5, -0.2] \)

    \( y = 1.0 \)

- Pesos da camada oculta:

    \( \mathbf{W}^{(1)} = \begin{bmatrix} 0.3 & -0.1 \\ 0.2 & 0.4 \end{bmatrix} \)

- Biases da camada oculta:

    \( \mathbf{b}^{(1)} = [0.1, -0.2] \)

- Pesos da camada de saída:

    \( \mathbf{W}^{(2)} = [0.5, -0.3] \)

- Bias da camada de saída:

    \( b^{(2)} = 0.2 \)

- Taxa de aprendizado: \( \eta = 0.3 \)

- Função de ativação: \( \tanh \)

Realize as seguintes etapas explicitamente, mostrando todas as derivações matemáticas e cálculos:

1. **Passagem Direta**:

    - Compute as pré-ativações da camada oculta: \( \mathbf{z}^{(1)} = \mathbf{W}^{(1)} \mathbf{x} + \mathbf{b}^{(1)} \).
    - Aplique tanh para obter as ativações ocultas: \( \mathbf{h}^{(1)} = \tanh(\mathbf{z}^{(1)}) \).
    - Compute a pré-ativação de saída: \( u^{(2)} = \mathbf{W}^{(2)} \mathbf{h}^{(1)} + b^{(2)} \).
    - Compute a saída final: \( \hat{y} = \tanh(u^{(2)}) \).

2. **Cálculo da Perda**:

    - Compute a perda MSE:

        \( L = \frac{1}{N} (y - \hat{y})^2 \).

3. **Passagem Reversa (Retropropagação)**: Compute os gradientes da perda em relação a todos os pesos e biases. Comece com \( \displaystyle \frac{\partial L}{\partial \hat{y}} \), depois compute:

    - \( \displaystyle \frac{\partial L}{\partial u^{(2)}} \) (usando a derivada do tanh: \( \displaystyle \frac{d}{du} \tanh(u) = 1 - \tanh^2(u) \)).
    - Gradientes para a camada de saída: \( \displaystyle \frac{\partial L}{\partial \mathbf{W}^{(2)}} \), \( \displaystyle \frac{\partial L}{\partial b^{(2)}} \).
    - Propague para a camada oculta: \( \displaystyle \frac{\partial L}{\partial \mathbf{h}^{(1)}} \), \( \displaystyle \frac{\partial L}{\partial \mathbf{z}^{(1)}} \).
    - Gradientes para a camada oculta: \( \displaystyle \frac{\partial L}{\partial \mathbf{W}^{(1)}} \), \( \displaystyle \frac{\partial L}{\partial \mathbf{b}^{(1)}} \).
    
    Mostre todos os passos intermediários e cálculos.

4. **Atualização dos Parâmetros**: Usando a taxa de aprendizado \( \eta = 0.1 \), atualize todos os pesos e biases via gradiente descendente. Forneça os valores numéricos para todos os parâmetros atualizados.

***

## Exercício 2: Classificação Binária com Dados Sintéticos e MLP do Zero

Usando a função `make_classification` do scikit-learn, gere um dataset sintético e implemente um MLP do zero (sem usar bibliotecas como TensorFlow ou PyTorch para o modelo; você pode usar NumPy para operações de array) para classificar esses dados.

Etapas:

1. Gere e divida os dados em conjuntos de treino (80%) e teste (20%).
2. Implemente a passagem direta, cálculo da perda, passagem reversa e atualizações de parâmetros em código.
3. Treine o modelo por um número razoável de épocas (ex: 100-500), acompanhando a perda de treinamento.
4. Avalie no conjunto de teste: Reporte a acurácia e opcionalmente plote fronteiras de decisão ou matriz de confusão.

***

## Exercício 3: Classificação Multiclasse com MLP Reutilizável

Similar ao Exercício 2, mas com maior complexidade. Use `make_classification` para gerar um dataset com 3 classes. Implemente um MLP do zero para classificar esses dados. Para ponto extra, reutilize exatamente o mesmo código MLP do Exercício 2, modificando apenas hiperparâmetros (ex: tamanho da camada de saída para 3 classes).

***

## Exercício 4: Classificação Multiclasse com MLP Mais Profundo

Repita o Exercício 3 exatamente, mas agora garanta que seu MLP tenha **pelo menos 2 camadas ocultas**. Reutilize código do Exercício 3 onde possível.

***


## **Critérios de Avaliação**

!!! failure "Uso de Toolboxes"

    Você pode usar toolboxes (ex: NumPy) ==APENAS para operações matriciais e cálculos== durante esta atividade. Todos os outros cálculos, incluindo funções de ativação, cálculos de perda, gradientes e a passagem direta, ==**DEVEM SER IMPLEMENTADOS** dentro do seu MLP==. O uso de ==bibliotecas de terceiros para a implementação do MLP **É ESTRITAMENTE PROIBIDO**==.

    **O descumprimento destas instruções resultará na rejeição de sua submissão.**

**Notas Importantes:**

- O entregável deve ser enviado em **GitHub Pages**. Existe um template do curso — [template](https://hsandmann.github.io/documentation.template/){target='_blank'};
- **O prazo não é estendido** — **NENHUMA EXCEÇÃO** para entregas atrasadas.
- **Colaboração com IA é permitida**, mas o aluno **DEVE ENTENDER** e explicar todo o código. **PROVAS ORAIS** podem ser realizadas.

**Critérios de Nota:**

- **Exercício 1 (2 pontos)**:
    - Passagem direta totalmente explícita (0,5 pontos)
    - Perda e passagem reversa com todos os gradientes derivados (1 ponto)
    - Atualizações de parâmetros mostradas corretamente (0,5 ponto)

- **Exercício 2 (3 pontos)**:
    - Geração de dados correta e divisão (0,5 pontos)
    - Implementação funcional do MLP do zero (2 pontos)
    - Treinamento, avaliação e resultados reportados (0,5 pontos)

- **Exercício 3 (2 pontos + 1 extra)**:
    - Geração de dados correta e divisão (0,5 pontos)
    - MLP funcional para multiclasse (1,5 pontos)
    - Ponto extra: Reutilização exata do código MLP do Exercício 2 (1 ponto, opcional)

- **Exercício 4 (2 pontos)**:
    - Adaptação bem-sucedida com pelo menos 2 camadas ocultas (1 ponto)
    - Resultados de treinamento e avaliação mostrando funcionalidade (1 ponto)
