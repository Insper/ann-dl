<div id="quiz-mlp"></div>
<script>
buildQuiz('mlp', 'Perceptron Multi-Camada', [
  {
    q: 'O que é a retropropagação (backpropagation)?',
    opts: [
      'Um algoritmo de passagem direta que calcula ativações',
      'Um algoritmo que calcula gradientes camada por camada usando a regra da cadeia para atualizar pesos',
      'Uma técnica de regularização que reverte pesos',
      'O processo de inicializar pesos aleatoriamente'
    ],
    ans: 1,
    exp: 'A retropropagação aplica a regra da cadeia do cálculo para propagar o gradiente da perda de volta pelas camadas, calculando ∂L/∂w para cada peso. É a base do treinamento de redes neurais.'
  },
  {
    q: 'O Teorema da Aproximação Universal afirma que:',
    opts: [
      'MLPs com infinitas camadas podem aproximar qualquer função',
      'Um MLP com uma única camada oculta com neurônios suficientes pode aproximar qualquer função contínua',
      'MLPs funcionam apenas para funções lineares',
      'Qualquer função pode ser aproximada por um único neurônio sigmoid'
    ],
    ans: 1,
    exp: 'Cybenko (1989) e Hornik (1991) provaram que MLPs com uma camada oculta e ativação não-linear podem aproximar qualquer função contínua com precisão arbitrária, dados neurônios suficientes.'
  },
  {
    q: 'Por que funções de ativação não-lineares são essenciais em MLPs?',
    opts: [
      'Para acelerar o treinamento',
      'Sem não-linearidade, compor camadas lineares colapsa para uma única transformação linear',
      'Para reduzir o número de parâmetros',
      'Para garantir que gradientes não explodam'
    ],
    ans: 1,
    exp: 'Uma composição de transformações lineares W₂(W₁x) é igual a Wx — uma única matriz. Sem ativações não-lineares, empilhar camadas não adiciona poder expressivo. ReLU, tanh e sigmoid quebram esse colapso.'
  },
  {
    q: 'Qual vantagem o ReLU tem sobre o Sigmoid em redes profundas?',
    opts: [
      'ReLU é sempre menor que 1, prevenindo explosão de gradientes',
      'ReLU não sofre com gradientes desvanecentes para entradas positivas (derivada = 1)',
      'ReLU produz saídas probabilísticas entre 0 e 1',
      'ReLU é diferenciável em todos os pontos'
    ],
    ans: 1,
    exp: 'Para x > 0, ReLU(x) = x e ReLU\'(x) = 1 — o gradiente não diminui. Sigmoid\'(x) ≤ 0,25, causando gradientes desvanecentes em redes profundas. ReLU também é computacionalmente mais eficiente.'
  },
  {
    q: 'O que o gradiente ∂L/∂w representa no treinamento?',
    opts: [
      'O valor da perda no ponto atual',
      'A direção e magnitude para mudar w de forma a AUMENTAR a perda',
      'A taxa de aprendizado ótima para convergência',
      'A acurácia do modelo no conjunto de validação'
    ],
    ans: 1,
    exp: 'O gradiente aponta na direção de maior aumento da perda. Para minimizar a perda, atualizamos os pesos na direção OPOSTA: w ← w - η·∂L/∂w. Essa é a essência do gradiente descendente.'
  }
]);
</script>
