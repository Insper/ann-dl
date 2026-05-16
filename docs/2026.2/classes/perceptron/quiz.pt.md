<div id="quiz-perceptron"></div>
<script>
buildQuiz('perceptron', 'Perceptron', [
  {
    q: 'Qual é a regra de atualização de pesos no Perceptron de Rosenblatt?',
    opts: [
      'w = w - η∇L (gradiente descendente)',
      'w = w + η·y·x (atualização apenas em amostras mal classificadas)',
      'w = w × η (multiplicar pela taxa de aprendizado)',
      'w = w / ||w|| (normalizar os pesos)'
    ],
    ans: 1,
    exp: 'A regra do Perceptron atualiza pesos apenas em erros: w ← w + η·y·x, onde y é o rótulo verdadeiro (±1) e x a entrada. Nenhuma atualização ocorre para amostras corretamente classificadas.'
  },
  {
    q: 'Por que um único Perceptron não consegue resolver o problema XOR?',
    opts: [
      'Porque o Perceptron não pode ter um termo de bias',
      'Porque XOR não é linearmente separável — nenhum único hiperplano separa as classes',
      'Porque o Perceptron usa ativação sigmoid em vez de função degrau',
      'Porque XOR requer gradiente descendente, não a regra do Perceptron'
    ],
    ans: 1,
    exp: 'O Perceptron aprende uma fronteira de decisão linear (hiperplano). XOR produz 1 para {(0,1), (1,0)} e 0 para {(0,0), (1,1)} — essas classes não podem ser separadas por nenhuma linha reta.'
  },
  {
    q: 'O Teorema de Convergência do Perceptron garante que:',
    opts: [
      'O Perceptron sempre converge independentemente dos dados',
      'Se os dados são linearmente separáveis, o Perceptron converge em um número finito de atualizações',
      'O Perceptron converge mais rápido com uma taxa de aprendizado maior',
      'O Perceptron converge apenas com dados normalizados'
    ],
    ans: 1,
    exp: 'Rosenblatt provou que se os dados são linearmente separáveis, o algoritmo do Perceptron encontra uma solução em um número finito de passos. Se os dados não são separáveis, o algoritmo oscila indefinidamente.'
  },
  {
    q: 'Qual função de ativação o Perceptron original usa?',
    opts: ['Sigmoid (logística)', 'ReLU', 'Função degrau de Heaviside', 'Tanh'],
    ans: 2,
    exp: 'O Perceptron usa a função degrau: saída 1 se Σwᵢxᵢ + b ≥ 0, caso contrário 0 (ou -1). Isso produz decisões binárias e não é diferenciável — daí a regra do Perceptron em vez do gradiente descendente padrão.'
  },
  {
    q: 'O que o termo de bias (b) no Perceptron representa?',
    opts: [
      'A taxa de aprendizado do modelo',
      'Um deslocamento que permite que a fronteira de decisão não passe pela origem',
      'O número de épocas de treinamento',
      'A norma do vetor de pesos'
    ],
    ans: 1,
    exp: 'O bias desloca a fronteira de decisão (hiperplano) para que não precise passar pela origem (0,...,0). Sem o bias, o modelo só pode aprender fronteiras de decisão forçadas a passar pela origem.'
  }
]);
</script>
