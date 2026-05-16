<div id="quiz-regularization"></div>
<script>
buildQuiz('regularization', 'Regularização', [
  {
    q: 'O que o Dropout faz durante o treinamento?',
    opts: [
      'Reduz progressivamente a taxa de aprendizado',
      'Zera aleatoriamente ativações de neurônios com probabilidade p, forçando representações redundantes',
      'Adiciona uma penalidade L2 aos pesos',
      'Para o treinamento quando a perda de validação para de melhorar'
    ],
    ans: 1,
    exp: 'Dropout (Srivastava et al., 2014) desativa neurônios aleatoriamente em cada passagem direta. Isso previne co-adaptação entre neurônios e age como um ensemble de sub-redes. Na inferência, todos os neurônios estão ativos e os pesos são escalados por (1-p).'
  },
  {
    q: 'O que é o tradeoff viés-variância?',
    opts: [
      'O tradeoff entre usar viés ou pesos em redes neurais',
      'Modelos complexos têm baixo viés mas alta variância (overfitting); modelos simples têm alto viés mas baixa variância (underfitting)',
      'O equilíbrio entre taxa de aprendizado e tamanho do batch',
      'A relação entre acurácia de treinamento e velocidade de inferência'
    ],
    ans: 1,
    exp: 'Viés = erro sistemático (premissas erradas). Variância = sensibilidade a flutuações nos dados de treinamento. O objetivo é minimizar ambos encontrando a complexidade de modelo ótima. Regularização, mais dados ou modelos mais simples reduzem a variância.'
  },
  {
    q: 'O que a regularização L2 (Weight Decay) faz?',
    opts: [
      'Zera pesos pequenos, produzindo esparsidade',
      'Adiciona uma penalidade proporcional ao quadrado dos pesos à função de perda, encolhendo-os em direção a zero',
      'Normaliza ativações em cada camada',
      'Recorta o gradiente a um valor máximo durante o treinamento'
    ],
    ans: 1,
    exp: 'L2 adiciona λ||w||² à perda, fazendo o gradiente incluir -λw. Isso "encolhe" todos os pesos em direção a zero, mas raramente para exatamente zero. Equivalente a uma prior Gaussiana sobre os pesos.'
  },
  {
    q: 'O que é Early Stopping?',
    opts: [
      'Parar o treinamento após um número fixo de épocas independentemente do desempenho',
      'Monitorar a perda de validação e parar quando ela começa a aumentar, antes do overfitting',
      'Reduzir a taxa de aprendizado após cada época',
      'Remover camadas profundas se o modelo não convergir'
    ],
    ans: 1,
    exp: 'Early stopping monitora a métrica de validação e salva o melhor checkpoint. Quando a validação piora por N épocas consecutivas (paciência), o treinamento para e o melhor modelo é restaurado.'
  },
  {
    q: 'O que é Batch Normalization e qual é seu principal benefício?',
    opts: [
      'Normaliza dados de entrada para [0,1] antes do treinamento',
      'Normaliza as ativações de cada mini-batch, acelerando o treinamento e reduzindo sensibilidade à taxa de aprendizado',
      'Embaralha mini-batches para reduzir correlação',
      'Normaliza gradientes para evitar explosão'
    ],
    ans: 1,
    exp: 'Batch Norm (Ioffe & Szegedy, 2015) normaliza as ativações de cada camada para média 0, variância 1 por mini-batch, seguido de escala γ e deslocamento β aprendíveis. Permite taxas de aprendizado maiores e mitiga gradientes desvanecentes.'
  }
]);
</script>
