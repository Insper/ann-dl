<div id="quiz-deep-learning"></div>
<script>
buildQuiz('deep-learning', 'Camadas de Deep Learning', [
  {
    q: 'O que uma camada Densa (Totalmente Conectada) faz?',
    opts: [
      'Conecta apenas neurônios adjacentes, preservando localidade espacial',
      'Conecta cada neurônio de entrada a cada neurônio de saída via matriz de pesos W e bias b',
      'Aplica convolução 2D sobre dados de imagem',
      'Normaliza ativações por batch durante o treinamento'
    ],
    ans: 1,
    exp: 'Dense(x) = ativação(Wx + b). Cada saída depende de TODAS as entradas. Eficaz para capturar relacionamentos globais mas ineficiente para dados com estrutura espacial (imagens), pois ignora localidade.'
  },
  {
    q: 'O que o Pooling (ex: Max Pooling) realiza?',
    opts: [
      'Aumenta a resolução espacial via interpolação',
      'Reduz dimensões espaciais selecionando o valor máximo por região, fornecendo invariância à translação',
      'Normaliza ativações por canal',
      'Aumenta o número de canais do mapa de features'
    ],
    ans: 1,
    exp: 'Max Pooling 2×2 com stride 2 reduz a resolução pela metade selecionando o valor máximo em cada bloco 2×2. Reduz parâmetros e memória, e fornece invariância a pequenas translações.'
  },
  {
    q: 'Qual é o propósito das skip connections (conexões residuais) nas ResNets?',
    opts: [
      'Acelerar a inferência pulando camadas desnecessárias',
      'Permitir que gradientes fluam diretamente pelas camadas, mitigando gradientes desvanecentes em redes muito profundas',
      'Reduzir o número de parâmetros do modelo',
      'Aplicar dropout seletivo em camadas intermediárias'
    ],
    ans: 1,
    exp: 'F(x) + x: a skip connection adiciona a entrada diretamente à saída do bloco. Os gradientes podem fluir pela adição sem passar pelas convoluções, resolvendo o problema do gradiente desvanecente e permitindo redes com 100+ camadas.'
  },
  {
    q: 'O que é um Embedding em deep learning?',
    opts: [
      'Uma técnica de compressão de imagem sem perdas',
      'Um mapeamento de entidades discretas (palavras, usuários, itens) para vetores contínuos densos de dimensão fixa',
      'A operação flatten que lineariza tensores 2D para 1D',
      'Um tipo de regularização que projeta gradientes'
    ],
    ans: 1,
    exp: 'Embeddings mapeiam índices discretos para vetores densos de dimensão d (ex: 256). Entidades similares são próximas no espaço de embedding. Essenciais em NLP (word2vec, BERT) e sistemas de recomendação.'
  },
  {
    q: 'O que a camada Flatten faz em uma CNN antes da camada densa final?',
    opts: [
      'Aplica normalização global dos mapas de features',
      'Converte o tensor multidimensional (ex: 7×7×512) em um vetor 1D para alimentar camadas densas',
      'Reduz o número de canais via pooling global',
      'Aplica dropout sobre os mapas de features'
    ],
    ans: 1,
    exp: 'Flatten(tensor shape [B, C, H, W]) → [B, C×H×W]. Necessário para conectar camadas convolucionais ao head de classificação denso. Alternativa: Global Average Pooling, que calcula a média por canal.'
  }
]);
</script>
