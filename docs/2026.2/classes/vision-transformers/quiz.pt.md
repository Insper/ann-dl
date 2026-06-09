<div id="quiz-vit"></div>
<script>
buildQuiz('vit', 'Vision Transformers', [
  {
    q: 'Como um Vision Transformer transforma uma imagem em uma sequência que o encoder consegue ler?',
    opts: [
      'Aplica várias camadas convolucionais e depois achata o mapa de features',
      'Divide a imagem em patches de tamanho fixo e projeta linearmente cada patch em um token',
      'Alimenta os pixels brutos um a um como tokens individuais',
      'Calcula um histograma de intensidades de pixel por canal'
    ],
    ans: 1,
    exp: 'O ViT corta a imagem em N patches P×P não-sobrepostos, achata cada um e o projeta com uma única camada linear compartilhada (equivalente a uma Conv2d com kernel=stride=P) → uma sequência de N embeddings de patch.'
  },
  {
    q: 'Por que os positional embeddings são necessários em um ViT?',
    opts: [
      'Para reduzir o número de patches e economizar memória',
      'Porque a self-attention é invariante a permutação, então sem eles o modelo não sabe de onde veio cada patch',
      'Para normalizar os embeddings de patch antes da atenção',
      'Para converter a imagem em tons de cinza'
    ],
    ans: 1,
    exp: 'A self-attention pura trata os tokens como um conjunto sem ordem. Positional embeddings aprendíveis são somados a cada token de patch para que o modelo recupere a localização espacial.'
  },
  {
    q: 'Qual é o papel do token [CLS] em um classificador ViT?',
    opts: [
      'Armazena o positional encoding da imagem inteira',
      'Sua representação final é passada para a cabeça MLP para produzir as probabilidades de classe',
      'Marca o fim da sequência de patches',
      'É o token do schedule de learning rate'
    ],
    ans: 1,
    exp: 'Um token [CLS] aprendível é adicionado ao início da sequência; após o encoder, apenas seu estado final é passado pela cabeça MLP para classificação.'
  },
  {
    q: 'Comparado a uma CNN, qual é a principal desvantagem prática do ViT?',
    opts: [
      'Não consegue processar imagens coloridas',
      'Tem viés indutivo fraco, então precisa de pré-treinamento em larga escala para igualar ou superar CNNs',
      'Só funciona em imagens quadradas menores que 32×32',
      'Não pode ser ajustado (finetuned) em tarefas posteriores'
    ],
    ans: 1,
    exp: 'Sem os priors de localidade/equivariância à translação, o ViT fica abaixo das CNNs em datasets pequenos (ex: só ImageNet-1k), mas as supera quando pré-treinado em datasets enormes (ImageNet-21k, JFT-300M).'
  },
  {
    q: 'Qual afirmação sobre o encoder do ViT está correta?',
    opts: [
      'É uma arquitetura totalmente nova, sem relação com o Transformer de texto',
      'É essencialmente o mesmo encoder Transformer (blocos de Multi-Head Self-Attention + FFN) usado para texto',
      'Substitui a atenção por convoluções dentro de cada bloco',
      'Usa uma máscara causal como o GPT'
    ],
    ans: 1,
    exp: 'Após o patch embedding, o ViT reutiliza o encoder Transformer padrão — blocos empilhados de MHSA + FFN com conexões residuais e LayerNorm (pre-norm, GELU). A novidade está apenas em como os tokens de entrada são construídos.'
  }
]);
</script>
