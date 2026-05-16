<div id="quiz-ar-gen"></div>
<script>
buildQuiz('ar-gen', 'Geração Autorregressiva de Imagens', [
  {
    q: 'O que o VQ-GAN (Vector Quantized GAN) faz?',
    opts: [
      'Gera imagens de alta resolução de forma adversarial sem etapa de quantização',
      'Aprende a comprimir imagens em tokens discretos via um codebook de K vetores, permitindo que imagens sejam tratadas como sequências de índices',
      'Aplica quantização de pesos para reduzir o tamanho do modelo',
      'Usa quantização de gradiente para acelerar o treinamento do GAN'
    ],
    ans: 1,
    exp: 'VQ-GAN: Encoder → mapa de features → quantização (cada feature mapeada para o vetor mais próximo do codebook) → índice inteiro. Decoder reconstrói a partir dos vetores quantizados. Uma imagem 256×256 → 16×16 = 256 tokens inteiros.'
  },
  {
    q: 'Como funciona a geração autorregressiva de imagens após o VQ-GAN?',
    opts: [
      'Todos os tokens são gerados simultaneamente em paralelo',
      'Um Transformer gera índices do codebook um a um, da mesma forma que o GPT gera texto',
      'O discriminador do VQ-GAN gera tokens em ordem decrescente de importância',
      'Um algoritmo de busca em feixe seleciona a sequência de tokens mais provável'
    ],
    ans: 1,
    exp: 'Com um codebook treinado, sequências de tokens representam imagens. Um Transformer estilo GPT aprende p(t_i | t_1,...,t_{i-1}) — o próximo token dado os anteriores. Geração: amostrar token por token e decodificar com o decoder do VQ-GAN.'
  },
  {
    q: 'Como o MaskGIT acelera a geração em comparação com o autorregressivo puro?',
    opts: [
      'Usando um modelo menor em paralelo',
      'Gerando todos os tokens mascarados em paralelo de forma iterativa, revelando os mais confiantes a cada passo',
      'Pulando tokens menos importantes com base na atenção',
      'Usando um codebook menor com menos tokens por imagem'
    ],
    ans: 1,
    exp: 'AR puro: N passos para N tokens. MaskGIT: começa com todos os tokens mascarados, prediz TODOS simultaneamente, revela os top-k mais confiantes, repete ~8-12 vezes. N=1024 tokens em 8-12 passes vs 1024 passes no AR puro.'
  },
  {
    q: 'O que caracteriza modelos "any-to-any" como Gemini e Chameleon?',
    opts: [
      'Podem gerar qualquer formato de arquivo a partir de qualquer entrada de texto',
      'Processam texto e imagem como uma única sequência de tokens misturados pelo mesmo Transformer, sem arquiteturas específicas por modalidade',
      'Usam múltiplos modelos especializados e combinam saídas por votação',
      'Rodam em qualquer hardware sem aceleração especial'
    ],
    ans: 1,
    exp: 'Any-to-any: tokens de texto e imagem coexistem na mesma sequência e passam pelo mesmo Transformer. [texto][img_tok][texto][img_tok] — o modelo aprende a atender a qualquer padrão. Gemini 2.0 Flash pode intercalar texto e imagens gerados nativamente.'
  },
  {
    q: 'Qual é a principal desvantagem da geração autorregressiva pura em comparação com a difusão para imagens?',
    opts: [
      'Qualidade visual inferior às imagens de difusão em todos os casos',
      'Geração lenta: N tokens = N forward passes sequenciais, enquanto a difusão atualiza todos os pixels em paralelo por passo',
      'Não suporta condicionamento por texto',
      'Requer treinamento de um discriminador adversarial separado'
    ],
    ans: 1,
    exp: 'AR puro: para 16×16 = 256 tokens, requer 256 passes. Difusão: a cada passo ODE, todos os pixels/latentes são atualizados em paralelo (um forward pass de U-Net/DiT). Para N grande, AR é muito mais lento — daí o MaskGIT e abordagens híbridas.'
  }
]);
</script>
