<div id="quiz-dit"></div>
<script>
buildQuiz('dit', 'Diffusion Transformers (DiT)', [
  {
    q: 'O que o DiT substitui em comparação ao Stable Diffusion clássico?',
    opts: [
      'Substitui o VAE por um encoder puramente convolucional',
      'Substitui a U-Net por blocos Transformer puros, mantendo difusão/FM no espaço latente',
      'Substitui o encoder de texto por redes recorrentes LSTM',
      'Substitui o processo de difusão por Flow Matching explícito'
    ],
    ans: 1,
    exp: 'DiT (Peebles & Xie, 2023) mantém o VAE, encoder de texto e processo de difusão. A mudança está na rede de denoising: de U-Net (conv + skip connections) para Transformer puro (atenção + FFN). Resultado: melhor escalabilidade com mais parâmetros.'
  },
  {
    q: 'O que é a operação "Patchify" no DiT?',
    opts: [
      'Dividir a imagem em patches antes de passar pelo VAE',
      'Dividir o latente em patches p×p e projetar cada um para um token de dimensão d_model',
      'Aplicar dropout em patches aleatórios do latente durante o treinamento',
      'Comprimir o latente por um fator de 2× antes do Transformer'
    ],
    ans: 1,
    exp: 'Patchify(latente H×W×C, patch p): divide em (H/p)×(W/p) patches → achatar → projeção linear → N tokens de dim d_model. Com p=2, o FLUX processa 4096 tokens para imagens de 1024px.'
  },
  {
    q: 'O que é o AdaLN (Adaptive Layer Normalization) no DiT?',
    opts: [
      'LayerNorm com parâmetros fixos aprendidos durante o treinamento',
      'LayerNorm cujos parâmetros γ e β são preditos dinamicamente pelo timestep e condicionamento, permitindo modulação por passo',
      'Normalização adaptativa ao tamanho do batch',
      'Batch Normalization adaptado para sequências de comprimento variável'
    ],
    ans: 1,
    exp: 'AdaLN(h, c) = γ(c) · norm(h) + β(c), onde c = MLP(emb(t) + emb(classe)). Diferente dos pesos fixos do LayerNorm, γ e β são gerados dinamicamente — o modelo "sabe" em qual timestep t está e ajusta a normalização de acordo.'
  },
  {
    q: 'O que distingue o MMDiT (FLUX/SD3) do DiT original?',
    opts: [
      'MMDiT usa convoluções enquanto DiT usa atenção',
      'MMDiT tem streams separados para tokens de imagem e texto que compartilham atenção bidirecional — texto vê imagem e imagem vê texto',
      'MMDiT é significativamente menor que DiT em número de parâmetros',
      'MMDiT substitui self-attention por cross-attention puro'
    ],
    ans: 1,
    exp: 'DiT injeta texto via cross-attention ou concatenação. MMDiT (SD3, FLUX) processa texto e imagem em streams paralelos com pesos separados, mas combina Q, K, V de ambos na mesma operação de atenção — bidirecional e condicionamento muito mais rico.'
  },
  {
    q: 'Por que o DiT escala melhor que a U-Net com mais parâmetros?',
    opts: [
      'Porque o DiT usa menos operações de ponto flutuante por parâmetro',
      'Porque a atenção global desde o primeiro bloco pode utilizar plenamente toda capacidade adicional, enquanto a U-Net tem gargalos hierárquicos',
      'Porque o DiT não requer batch normalization e é mais estável',
      'Porque o DiT processa menos tokens que a U-Net'
    ],
    ans: 1,
    exp: 'U-Net tem pooling/upsampling criando gargalos de informação. Skip connections ajudam mas há um limite hierárquico. DiT: atenção global completa (O(N²)) desde o bloco 1, sem hierarquia de resolução. Cada novo bloco/dimensão adiciona capacidade totalmente aproveitável.'
  }
]);
</script>
