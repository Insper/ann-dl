<div id="quiz-clip"></div>
<script>
buildQuiz('clip', 'CLIP — Contrastive Language-Image Pretraining', [
  {
    q: 'Qual é o objetivo de treinamento do CLIP?',
    opts: [
      'Gerar imagens a partir de descrições de texto',
      'Maximizar a similaridade cosseno de pares imagem-texto correspondentes e minimizá-la para pares não correspondentes (aprendizado contrastivo)',
      'Classificar imagens nas 1000 categorias do ImageNet',
      'Segmentar objetos usando descrições de texto como prompts'
    ],
    ans: 1,
    exp: 'CLIP treina em 400M pares (imagem, texto) da internet. Para um batch de N pares, maximiza a diagonal (pares corretos) e minimiza as N²-N entradas fora da diagonal na matriz de similaridade, alinhando ambos os espaços de embedding.'
  },
  {
    q: 'O que é classificação zero-shot com CLIP?',
    opts: [
      'Classificar imagens com zero exemplos de treinamento usando apenas descrições textuais das classes',
      'Usar CLIP para classificar texto sem fine-tuning',
      'Treinar um classificador com 0 exemplos positivos por classe',
      'Usar CLIP sem o encoder de texto'
    ],
    ans: 0,
    exp: 'Para classificar "gato" vs "cachorro" com CLIP: codifique a imagem e compare com embeddings de "uma foto de um gato" e "uma foto de um cachorro". A maior similaridade cosseno define a classe — sem treinamento de classificador específico para a tarefa.'
  },
  {
    q: 'Quais são os dois encoders no CLIP?',
    opts: [
      'Encoder de imagem (CNN/ViT) e encoder de texto (Transformer), ambos mapeando para o mesmo espaço de embedding',
      'Encoder de imagem e decoder de texto, para geração de legendas',
      'Um encoder compartilhado para imagem e texto',
      'Encoder de patches e encoder de tokens para alinhamento local'
    ],
    ans: 0,
    exp: 'Encoder de Imagem: ViT ou ResNet modificado (ex: ViT-L/14). Encoder de Texto: Transformer estilo GPT. Ambos projetam para um espaço d-dimensional compartilhado (ex: 512 ou 768) onde a similaridade cosseno é calculada.'
  },
  {
    q: 'Qual é uma limitação conhecida do CLIP para classificação de grão fino?',
    opts: [
      'CLIP não funciona com imagens coloridas',
      'CLIP tem dificuldades com subcategorias de grão fino (modelos de carros, espécies de flores) e requer engenharia cuidadosa de prompts',
      'CLIP só funciona em inglês',
      'CLIP não pode ser usado sem uma GPU'
    ],
    ans: 1,
    exp: 'CLIP generaliza bem para categorias comuns (animais, objetos) mas tem desempenho próximo ao aleatório para distinções de grão fino (ex: variantes de modelos de carros, tipos de aeronaves). Também é sensível ao texto do prompt: "uma foto de um gato" vs "gato" pode diferir em 10+ pontos percentuais.'
  },
  {
    q: 'Como o CLIP é usado no Stable Diffusion?',
    opts: [
      'Como gerador de imagens usando apenas o encoder de texto',
      'O encoder de texto do CLIP converte o prompt em embeddings que condicionam o processo de difusão via cross-attention na U-Net/DiT',
      'CLIP é usado como discriminador adversarial para avaliar a qualidade das imagens geradas',
      'CLIP comprime imagens para o espaço latente do VAE'
    ],
    ans: 1,
    exp: 'No SD: texto → encoder de texto CLIP → embeddings de condicionamento → U-Net os recebe via cross-attention em cada bloco de resolução. Os embeddings guiam quais features são desruidadas. FLUX usa T5-XXL + CLIP para condicionamento mais rico.'
  }
]);
</script>
