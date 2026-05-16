<div id="quiz-gan"></div>
<script>
buildQuiz('gan', 'Redes Adversariais Generativas (GAN)', [
  {
    q: 'Qual é o objetivo do Gerador em uma GAN?',
    opts: [
      'Distinguir entre imagens reais e falsas',
      'Produzir amostras tão realistas que o Discriminador não consiga distingui-las de dados reais',
      'Classificar as saídas do Discriminador em categorias',
      'Minimizar a divergência KL entre distribuições aprendida e prior'
    ],
    ans: 1,
    exp: 'Gerador G: max E[log D(G(z))]. Ele quer que o Discriminador classifique suas saídas como reais (D(G(z))→1). O jogo adversarial: G engana D, D aprende a detectar G, ambos melhoram iterativamente.'
  },
  {
    q: 'O que é colapso de modo (mode collapse) em GANs?',
    opts: [
      'Quando o Discriminador converge antes do Gerador',
      'Quando o Gerador produz apenas algumas saídas diversas, ignorando a variabilidade da distribuição real',
      'Quando a GAN diverge e as perdas explodem',
      'Quando o Gerador memoriza imagens de treinamento'
    ],
    ans: 1,
    exp: 'Mode collapse: G encontra um conjunto limitado de saídas que consistentemente enganam D e as repete. D não consegue aprender a distinção → equilíbrio ruim. Sintoma: todas as imagens geradas parecem similares. WGAN e outras variantes mitigam isso.'
  },
  {
    q: 'Como a Wasserstein GAN (WGAN) melhora o treinamento clássico de GANs?',
    opts: [
      'Usando um Discriminador mais profundo com mais camadas',
      'Substituindo a divergência JS pela distância de Wasserstein, fornecendo gradientes estáveis mesmo quando as distribuições não se sobrepõem',
      'Treinando Gerador e Discriminador com mais passos por época',
      'Adicionando regularização L2 ao Discriminador'
    ],
    ans: 1,
    exp: 'A divergência JS satura quando G e D são muito diferentes (gradiente ≈ 0). WGAN usa a distância de Earth Mover — fornece gradientes informativos mesmo quando as distribuições não se sobrepõem, estabilizando o treinamento.'
  },
  {
    q: 'O que é uma GAN Condicional (cGAN)?',
    opts: [
      'Uma GAN condicionada na distribuição de um Autoencoder',
      'Uma GAN onde Gerador e Discriminador recebem informação adicional (ex: rótulo de classe) para geração controlada',
      'Uma GAN com discriminação pareada (image-to-image)',
      'Uma GAN que usa atenção para condicionar a geração em features'
    ],
    ans: 1,
    exp: 'cGAN (Mirza & Osindero, 2014): G recebe (z, y) e gera uma imagem da classe y. D recebe (imagem, y) e classifica real/falso dentro da classe. Permite gerar dígitos MNIST específicos, rostos com atributos controlados, etc.'
  },
  {
    q: 'Na função de perda do Discriminador, o que ele maximiza?',
    opts: [
      'A probabilidade de imagens reais serem classificadas como falsas',
      'E[log D(x)] + E[log(1 - D(G(z)))] — classificar real como real E falso como falso',
      'Apenas a probabilidade de rejeitar imagens geradas',
      'A norma dos gradientes do Gerador'
    ],
    ans: 1,
    exp: 'L_D = -E[log D(x)] - E[log(1-D(G(z)))]. Maximizar isso significa D(x)→1 para imagens reais e D(G(z))→0 para geradas. Equivalente a minimizar o erro de classificação binária.'
  }
]);
</script>
