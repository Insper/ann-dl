<div id="quiz-metrics-gen"></div>
<script>
buildQuiz('metrics-gen', 'Métricas de Modelos Generativos', [
  {
    q: 'O que o FID (Fréchet Inception Distance) mede?',
    opts: [
      'Perda de reconstrução pixel a pixel entre imagens reais e geradas',
      'A distância entre distribuições de features de imagens reais e geradas no espaço de ativação do InceptionV3',
      'A acurácia de um classificador que distingue imagens reais de geradas',
      'O número médio de artefatos visuais por imagem gerada'
    ],
    ans: 1,
    exp: 'FID compara médias e covariâncias das features do InceptionV3 de imagens reais vs. geradas usando a distância de Fréchet. FID menor = distribuições mais similares = melhor qualidade. É a métrica padrão para GANs e modelos de difusão.'
  },
  {
    q: 'O que é colapso de modo (mode collapse) em GANs?',
    opts: [
      'Quando o discriminador para de aprender',
      'Quando o gerador aprende a produzir apenas algumas amostras plausíveis, ignorando a diversidade da distribuição real',
      'Quando a perda do gerador colapsa para zero prematuramente',
      'Quando a GAN fica instável e as imagens ficam extremamente ruidosas'
    ],
    ans: 1,
    exp: 'Mode collapse: o gerador encontra saídas que enganam o discriminador e as repete, perdendo diversidade. FID e Inception Score capturam esse modo de falha pois as amostras geradas cobrem menos modos.'
  },
  {
    q: 'O Inception Score (IS) mede:',
    opts: [
      'Apenas qualidade visual, sem considerar diversidade',
      'Tanto qualidade (imagens claramente classificáveis) QUANTO diversidade (distribuição uniforme entre classes)',
      'Apenas a diversidade do conjunto gerado',
      'A velocidade de inferência do gerador'
    ],
    ans: 1,
    exp: 'IS = exp(E[KL(p(y|x) || p(y))]). p(y|x) deve ser concentrada (imagem clara) e p(y) deve ser uniforme (diversidade). IS alto significa imagens nítidas E variadas. No entanto, não compara com dados reais — FID é mais confiável.'
  },
  {
    q: 'O que o CLIP Score é usado para avaliar?',
    opts: [
      'Qualidade perceptual de imagens geradas independente do texto',
      'Alinhamento semântico entre a imagem gerada e o prompt de texto',
      'Velocidade de geração de imagens em quadros por segundo',
      'Fidelidade de reconstrução do autoencoder'
    ],
    ans: 1,
    exp: 'CLIP Score mede a similaridade cosseno entre o embedding da imagem gerada e o embedding do prompt no espaço compartilhado do CLIP. Alto CLIP Score indica que a imagem corresponde semanticamente ao prompt — crucial para avaliação texto-para-imagem.'
  },
  {
    q: 'Por que métricas automáticas de qualidade de texto (BLEU, ROUGE) são insuficientes para avaliar LLMs modernos?',
    opts: [
      'Porque são muito lentas para calcular',
      'Porque comparam n-gramas superficialmente e não capturam semântica, criatividade ou factualidade',
      'Porque só funcionam em inglês',
      'Porque requerem GPU para cálculo'
    ],
    ans: 1,
    exp: 'BLEU/ROUGE medem sobreposição de palavras com referências fixas. Para texto aberto (chatbots, raciocínio, código), há múltiplas respostas corretas e a qualidade semântica não é capturada por n-gramas superficiais. Daí o surgimento da avaliação LLM-as-Judge.'
  }
]);
</script>
