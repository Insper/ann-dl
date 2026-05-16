<div id="quiz-gen-models"></div>
<script>
buildQuiz('gen-models', 'Modelos Generativos', [
  {
    q: 'O que um modelo generativo aprende, diferentemente de um modelo discriminativo?',
    opts: [
      'Apenas como classificar amostras em categorias',
      'A distribuição conjunta p(x, y) ou marginal p(x), permitindo a geração de novas amostras',
      'Apenas a fronteira de decisão entre classes sem modelar os dados',
      'Apenas regressão entre variáveis contínuas'
    ],
    ans: 1,
    exp: 'Modelos discriminativos aprendem p(y|x) — a probabilidade da classe dada a entrada. Modelos generativos aprendem p(x) ou p(x,y) — a própria distribuição dos dados, permitindo amostrar novas instâncias plausíveis.'
  },
  {
    q: 'O que é o espaço latente?',
    opts: [
      'O espaço de hiperparâmetros do modelo',
      'Uma representação contínua comprimida onde pontos próximos correspondem a dados similares',
      'O conjunto de vocabulário de tokens de um LLM',
      'A camada de saída do modelo que produz probabilidades'
    ],
    ans: 1,
    exp: 'O espaço latente é de dimensionalidade reduzida onde cada ponto z representa estrutura dos dados. Em VAEs, z ~ N(0,I) e o decoder G(z) gera dados realistas. Interpolar entre dois pontos latentes produz transições suaves.'
  },
  {
    q: 'Quais são as três principais famílias de modelos generativos profundos?',
    opts: [
      'CNN, RNN, Transformer',
      'GANs (adversariais), VAEs (variacionais), Modelos de Difusão',
      'Aprendizado Supervisionado, Não Supervisionado, por Reforço',
      'Autoencoder, Autorregressivo, Baseado em Atenção'
    ],
    ans: 1,
    exp: 'GANs: geração via jogo adversarial G vs D. VAEs: geração via espaço latente Gaussiano e ELBO. Difusão: geração via reversão de ruído. Cada família tem trade-offs em qualidade, diversidade e estabilidade de treinamento.'
  },
  {
    q: 'O que é a "maldição da dimensionalidade" em modelos generativos?',
    opts: [
      'A impossibilidade de treinar modelos com mais de 1 bilhão de parâmetros',
      'Em espaços de alta dimensionalidade, os dados ficam esparsos e distâncias Euclidianas perdem significado',
      'O fato de que mais dimensões inevitavelmente causam overfitting',
      'A limitação das GPUs no processamento de tensores de alta dimensionalidade'
    ],
    ans: 1,
    exp: 'Em R^d com d grande, o volume cresce exponencialmente e os dados ficam extremamente esparsos. Isso torna a estimação de densidade muito difícil. É por isso que modelos como VAE e difusão trabalham em espaços latentes comprimidos.'
  },
  {
    q: 'Por que a geração autossupervisionada permite treinamento com dados não rotulados?',
    opts: [
      'Porque modelos generativos não precisam de exemplos de entrada',
      'Porque os próprios dados são o alvo — o modelo aprende a reconstruir ou prever partes dos dados sem rótulos externos',
      'Porque rótulos são gerados automaticamente por um LLM',
      'Porque o treinamento usa apenas aprendizado por reforço sem rótulos'
    ],
    ans: 1,
    exp: 'No GPT (predição do próximo token) e MAE (autoencoding mascarado), os alvos são extraídos dos próprios dados. Não é necessária anotação manual — qualquer texto ou imagem é um exemplo de treinamento. Isso permite treinamento em escala da internet.'
  }
]);
</script>
