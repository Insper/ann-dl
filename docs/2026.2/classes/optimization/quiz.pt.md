<div id="quiz-optimization"></div>
<script>
buildQuiz('optimization', 'Otimização', [
  {
    q: 'O que a taxa de aprendizado η controla no gradiente descendente?',
    opts: [
      'O número de épocas de treinamento',
      'O tamanho do passo das atualizações de parâmetros na direção do gradiente',
      'A proporção de dados usados em cada mini-batch',
      'A taxa de dropout aplicada durante o treinamento'
    ],
    ans: 1,
    exp: 'θ ← θ - η·∇L. Uma η muito grande causa oscilações ou divergência; muito pequena causa convergência lenta. Agendamento de taxa de aprendizado (warm-up, decaimento cosseno) ajuda a encontrar o equilíbrio correto.'
  },
  {
    q: 'Qual é a principal vantagem do Mini-batch SGD sobre o Batch GD completo?',
    opts: [
      'Mini-batch sempre converge para um mínimo melhor',
      'Atualizações mais frequentes com estimativas ruidosas de gradiente que podem escapar de mínimos locais',
      'Mini-batch não requer carregar dados na memória',
      'Mini-batch tem gradientes mais precisos que o Batch GD'
    ],
    ans: 1,
    exp: 'Mini-batch calcula gradientes em subconjuntos (32–512 amostras), fornecendo atualizações frequentes com ruído controlado. O ruído pode ajudar a escapar de mínimos locais ruins e é eficiente em hardware paralelo (GPU).'
  },
  {
    q: 'O que o Momentum faz no gradiente descendente?',
    opts: [
      'Ajusta automaticamente a taxa de aprendizado por parâmetro',
      'Acumula uma média móvel exponencial dos gradientes passados, suavizando a trajetória de otimização',
      'Normaliza gradientes para norma unitária',
      'Adiciona ruído ao gradiente para exploração'
    ],
    ans: 1,
    exp: 'Momentum mantém uma "velocidade" v ← βv + (1-β)∇L e atualiza θ ← θ - η·v. Suaviza oscilações e acelera convergência em direções consistentes — como uma bola rolando morro abaixo.'
  },
  {
    q: 'O otimizador Adam combina quais dois conceitos?',
    opts: [
      'Batch GD e SGD',
      'Momentum de primeira ordem e estimativa de momento de segunda ordem (taxas de aprendizado adaptativas por parâmetro)',
      'Regularização L1 e L2',
      'Dropout e batch normalization'
    ],
    ans: 1,
    exp: 'Adam mantém m_t (momentum de gradiente) e v_t (média móvel do gradiente ao quadrado — como RMSProp). Divide a atualização por sqrt(v_t), adaptando a taxa de aprendizado por parâmetro. Padrão para treinamento de LLMs e modelos de visão.'
  },
  {
    q: 'Qual é a diferença entre Adam e AdamW?',
    opts: [
      'AdamW usa uma taxa de aprendizado maior por padrão',
      'AdamW aplica weight decay diretamente nos pesos, não via gradiente — tornando-o matematicamente correto',
      'AdamW não usa momentum de segunda ordem',
      'AdamW é mais lento mas mais preciso que Adam'
    ],
    ans: 1,
    exp: 'No Adam original, o weight decay é adicionado ao gradiente antes da escala adaptativa, distorcendo o decaimento. AdamW (Loshchilov & Hutter, 2019) aplica decaimento diretamente em θ, separado da atualização adaptativa. Padrão para treinamento de Transformers.'
  }
]);
</script>
