<div id="quiz-flow-matching"></div>
<script>
buildQuiz('flow-matching', 'Flow Matching', [
  {
    q: 'O que o Flow Matching treina o modelo para prever?',
    opts: [
      'O ruído ε adicionado a cada passo de difusão',
      'O campo de velocidade v_θ(x_t, t) que move partículas de p_0 (ruído) para p_1 (dados)',
      'Probabilidades de tokens em modelos de linguagem',
      'A distribuição de latentes z no espaço VAE'
    ],
    ans: 1,
    exp: 'dx/dt = v_θ(x, t). FM treina o campo de velocidade que define a trajetória de transformação. Com o caminho OT (Transporte Ótimo), a velocidade alvo é simplesmente x_1 - x_0 — constante ao longo da trajetória reta.'
  },
  {
    q: 'Qual é o caminho condicional mais simples no Flow Matching (CFM)?',
    opts: [
      'Uma trajetória browniana estocástica como no DDPM',
      'Interpolação linear (linha reta): x_t = (1-t)x_0 + t·x_1',
      'Uma curva de Bezier entre x_0 e x_1',
      'Um arco de grande círculo no espaço hiperesférico'
    ],
    ans: 1,
    exp: 'O caminho OT no CFM é uma linha reta entre ruído (x_0~N(0,I)) e dados reais (x_1). O vetor de velocidade alvo é constante: u_t(x|x_0,x_1) = x_1 - x_0. Muito mais simples que o cronograma de ruído β_t do DDPM.'
  },
  {
    q: 'Qual é a principal vantagem de inferência do Flow Matching sobre o DDPM?',
    opts: [
      'FM gera imagens de maior resolução',
      'FM tem trajetórias mais retas, precisando de 10-50 passos ODE vs 500-1000 para DDPM',
      'FM não requer GPU para inferência',
      'FM pode gerar em tempo real sem passos iterativos'
    ],
    ans: 1,
    exp: 'Trajetórias retas = menos curvatura = integradores de ODE precisam de menos passos para boa aproximação. DDPM empírico usa 50-1000 passos (DDIM). FM com integração Euler precisa de ~20-50 passos com qualidade comparável.'
  },
  {
    q: 'Qual modelo de geração de imagens de ponta usa Flow Matching com um Diffusion Transformer?',
    opts: ['Stable Diffusion 1.5', 'Midjourney v5', 'FLUX.1 (Black Forest Labs, 2024)', 'DALL-E 2'],
    ans: 2,
    exp: 'FLUX.1 (2024) combina: Flow Matching como objetivo de treinamento + MMDiT (Diffusion Transformer Multi-Modal) como arquitetura + espaço latente VAE. Estado da arte em geração open-source com 12B parâmetros.'
  },
  {
    q: 'Como é realizada a inferência (geração) no Flow Matching?',
    opts: [
      'Amostragem diretamente do modelo sem iteração',
      'Amostrar z~N(0,I) e integrar a ODE dx/dt = v_θ de t=0 a t=1',
      'Remoção de ruído reversa de t=T a t=0 como no DDPM',
      'Amostrar tokens autorregressivamente e decodificar'
    ],
    ans: 1,
    exp: 'FM vai de t=0 (ruído) a t=1 (dados). Cada passo Euler: x_{t+Δt} = x_t + Δt·v_θ(x_t, t). Direção oposta ao DDPM (que vai de T→0 no espaço de ruído). Solucionadores Heun ou RK4 precisam de menos passos.'
  }
]);
</script>
