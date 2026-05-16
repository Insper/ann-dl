<div id="quiz-cnn"></div>
<script>
buildQuiz('cnn', 'Redes Neurais Convolucionais', [
  {
    q: 'Uma camada convolucional com entrada 32×32, kernel 3×3, padding=1, stride=1 produz saída de qual tamanho?',
    opts: ['30×30', '32×32', '34×34', '16×16'],
    ans: 1,
    exp: 'H_out = (H_in + 2p - K) / s + 1 = (32 + 2 - 3)/1 + 1 = 32. Padding=1 preserva as dimensões espaciais com kernel 3×3 — conhecido como "same padding".'
  },
  {
    q: 'O que é compartilhamento de pesos (weight sharing) em CNNs?',
    opts: [
      'Compartilhar pesos entre diferentes redes na mesma tarefa',
      'O mesmo kernel é aplicado em todas as posições espaciais da entrada, reduzindo drasticamente o número de parâmetros',
      'Inicializar pesos de CNN com pesos de redes densas',
      'Usar dropout para que neurônios compartilhem representações'
    ],
    ans: 1,
    exp: 'Um filtro convolucional (ex: 3×3×3 = 27 parâmetros) desliza sobre a imagem inteira. Aplicado separadamente a cada posição (64×64 = 4096 regiões) seriam 4096×27 parâmetros. O compartilhamento de pesos é a chave da eficiência das CNNs.'
  },
  {
    q: 'Por que CNNs são mais eficientes que MLPs para imagens?',
    opts: [
      'CNNs usam funções de ativação mais rápidas',
      'CNNs exploram localidade espacial e compartilhamento de pesos — pixels próximos são correlacionados; o mesmo padrão pode ocorrer em qualquer lugar',
      'CNNs não requerem GPUs para treinamento',
      'CNNs não precisam de retropropagação'
    ],
    ans: 1,
    exp: 'Uma imagem 224×224×3 tem 150k entradas → um MLP com 1k neurônios = 150M parâmetros. Uma CNN com filtros 3×3 aprende detectores de features locais reutilizáveis em qualquer posição, com muito menos parâmetros.'
  },
  {
    q: 'O que são mapas de features em uma CNN?',
    opts: [
      'Um mapa de posições de pixels com alta intensidade',
      'As saídas de ativação de cada filtro convolucional, representando a resposta do filtro em cada posição espacial',
      'A matriz de pesos de uma camada convolucional',
      'O mapa de gradientes durante a retropropagação'
    ],
    ans: 1,
    exp: 'Cada filtro convolucional produz um mapa de features: uma grade 2D indicando onde na imagem um padrão particular (borda, textura, etc.) foi detectado. Com C_out filtros, obtemos C_out mapas de features por camada.'
  },
  {
    q: 'Qual arquitetura introduziu conexões residuais (skip) para permitir redes com 100+ camadas?',
    opts: ['AlexNet', 'VGGNet', 'ResNet (He et al., 2015)', 'Inception/GoogLeNet'],
    ans: 2,
    exp: 'ResNet (Deep Residual Learning, He et al., 2015) introduziu skip connections F(x)+x, permitindo treinamento de redes com 152+ camadas que superaram todos os benchmarks da época e inspirou praticamente todas as arquiteturas profundas subsequentes.'
  }
]);
</script>
