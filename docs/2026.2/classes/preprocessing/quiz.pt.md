<div id="quiz-preprocessing"></div>
<script>
buildQuiz('preprocessing', 'Pré-processamento', [
  {
    q: 'Qual é a diferença entre normalização Min-Max e padronização Z-score?',
    opts: [
      'Normalização mapeia para [0,1]; padronização transforma para média 0 e desvio padrão 1',
      'São equivalentes e produzem o mesmo resultado',
      'Normalização é para dados contínuos; padronização para categóricos',
      'Padronização mapeia para [0,1]; normalização para média 0'
    ],
    ans: 0,
    exp: 'Min-Max: x\' = (x-min)/(max-min) → [0,1]. Z-score: x\' = (x-μ)/σ → média 0, std 1. Use padronização quando os dados têm outliers; normalização quando os limites são conhecidos e significativos.'
  },
  {
    q: 'Para que serve a Análise de Componentes Principais (PCA)?',
    opts: [
      'Classificar amostras em grupos',
      'Reduzir dimensionalidade preservando a máxima variância dos dados',
      'Aumentar artificialmente o número de amostras de treinamento',
      'Normalizar dados para [0,1]'
    ],
    ans: 1,
    exp: 'PCA encontra os eixos de maior variância (componentes principais) e projeta os dados nesse espaço reduzido. Útil para visualização, compressão e remoção de features correlacionadas.'
  },
  {
    q: 'O que é One-Hot Encoding?',
    opts: [
      'Uma técnica de regularização para redes neurais',
      'Converter variáveis categóricas em vetores binários com um único bit ativo por categoria',
      'Normalizar dados numéricos para [0,1]',
      'Uma estratégia para lidar com valores faltantes'
    ],
    ans: 1,
    exp: 'One-Hot transforma "cor: [vermelho, azul, verde]" em três colunas binárias: vermelho=[1,0,0], azul=[0,1,0], verde=[0,0,1]. Evita que o modelo assuma relacionamentos ordinais entre categorias.'
  },
  {
    q: 'Qual estratégia é mais adequada para preencher valores faltantes (NaN) em uma feature numérica?',
    opts: [
      'Sempre deletar linhas com NaN',
      'Substituir pela mediana ou média da coluna (imputação)',
      'Sempre substituir por zero',
      'Valores faltantes não requerem tratamento'
    ],
    ans: 1,
    exp: 'Imputação pela mediana é robusta a outliers. A média funciona para distribuições simétricas. Deletar linhas perde informação e pode introduzir viés se os dados não são Missing At Random (MAR).'
  },
  {
    q: 'Por que um scaler deve ser ajustado APENAS no conjunto de treinamento?',
    opts: [
      'Para economizar tempo de processamento',
      'Para evitar vazamento de dados — o scaler não deve ver estatísticas do conjunto de teste',
      'O scaler deve ser calculado em todos os dados para maior precisão',
      'Porque o conjunto de teste não precisa de normalização'
    ],
    ans: 1,
    exp: 'Calcular min/max ou μ/σ usando o conjunto de teste é vazamento de dados. O pré-processador deve ser ajustado apenas nos dados de treinamento e depois aplicado (transform) em treino e teste separadamente.'
  }
]);
</script>
