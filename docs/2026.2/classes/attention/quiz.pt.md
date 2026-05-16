<div id="quiz-attention"></div>
<script>
buildQuiz('attention', 'Mecanismos de Atenção', [
  {
    q: 'No mecanismo de atenção, o que representam Query (Q), Key (K) e Value (V)?',
    opts: [
      'Q é a saída, K é o kernel convolucional, V é o vetor de bias',
      'Q é o que você está buscando, K é o índice de cada item, V é o conteúdo retornado na correspondência',
      'Q, K e V são três cópias idênticas da entrada sem transformação',
      'Q é para geração, K é para classificação, V é para regressão'
    ],
    ans: 1,
    exp: 'Analogia de banco de dados: Query é a busca, Keys são os índices dos itens, Values são o conteúdo. A atenção calcula a similaridade Q·Kᵀ, normaliza com softmax e usa os pesos para combinar os Values.'
  },
  {
    q: 'Por que o fator de escala √d_k é necessário na atenção?',
    opts: [
      'Para garantir que os pesos de atenção somem 1 após o softmax',
      'Para evitar que os produtos Q·Kᵀ se tornem grandes em altas dimensões, causando gradientes softmax quase nulos',
      'Para acelerar o cálculo de matrizes em GPUs',
      'Para normalizar os embeddings posicionais'
    ],
    ans: 1,
    exp: 'Para vetores de dimensão d_k com componentes ~N(0,1), Q·Kᵀ tem variância d_k. Dividir por √d_k estabiliza a variância para 1, evitando saturação do softmax em regiões de gradiente nulo.'
  },
  {
    q: 'O que é Atenção Multi-Cabeça (Multi-Head Attention)?',
    opts: [
      'Atenção aplicada em múltiplas camadas sequencialmente',
      'Múltiplas cabeças de atenção paralelas com projeções independentes, cada uma capturando diferentes tipos de relacionamento',
      'Atenção com múltiplos tokens de consulta simultaneamente',
      'Um ensemble de modelos de atenção com votação'
    ],
    ans: 1,
    exp: 'Multi-Head usa h cabeças independentes com projeções distintas W_Q^i, W_K^i, W_V^i. Cada cabeça pode especializar-se em diferentes relacionamentos (sintáticos, semânticos, posicionais). As saídas são concatenadas e projetadas.'
  },
  {
    q: 'O que é Causal Self-Attention e onde é usada?',
    opts: [
      'Atenção em que cada token atende a todos os outros tokens bidirecionalmente',
      'Self-attention com máscara triangular que impede ver tokens futuros — usada em modelos autorregressivos como GPT',
      'Atenção em que apenas tokens adjacentes interagem',
      'Um tipo especial de cross-attention entre encoder e decoder'
    ],
    ans: 1,
    exp: 'A máscara causal define posições j > i como −∞ antes do softmax, garantindo que o token i veja apenas i e tokens anteriores. Essencial para geração autorregressiva: o modelo não pode "trapacear" vendo tokens futuros.'
  },
  {
    q: 'Qual é a complexidade computacional da atenção padrão em função do comprimento de sequência n?',
    opts: ['O(n)', 'O(n log n)', 'O(n²)', 'O(n³)'],
    ans: 2,
    exp: 'A matriz Q·Kᵀ tem dimensão n×n — cada um dos n tokens deve calcular atenção com todos os outros n tokens. Isso é O(n²) em tempo e memória, tornando a atenção padrão cara para sequências longas (ex: documentos, imagens HD).'
  }
]);
</script>
