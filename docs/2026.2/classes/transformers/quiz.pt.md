<div id="quiz-transformers"></div>
<script>
buildQuiz('transformers', 'Transformers', [
  {
    q: 'O que substituiu as RNNs no Transformer, permitindo paralelização total?',
    opts: ['Convoluções dilatadas', 'Self-Attention sobre toda a sequência simultaneamente', 'Células LSTM bidirecionais', 'Conexões residuais profundas'],
    ans: 1,
    exp: 'RNNs processam tokens sequencialmente (t depende de t-1), impedindo a paralelização. Transformers calculam atenção entre todos os pares de tokens simultaneamente, acelerando dramaticamente o treinamento em GPUs.'
  },
  {
    q: 'Qual é o propósito do Positional Encoding no Transformer?',
    opts: [
      'Para normalizar embeddings de tokens',
      'Para injetar informação de posição/ordem, pois atenção é invariante a permutações',
      'Para reduzir a dimensionalidade dos embeddings',
      'Para separar tokens de texto de tokens de imagem'
    ],
    ans: 1,
    exp: 'A self-attention pura trata uma sequência como um conjunto não ordenado. O Positional Encoding adiciona vetores senoidais (ou aprendidos) que codificam a posição absoluta ou relativa de cada token na sequência.'
  },
  {
    q: 'Qual é a diferença arquitetural fundamental entre BERT e GPT?',
    opts: [
      'BERT usa CNN; GPT usa Transformer',
      'BERT é apenas encoder (bidirecional); GPT é apenas decoder (causal/unidirecional)',
      'BERT gera texto; GPT classifica texto',
      'BERT sempre usa 12 camadas; GPT sempre usa 24'
    ],
    ans: 1,
    exp: 'BERT vê o contexto esquerdo E direito (mascaramento aleatório durante o treinamento) → excelente para classificação, NER, QA. GPT vê apenas o contexto anterior (máscara causal) → geração autorregressiva de texto token por token.'
  },
  {
    q: 'O que é a Feed-Forward Network (FFN) dentro de um bloco Transformer?',
    opts: [
      'Uma rede recorrente para processar sequências dentro do bloco',
      'Duas camadas lineares com ativação não-linear aplicadas independentemente a cada posição',
      'A projeção final que mapeia embeddings para o vocabulário',
      'O mecanismo de cross-attention entre encoder e decoder'
    ],
    ans: 1,
    exp: 'FFN(x) = max(0, xW₁+b₁)W₂+b₂, aplicada posição por posição (cada token independentemente). Com d_model=512 e d_ff=2048, a FFN é 4× mais larga — a maioria dos parâmetros do Transformer está aqui.'
  },
  {
    q: 'O que são as Leis de Escala para LLMs (Kaplan et al., 2020)?',
    opts: [
      'Regras para escalar modelos sem exceder os limites de memória da GPU',
      'Relações empíricas de lei de potência entre perda, número de parâmetros e tamanho dos dados de treinamento',
      'Diretrizes para escalar a taxa de aprendizado proporcionalmente ao tamanho do batch',
      'Fórmulas para calcular o número ideal de camadas dado um orçamento de parâmetros'
    ],
    ans: 1,
    exp: 'As Leis de Escala mostram que L(N,D) segue leis de potência: mais parâmetros N e mais dados D reduzem a perda de forma previsível. Isso guiou o desenvolvimento do GPT-3/4: treinar modelos maiores com mais dados é consistentemente vantajoso.'
  }
]);
</script>
