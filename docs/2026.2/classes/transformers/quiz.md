<div id="quiz-transformers"></div>
<script>
buildQuiz('transformers', 'Transformers', [
  {
    q: 'What replaced RNNs in the Transformer, enabling full parallelization?',
    opts: ['Dilated convolutions', 'Self-Attention over the entire sequence simultaneously', 'Bidirectional LSTM cells', 'Deep residual connections'],
    ans: 1,
    exp: 'RNNs process tokens sequentially (t depends on t-1), preventing parallelization. Transformers compute attention between all token pairs simultaneously, dramatically accelerating training on GPUs.'
  },
  {
    q: 'What is the purpose of Positional Encoding in the Transformer?',
    opts: [
      'To normalize token embeddings',
      'To inject position/order information, since attention is permutation-invariant',
      'To reduce the dimensionality of embeddings',
      'To separate text tokens from image tokens'
    ],
    ans: 1,
    exp: 'Pure self-attention treats a sequence as an unordered set. Positional Encoding adds sinusoidal (or learned) vectors that encode the absolute or relative position of each token in the sequence.'
  },
  {
    q: 'What is the key architectural difference between BERT and GPT?',
    opts: [
      'BERT uses CNN; GPT uses Transformer',
      'BERT is encoder-only (bidirectional); GPT is decoder-only (causal/unidirectional)',
      'BERT generates text; GPT classifies text',
      'BERT always uses 12 layers; GPT always uses 24'
    ],
    ans: 1,
    exp: 'BERT sees left AND right context (random masking during training) → excellent for classification, NER, QA. GPT sees only prior context (causal mask) → autoregressive text generation token by token.'
  },
  {
    q: 'What is the Feed-Forward Network (FFN) inside a Transformer block?',
    opts: [
      'A recurrent network for processing sequences within the block',
      'Two linear layers with non-linear activation applied independently to each position',
      'The final projection that maps embeddings to the vocabulary',
      'The cross-attention mechanism between encoder and decoder'
    ],
    ans: 1,
    exp: 'FFN(x) = max(0, xW₁+b₁)W₂+b₂, applied position-by-position (each token independently). With d_model=512 and d_ff=2048, the FFN is 4× wider — most Transformer parameters live here.'
  },
  {
    q: 'What are Scaling Laws for LLMs (Kaplan et al., 2020)?',
    opts: [
      'Rules for scaling models without exceeding GPU memory limits',
      'Empirical power-law relationships between loss, number of parameters, and training data size',
      'Guidelines for scaling learning rate proportionally to batch size',
      'Formulas for computing the optimal number of layers given a parameter budget'
    ],
    ans: 1,
    exp: 'Scaling Laws show that L(N,D) follows power laws: more parameters N and more data D predictably reduce loss. This guided GPT-3/4 development: training larger models on more data is consistently worth the investment.'
  }
]);
</script>
