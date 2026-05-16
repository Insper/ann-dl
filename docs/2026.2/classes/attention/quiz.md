<div id="quiz-attention"></div>
<script>
buildQuiz('attention', 'Attention Mechanisms', [
  {
    q: 'In the attention mechanism, what do Query (Q), Key (K), and Value (V) represent?',
    opts: [
      'Q is the output, K is the convolutional kernel, V is the bias vector',
      'Q is what you are searching for, K is the index of each item, V is the content returned on a match',
      'Q, K, and V are three identical copies of the input without transformation',
      'Q is for generation, K is for classification, V is for regression'
    ],
    ans: 1,
    exp: 'Database analogy: Query is the search, Keys are item indices, Values are the content. Attention computes similarity Q·Kᵀ, normalizes with softmax, and uses the weights to combine the Values.'
  },
  {
    q: 'Why is the √d_k scaling factor necessary in attention?',
    opts: [
      'To ensure attention weights sum to 1 after softmax',
      'To prevent Q·Kᵀ products from becoming large in high dimensions, causing near-zero softmax gradients',
      'To speed up matrix computation on GPUs',
      'To normalize positional embeddings'
    ],
    ans: 1,
    exp: 'For vectors of dimension d_k with ~N(0,1) components, Q·Kᵀ has variance d_k. Dividing by √d_k stabilizes variance to 1, preventing softmax saturation into zero-gradient regions.'
  },
  {
    q: 'What is Multi-Head Attention?',
    opts: [
      'Attention applied across multiple layers sequentially',
      'Multiple parallel attention heads with independent projections, each capturing different types of relationship',
      'Attention with multiple query tokens simultaneously',
      'An ensemble of attention models with voting'
    ],
    ans: 1,
    exp: 'Multi-Head uses h independent heads with distinct projections W_Q^i, W_K^i, W_V^i. Each head can specialize in different relationships (syntactic, semantic, positional). Outputs are concatenated and projected.'
  },
  {
    q: 'What is Causal Self-Attention and where is it used?',
    opts: [
      'Attention where each token attends to all other tokens bidirectionally',
      'Self-attention with a triangular mask preventing future tokens from being seen — used in autoregressive models like GPT',
      'Attention where only adjacent tokens interact',
      'A special type of cross-attention between encoder and decoder'
    ],
    ans: 1,
    exp: 'The causal mask sets positions j > i to −∞ before softmax, ensuring token i only sees i and earlier tokens. Essential for autoregressive generation: the model cannot "cheat" by seeing future tokens.'
  },
  {
    q: 'What is the computational complexity of standard attention as a function of sequence length n?',
    opts: ['O(n)', 'O(n log n)', 'O(n²)', 'O(n³)'],
    ans: 2,
    exp: 'The Q·Kᵀ matrix has dimension n×n — each of the n tokens must compute attention with all other n tokens. This is O(n²) in both time and memory, making standard attention expensive for long sequences (e.g., documents, HD images).'
  }
]);
</script>
