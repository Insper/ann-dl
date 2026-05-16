<div id="quiz-optimization"></div>
<script>
buildQuiz('optimization', 'Optimization', [
  {
    q: 'What does the learning rate η control in gradient descent?',
    opts: [
      'The number of training epochs',
      'The step size of parameter updates in the gradient direction',
      'The proportion of data used in each mini-batch',
      'The dropout rate applied during training'
    ],
    ans: 1,
    exp: 'θ ← θ - η·∇L. Too large a η causes oscillations or divergence; too small causes slow convergence. Learning rate scheduling (warm-up, cosine decay) helps find the right balance throughout training.'
  },
  {
    q: 'What is the main advantage of Mini-batch SGD over full Batch GD?',
    opts: [
      'Mini-batch always converges to a better minimum',
      'More frequent updates with noisy gradient estimates that can escape local minima',
      'Mini-batch does not require loading data into memory',
      'Mini-batch has more accurate gradients than Batch GD'
    ],
    ans: 1,
    exp: 'Mini-batch computes gradients on subsets (32–512 samples), giving frequent updates with controlled noise. The noise can help escape poor local minima and is efficient on parallel hardware (GPU).'
  },
  {
    q: 'What does Momentum do in gradient descent?',
    opts: [
      'Automatically adjusts learning rate per parameter',
      'Accumulates an exponential moving average of past gradients, smoothing the optimization trajectory',
      'Normalizes gradients to unit norm',
      'Adds noise to the gradient for exploration'
    ],
    ans: 1,
    exp: 'Momentum keeps a "velocity" v ← βv + (1-β)∇L and updates θ ← θ - η·v. It smooths oscillations and accelerates convergence in consistent directions — like a ball rolling downhill.'
  },
  {
    q: 'The Adam optimizer combines which two concepts?',
    opts: [
      'Batch GD and SGD',
      'First-order momentum and second-order moment estimation (adaptive per-parameter learning rates)',
      'L1 and L2 regularization',
      'Dropout and batch normalization'
    ],
    ans: 1,
    exp: 'Adam maintains m_t (gradient momentum) and v_t (squared gradient moving average — like RMSProp). It divides the update by sqrt(v_t), adapting the learning rate per parameter. Standard for training LLMs and vision models.'
  },
  {
    q: 'What is the difference between Adam and AdamW?',
    opts: [
      'AdamW uses a larger learning rate by default',
      'AdamW applies weight decay directly to weights, not via the gradient — making it mathematically correct',
      'AdamW does not use second-order momentum',
      'AdamW is slower but more accurate than Adam'
    ],
    ans: 1,
    exp: 'In original Adam, weight decay is added to the gradient before adaptive scaling, which distorts the decay. AdamW (Loshchilov & Hutter, 2019) applies decay directly to θ, separate from the adaptive update. Standard for Transformer training.'
  }
]);
</script>
