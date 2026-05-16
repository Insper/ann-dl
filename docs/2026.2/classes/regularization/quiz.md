<div id="quiz-regularization"></div>
<script>
buildQuiz('regularization', 'Regularization', [
  {
    q: 'What does Dropout do during training?',
    opts: [
      'Progressively reduces the learning rate',
      'Randomly zeroes neuron activations with probability p, forcing redundant representations',
      'Adds an L2 penalty to the weights',
      'Stops training when validation loss stops improving'
    ],
    ans: 1,
    exp: 'Dropout (Srivastava et al., 2014) deactivates neurons randomly during each forward pass. This prevents co-adaptation between neurons and acts as an ensemble of sub-networks. At inference, all neurons are active and weights are scaled by (1-p).'
  },
  {
    q: 'What is the bias-variance tradeoff?',
    opts: [
      'The tradeoff between using bias or weights in neural networks',
      'Complex models have low bias but high variance (overfitting); simple models have high bias but low variance (underfitting)',
      'The balance between learning rate and batch size',
      'The relationship between training accuracy and inference speed'
    ],
    ans: 1,
    exp: 'Bias = systematic error (wrong assumptions). Variance = sensitivity to training data fluctuations. The goal is to minimize both by finding optimal model complexity. Regularization, more data, or simpler models reduce variance.'
  },
  {
    q: 'What does L2 regularization (Weight Decay) do?',
    opts: [
      'Zeros small weights, producing sparsity',
      'Adds a penalty proportional to the square of the weights to the loss function, shrinking them toward zero',
      'Normalizes activations in each layer',
      'Clips the gradient to a maximum value during training'
    ],
    ans: 1,
    exp: 'L2 adds λ||w||² to the loss, making the gradient include -λw. This "shrinks" all weights toward zero but rarely to exactly zero. Equivalent to a Gaussian prior on the weights.'
  },
  {
    q: 'What is Early Stopping?',
    opts: [
      'Stopping training after a fixed number of epochs regardless of performance',
      'Monitoring validation loss and stopping when it starts increasing, before overfitting',
      'Reducing the learning rate after each epoch',
      'Removing deep layers if the model does not converge'
    ],
    ans: 1,
    exp: 'Early stopping monitors the validation metric and saves the best checkpoint. When validation worsens for N consecutive epochs (patience), training stops and the best model is restored.'
  },
  {
    q: 'What is Batch Normalization and what is its main benefit?',
    opts: [
      'Normalizes input data to [0,1] before training',
      'Normalizes each mini-batch\'s activations, accelerating training and reducing sensitivity to learning rate',
      'Shuffles mini-batches to reduce correlation',
      'Normalizes gradients to prevent explosion'
    ],
    ans: 1,
    exp: 'Batch Norm (Ioffe & Szegedy, 2015) normalizes each layer\'s activations to mean 0, variance 1 per mini-batch, followed by learnable scale γ and shift β. Enables larger learning rates and mitigates vanishing gradients.'
  }
]);
</script>
