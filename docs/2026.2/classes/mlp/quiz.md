<div id="quiz-mlp"></div>
<script>
buildQuiz('mlp', 'Multi-Layer Perceptron', [
  {
    q: 'What is backpropagation?',
    opts: [
      'A forward pass algorithm that computes activations',
      'An algorithm that calculates gradients layer by layer using the chain rule to update weights',
      'A regularization technique that reverses weights',
      'The process of randomly initializing weights'
    ],
    ans: 1,
    exp: 'Backpropagation applies the calculus chain rule to propagate the loss gradient backward through the layers, computing ∂L/∂w for each weight. It is the foundation of neural network training.'
  },
  {
    q: 'The Universal Approximation Theorem states that:',
    opts: [
      'MLPs with infinitely many layers can approximate any function',
      'An MLP with a single hidden layer with enough neurons can approximate any continuous function',
      'MLPs only work for linear functions',
      'Any function can be approximated by a single sigmoid neuron'
    ],
    ans: 1,
    exp: 'Cybenko (1989) and Hornik (1991) proved that MLPs with one hidden layer and a non-linear activation can approximate any continuous function to arbitrary precision, given enough neurons.'
  },
  {
    q: 'Why are non-linear activation functions essential in MLPs?',
    opts: [
      'To speed up training',
      'Without non-linearity, composing linear layers collapses to a single linear transformation',
      'To reduce the number of parameters',
      'To guarantee gradients do not explode'
    ],
    ans: 1,
    exp: 'A composition of linear transformations W₂(W₁x) equals Wx — a single matrix. Without non-linear activations, stacking layers adds no expressive power. ReLU, tanh, and sigmoid break this collapse.'
  },
  {
    q: 'What advantage does ReLU have over Sigmoid in deep networks?',
    opts: [
      'ReLU is always less than 1, preventing gradient explosion',
      'ReLU does not suffer from vanishing gradients for positive inputs (derivative = 1)',
      'ReLU produces probabilistic outputs between 0 and 1',
      'ReLU is differentiable at all points'
    ],
    ans: 1,
    exp: 'For x > 0, ReLU(x) = x and ReLU\'(x) = 1 — the gradient does not shrink. Sigmoid\'(x) ≤ 0.25, causing vanishing gradients in deep networks. ReLU is also computationally cheaper.'
  },
  {
    q: 'What does the gradient ∂L/∂w represent in training?',
    opts: [
      'The loss value at the current point',
      'The direction and magnitude to change w to INCREASE the loss',
      'The optimal learning rate for convergence',
      'The model accuracy on the validation set'
    ],
    ans: 1,
    exp: 'The gradient points in the direction of steepest loss increase. To minimize loss, we update weights in the OPPOSITE direction: w ← w - η·∂L/∂w. This is the essence of gradient descent.'
  }
]);
</script>
