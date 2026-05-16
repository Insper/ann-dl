<div id="quiz-perceptron"></div>
<script>
buildQuiz('perceptron', 'Perceptron', [
  {
    q: 'What is the weight update rule in Rosenblatt\'s Perceptron?',
    opts: [
      'w = w - η∇L (gradient descent)',
      'w = w + η·y·x (update only on misclassified samples)',
      'w = w × η (multiply by learning rate)',
      'w = w / ||w|| (normalize weights)'
    ],
    ans: 1,
    exp: 'The Perceptron rule updates weights only on errors: w ← w + η·y·x, where y is the true label (±1) and x the input. No update occurs for correctly classified samples.'
  },
  {
    q: 'Why can a single Perceptron not solve the XOR problem?',
    opts: [
      'Because the Perceptron cannot have a bias term',
      'Because XOR is not linearly separable — no single hyperplane separates the classes',
      'Because the Perceptron uses sigmoid activation instead of a step function',
      'Because XOR requires gradient descent, not the Perceptron rule'
    ],
    ans: 1,
    exp: 'The Perceptron learns a linear decision boundary (hyperplane). XOR outputs 1 for {(0,1), (1,0)} and 0 for {(0,0), (1,1)} — these classes cannot be separated by any straight line.'
  },
  {
    q: 'The Perceptron Convergence Theorem guarantees that:',
    opts: [
      'The Perceptron always converges regardless of the data',
      'If the data is linearly separable, the Perceptron converges in a finite number of updates',
      'The Perceptron converges faster with a larger learning rate',
      'The Perceptron converges only with normalized data'
    ],
    ans: 1,
    exp: 'Rosenblatt proved that if data is linearly separable, the Perceptron algorithm finds a solution in a finite number of steps. If the data is not separable, the algorithm oscillates indefinitely.'
  },
  {
    q: 'What activation function does the original Perceptron use?',
    opts: ['Sigmoid (logistic)', 'ReLU', 'Heaviside step function', 'Tanh'],
    ans: 2,
    exp: 'The Perceptron uses the step function: output 1 if Σwᵢxᵢ + b ≥ 0, else 0 (or -1). This produces binary decisions and is non-differentiable — hence the Perceptron rule instead of standard gradient descent.'
  },
  {
    q: 'What does the bias term (b) in the Perceptron represent?',
    opts: [
      'The model\'s learning rate',
      'An offset that allows the decision boundary to not pass through the origin',
      'The number of training epochs',
      'The norm of the weight vector'
    ],
    ans: 1,
    exp: 'The bias shifts the decision boundary (hyperplane) so it need not pass through the origin (0,...,0). Without bias, the model can only learn decision boundaries constrained to pass through the origin.'
  }
]);
</script>
