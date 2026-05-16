<div id="quiz-ann"></div>
<script>
buildQuiz('ann', 'Neural Networks — History', [
  {
    q: 'Who proposed the first mathematical model of an artificial neuron in 1943?',
    opts: ['Rosenblatt and Minsky', 'McCulloch and Pitts', 'Hopfield and Hinton', 'LeCun and Bengio'],
    ans: 1,
    exp: 'Warren McCulloch and Walter Pitts proposed the MP neuron in 1943 — a binary model that fires when the weighted sum of inputs exceeds a threshold. It was the foundation of artificial neural networks.'
  },
  {
    q: 'What was the "AI Winter"?',
    opts: [
      'Cold weather that damaged data centers',
      'Periods of funding cuts and pessimism after excessive expectations went unmet',
      'The phase when genetic algorithms replaced neural networks',
      'The period when only quantum computers advanced'
    ],
    ans: 1,
    exp: 'Two periods (1974–80 and 1987–93) of stagnated progress — driven by Minsky/Papert\'s critique of the Perceptron, lack of compute, and unmet promises — caused severe retraction in AI research funding and interest.'
  },
  {
    q: 'What was the impact of AlexNet (2012) on deep learning history?',
    opts: [
      'It was the first model to use gradient descent',
      'It won ImageNet by a large margin using CNNs + GPUs, proving the effectiveness of DL at scale',
      'It introduced the attention mechanism that led to Transformers',
      'It created the first large-scale image dataset'
    ],
    ans: 1,
    exp: 'AlexNet (Krizhevsky, Sutskever, Hinton) reduced ImageNet top-5 error from ~26% to ~15% — an enormous margin. It showed GPU-trained deep CNNs decisively outperformed classical computer vision methods.'
  },
  {
    q: 'What is the Vanishing Gradient Problem?',
    opts: [
      'When gradients become so large that weights explode',
      'When gradients become so small during backpropagation that early layers learn extremely slowly',
      'When the learning rate is too high',
      'When the loss function oscillates without converging'
    ],
    ans: 1,
    exp: 'During backpropagation, gradients are multiplied by activation function derivatives (e.g., sigmoid: ≤0.25). In deep networks, this chain of small values approaches zero, preventing early layers from learning meaningful representations.'
  },
  {
    q: 'Which three researchers are often called the "Godfathers of Deep Learning"?',
    opts: [
      'Turing, Shannon, and von Neumann',
      'Minsky, McCarthy, and Simon',
      'LeCun, Bengio, and Hinton (Turing Award 2018)',
      'Goodfellow, Schmidhuber, and Hochreiter'
    ],
    ans: 2,
    exp: 'Yann LeCun (CNNs), Yoshua Bengio (neural language models, RNNs), and Geoffrey Hinton (Boltzmann machines, backprop) won the 2018 Turing Award for foundational contributions to deep learning.'
  }
]);
</script>
