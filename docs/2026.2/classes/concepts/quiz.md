<div id="quiz-concepts"></div>
<script>
buildQuiz('concepts', 'AI Concepts', [
  {
    q: 'What best differentiates Deep Learning from classical Machine Learning?',
    opts: [
      'DL uses deep neural networks to automatically learn hierarchical representations from raw data',
      'DL does not require training data',
      'DL is exclusively for computer vision tasks',
      'DL uses only symbolic rules defined by domain experts'
    ],
    ans: 0,
    exp: 'Deep Learning uses multiple processing layers to learn progressively more abstract representations, eliminating the need for manual feature engineering. Classical ML typically requires handcrafted features.'
  },
  {
    q: 'Which AI paradigm combines symbolic reasoning with neural networks?',
    opts: ['Connectionist AI', 'Symbolic AI', 'Neuro-Symbolic AI', 'Reinforcement Learning'],
    ans: 2,
    exp: 'Neuro-Symbolic AI combines the strengths of logical symbolic reasoning with the learning capability of neural networks, as in Knowledge-Augmented Language Models and Graph Neural Networks.'
  },
  {
    q: 'In supervised learning, what does the model receive during training?',
    opts: [
      'Only unlabeled data',
      'Input-output pairs with correct labels',
      'Rewards from the environment',
      'Manually coded logical rules'
    ],
    ans: 1,
    exp: 'In supervised learning, each training sample has an associated correct label. The model learns to map inputs to outputs by minimizing prediction error. Examples: classification, regression.'
  },
  {
    q: 'Which ML technique is most appropriate for training an agent to play a video game?',
    opts: ['Linear Regression', 'K-Means Clustering', 'Reinforcement Learning', 'PCA'],
    ans: 2,
    exp: 'Reinforcement Learning trains agents that interact with an environment, receiving rewards or penalties, learning optimal action policies — ideal for sequential decision-making tasks like games.'
  },
  {
    q: 'What is the Turing Test?',
    opts: [
      'A benchmark for measuring LLM perplexity',
      'A criterion to assess whether a machine exhibits intelligent behavior indistinguishable from a human',
      'An accuracy metric for classifiers',
      'A regularization method for neural networks'
    ],
    ans: 1,
    exp: 'Proposed by Alan Turing in 1950, the test evaluates whether a human judge can distinguish machine responses from human responses in a text conversation. If the judge cannot, the machine passes the test.'
  }
]);
</script>
