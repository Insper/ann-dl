<div id="quiz-metrics-llm"></div>
<script>
buildQuiz('metrics-llm', 'LLM Metrics', [
  {
    q: 'What does Perplexity measure in language models?',
    opts: [
      'The model\'s accuracy on classification tasks',
      'How surprised the model is by test corpus tokens — lower PPL indicates better fit',
      'The number of parameters needed to generate coherent text',
      'The inference speed in tokens per second'
    ],
    ans: 1,
    exp: 'PPL = exp(-1/N · Σlog p(wᵢ|w<i)). PPL=10 means the model is, on average, as uncertain as between 10 equally likely choices. Low perplexity ≠ useful or aligned model.'
  },
  {
    q: 'What does the MMLU benchmark evaluate?',
    opts: [
      'Code generation in multiple programming languages',
      'Multidisciplinary knowledge across 57 subjects via multiple-choice questions',
      'Ability to follow instructions in multi-turn dialogues',
      'Advanced mathematical reasoning in calculus problems'
    ],
    ans: 1,
    exp: 'MMLU (Massive Multitask Language Understanding) covers 57 disciplines — medicine, law, history, physics, etc. It is the gold standard for measuring the breadth of knowledge acquired during pre-training.'
  },
  {
    q: 'Why is Chatbot Arena (ELO Rating) considered one of the most reliable LLM metrics?',
    opts: [
      'Because it uses a rigorous mathematical algorithm without human intervention',
      'Because it reflects real human preferences via A/B comparisons, capturing aspects automatic benchmarks miss',
      'Because it is faster to compute than other benchmarks',
      'Because it directly penalizes hallucinations in the metric'
    ],
    ans: 1,
    exp: 'In Chatbot Arena, real users compare two responses blind (without knowing which model) and vote for the better one. The ELO rating captures real utility, response style, and creativity — aspects impossible to measure automatically.'
  },
  {
    q: 'What is Faithfulness in RAG systems?',
    opts: [
      'The semantic similarity between question and answer',
      'The proportion of claims in the response that are supported by the retrieved documents',
      'The retriever\'s hit rate in finding relevant documents',
      'The response latency of the RAG system'
    ],
    ans: 1,
    exp: 'Faithfulness measures whether the LLM "invented" information or whether each claim in the response can be attributed to the retrieved context. High faithfulness indicates grounded responses, not hallucinations.'
  },
  {
    q: 'What is the main known bias in LLM-as-Judge evaluation?',
    opts: [
      'Judge models always prefer responses in English',
      'Tendency to prefer longer and more verbose responses, regardless of actual quality',
      'Inability to evaluate responses that contain code',
      'Always needing a human reference answer'
    ],
    ans: 1,
    exp: 'LLM evaluators tend to favor long, well-formatted responses and language similar to their own. Other biases include position bias (favoring the first response shown) and self-preference (favoring outputs from the same model family).'
  }
]);
</script>
