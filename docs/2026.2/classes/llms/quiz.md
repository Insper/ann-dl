<div id="quiz-llms"></div>
<script>
buildQuiz('llms', 'Large Language Models', [
  {
    q: 'What is the pre-training objective of GPT-style models?',
    opts: [
      'Predicting the class of a text document',
      'Predicting the next token given the preceding sequence (next token prediction)',
      'Comparing sentence pairs to determine entailment',
      'Answering multiple-choice questions'
    ],
    ans: 1,
    exp: 'GPT (decoder-only) is trained with causal language modeling: p(x_t | x_1,...,x_{t-1}). Self-supervised — labels are the tokens themselves. Scales to trillions of tokens without manual annotation.'
  },
  {
    q: 'What are emergent abilities in LLMs?',
    opts: [
      'Capabilities explicitly programmed during fine-tuning',
      'Capabilities that appear abruptly in models above a certain scale, absent in smaller models',
      'The generation speed that emerges with hardware optimizations',
      'The ability of a model to reduce its own size'
    ],
    ans: 1,
    exp: 'Wei et al. (2022) documented that capabilities like multi-digit arithmetic, chain-of-thought reasoning, and translation emerge non-linearly when crossing scale thresholds. They cannot be explained by simple interpolation from smaller model capabilities.'
  },
  {
    q: 'What is RLHF (Reinforcement Learning from Human Feedback)?',
    opts: [
      'Training an RL agent to play games using human feedback as reward',
      'Aligning LLMs to human preferences using a reward model trained on human rankings',
      'A data augmentation technique using real user feedback',
      'Training the tokenizer using linguist preferences'
    ],
    ans: 1,
    exp: 'RLHF (InstructGPT, 2022): 1) SFT on human demonstrations; 2) Reward Model learning to rank responses by human preference; 3) PPO/DPO to optimize the LLM against the RM. Transforms a text predictor into a helpful assistant.'
  },
  {
    q: 'What is Mixture of Experts (MoE) in modern LLMs?',
    opts: [
      'An ensemble of multiple different LLMs voting on the final response',
      'Replacing each FFN with E independent expert networks, with a router that activates only k per token',
      'Fine-tuning different model parts by different expert teams',
      'A quantization method using experts for different weight value ranges'
    ],
    ans: 1,
    exp: 'MoE (e.g., Mixtral 8×7B, DeepSeek-V3 671B/37B active): each FFN layer has E experts and a router that selects top-k per token. More total capacity with the same computational cost per forward pass.'
  },
  {
    q: 'What is Chain-of-Thought (CoT) prompting?',
    opts: [
      'Chaining multiple LLMs in series to solve complex tasks',
      'Including intermediate reasoning steps in the prompt to guide the model to solve problems step by step',
      'A tokenization technique linking semantically related words',
      'Training the model with formal logical reasoning examples'
    ],
    ans: 1,
    exp: 'CoT (Wei et al., 2022): adding "Let\'s think step by step" or examples with explicit reasoning dramatically improves math, logic, and coding. The model uses intermediate tokens as working memory.'
  }
]);
</script>
