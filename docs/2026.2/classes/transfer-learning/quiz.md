<div id="quiz-transfer-learning"></div>
<script>
buildQuiz('transfer-learning', 'Transfer Learning & Fine-Tuning', [
  {
    q: 'In Transfer Learning, what does "freezing" layers mean?',
    opts: [
      'Saving weights to disk for later use',
      'Preventing weights from being updated during training, preserving pre-trained representations',
      'Initializing weights with low temperature (close to zero)',
      'Applying 100% dropout rate to those layers'
    ],
    ans: 1,
    exp: 'Freezing sets requires_grad=False for the parameters. Reduces memory and avoids destroying useful pre-trained representations. Early layers (edge detectors, texture patterns) rarely need to be updated for new tasks.'
  },
  {
    q: 'What does LoRA (Low-Rank Adaptation) do?',
    opts: [
      'Removes layers from the model to reduce memory',
      'Adds trainable low-rank matrices (ΔW = BA) in parallel to frozen weights',
      'Trains only the bias of each layer, freezing the weights',
      'Quantizes weights to 4 bits to reduce memory'
    ],
    ans: 1,
    exp: 'LoRA freezes W₀ and adds W = W₀ + BA where B∈R^{d×r}, A∈R^{r×k} with r≪min(d,k). For a 7B-parameter model with r=8, only ~0.1% of parameters are trainable, with quality comparable to full fine-tuning.'
  },
  {
    q: 'What is the difference between Feature Extraction and Full Fine-Tuning?',
    opts: [
      'Feature Extraction trains all layers; Full Fine-Tuning freezes everything',
      'Feature Extraction freezes the backbone and trains only the head; Full Fine-Tuning updates all weights',
      'They are equivalent with different learning rates',
      'Feature Extraction uses gradient descent; Full Fine-Tuning uses evolutionary methods'
    ],
    ans: 1,
    exp: 'Feature Extraction: frozen backbone → extracted features → new classifier trained from scratch. Fast, few data needed. Full FT: all weights adjusted with small lr. Better quality, more data and GPU required.'
  },
  {
    q: 'What is QLoRA?',
    opts: [
      'LoRA applied exclusively to the Q (query) attention layer',
      'LoRA combined with 4-bit quantization of the base model, drastically reducing VRAM usage',
      'A quantized version of the Adam optimizer for faster LoRA training',
      'Q-Learning combined with LoRA for reinforcement fine-tuning'
    ],
    ans: 1,
    exp: 'QLoRA (Dettmers et al., 2023) quantizes the base model to NF4 (4 bits) — reducing memory ~4×. LoRA adapters remain in fp16. Enables fine-tuning LLaMA-2 70B on a single 48GB GPU.'
  },
  {
    q: 'When is Domain Adaptation necessary before task fine-tuning?',
    opts: [
      'Never — direct task fine-tuning is always sufficient',
      'When the target domain (medical, legal, code) has vocabulary and style very different from pre-training data',
      'When the model has more than 1B parameters',
      'When the fine-tuning data is imbalanced'
    ],
    ans: 1,
    exp: 'Domain Adaptation (continued pre-training) on unlabeled domain text helps the model learn domain-specific terminology and style. Example: BioGPT does domain adaptation on medical literature before fine-tuning on clinical NER.'
  }
]);
</script>
