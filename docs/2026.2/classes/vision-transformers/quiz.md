<div id="quiz-vit"></div>
<script>
buildQuiz('vit', 'Vision Transformers', [
  {
    q: 'How does a Vision Transformer turn an image into a sequence the encoder can read?',
    opts: [
      'It applies several convolutional layers and then flattens the feature map',
      'It splits the image into fixed-size patches and linearly projects each patch into a token',
      'It feeds raw pixels one by one as individual tokens',
      'It computes a histogram of pixel intensities per channel'
    ],
    ans: 1,
    exp: 'ViT cuts the image into N non-overlapping P×P patches, flattens each, and projects it with a single shared linear layer (equivalent to a Conv2d with kernel=stride=P) → a sequence of N patch embeddings.'
  },
  {
    q: 'Why are positional embeddings necessary in a ViT?',
    opts: [
      'To reduce the number of patches and save memory',
      'Because self-attention is permutation-invariant, so without them the model cannot tell where each patch came from',
      'To normalize the patch embeddings before attention',
      'To convert the image to grayscale'
    ],
    ans: 1,
    exp: 'Pure self-attention treats the tokens as an unordered set. Learnable positional embeddings are added to each patch token so the model can recover spatial location.'
  },
  {
    q: 'What is the role of the [CLS] token in a ViT classifier?',
    opts: [
      'It stores the positional encoding for the whole image',
      'Its final representation is fed to the MLP head to produce class probabilities',
      'It marks the end of the patch sequence',
      'It is the learning rate schedule token'
    ],
    ans: 1,
    exp: 'A learnable [CLS] token is prepended to the sequence; after the encoder, only its final state is passed through the MLP head for classification.'
  },
  {
    q: 'Compared to a CNN, what is the main practical downside of ViT?',
    opts: [
      'It cannot process color images',
      'It has weak inductive bias, so it needs large-scale pretraining to match or beat CNNs',
      'It only works on square images smaller than 32×32',
      'It cannot be fine-tuned on downstream tasks'
    ],
    ans: 1,
    exp: 'Lacking locality/translation-equivariance priors, ViT underperforms CNNs on small datasets (e.g. ImageNet-1k alone) but surpasses them when pretrained on huge datasets (ImageNet-21k, JFT-300M).'
  },
  {
    q: 'Which statement about the ViT encoder is correct?',
    opts: [
      'It is a brand-new architecture unrelated to the text Transformer',
      'It is essentially the same Transformer encoder (Multi-Head Self-Attention + FFN blocks) used for text',
      'It replaces attention with convolutions inside each block',
      'It uses a causal mask like GPT'
    ],
    ans: 1,
    exp: 'After patch embedding, ViT reuses the standard Transformer encoder — stacked MHSA + FFN blocks with residual connections and LayerNorm (pre-norm, GELU). The novelty is only in how the input tokens are built.'
  }
]);
</script>
