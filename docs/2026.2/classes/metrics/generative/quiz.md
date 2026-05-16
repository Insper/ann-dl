<div id="quiz-metrics-gen"></div>
<script>
buildQuiz('metrics-gen', 'Generative Model Metrics', [
  {
    q: 'What does FID (Fréchet Inception Distance) measure?',
    opts: [
      'Pixel-level reconstruction loss between real and generated images',
      'The distance between feature distributions of real and generated images in InceptionV3 activation space',
      'The accuracy of a classifier distinguishing real from generated images',
      'The average number of visual artifacts per generated image'
    ],
    ans: 1,
    exp: 'FID compares means and covariances of InceptionV3 features from real vs. generated images using Fréchet distance. Lower FID = more similar distributions = better quality. It is the standard metric for GANs and diffusion models.'
  },
  {
    q: 'What is mode collapse in GANs?',
    opts: [
      'When the discriminator stops learning',
      'When the generator learns to produce only a few plausible samples, ignoring the diversity of the real distribution',
      'When the generator\'s loss collapses to zero prematurely',
      'When the GAN becomes unstable and images become extremely noisy'
    ],
    ans: 1,
    exp: 'Mode collapse: the generator finds outputs that fool the discriminator and repeats them, losing diversity. FID and Inception Score capture this failure mode since generated samples cover fewer modes.'
  },
  {
    q: 'The Inception Score (IS) measures:',
    opts: [
      'Only visual quality, without considering diversity',
      'Both quality (clearly classifiable images) AND diversity (uniform distribution across classes)',
      'Only the diversity of the generated set',
      'The inference speed of the generator'
    ],
    ans: 1,
    exp: 'IS = exp(E[KL(p(y|x) || p(y))]). p(y|x) should be concentrated (clear image) and p(y) should be uniform (diversity). High IS means sharp AND varied images. However, it does not compare to real data — FID is more reliable.'
  },
  {
    q: 'What is CLIP Score used to evaluate?',
    opts: [
      'Perceptual quality of generated images independent of text',
      'Semantic alignment between the generated image and the text prompt',
      'Image generation speed in frames per second',
      'Autoencoder reconstruction fidelity'
    ],
    ans: 1,
    exp: 'CLIP Score measures the cosine similarity between the generated image embedding and the prompt embedding in CLIP\'s shared space. High CLIP Score indicates the image matches the prompt semantically — crucial for text-to-image evaluation.'
  },
  {
    q: 'Why are automatic text quality metrics (BLEU, ROUGE) insufficient for evaluating modern LLMs?',
    opts: [
      'Because they are too slow to compute',
      'Because they compare n-grams superficially and fail to capture semantics, creativity, or factuality',
      'Because they only work in English',
      'Because they require a GPU for computation'
    ],
    ans: 1,
    exp: 'BLEU/ROUGE measure word overlap with fixed references. For open-ended text (chatbots, reasoning, code), there are multiple correct answers and semantic quality is not captured by surface n-grams. Hence the rise of LLM-as-Judge evaluation.'
  }
]);
</script>
