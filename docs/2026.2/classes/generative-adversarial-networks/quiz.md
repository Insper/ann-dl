<div id="quiz-gan"></div>
<script>
buildQuiz('gan', 'Generative Adversarial Networks (GAN)', [
  {
    q: 'What is the Generator\'s objective in a GAN?',
    opts: [
      'To distinguish between real and fake images',
      'To produce samples so realistic the Discriminator cannot distinguish them from real data',
      'To classify the Discriminator\'s outputs into categories',
      'To minimize the KL divergence between learned and prior distributions'
    ],
    ans: 1,
    exp: 'Generator G: max E[log D(G(z))]. It wants the Discriminator to classify its outputs as real (D(G(z))→1). The adversarial game: G fools D, D learns to detect G, both improve iteratively.'
  },
  {
    q: 'What is mode collapse in GANs?',
    opts: [
      'When the Discriminator converges before the Generator',
      'When the Generator produces only a few diverse outputs, ignoring the variability of the real distribution',
      'When the GAN diverges and losses explode',
      'When the Generator memorizes training images'
    ],
    ans: 1,
    exp: 'Mode collapse: G finds a limited set of outputs that consistently fool D and repeats them. D cannot learn the distinction → bad equilibrium. Symptom: all generated images look similar. WGAN and other variants mitigate this.'
  },
  {
    q: 'How does Wasserstein GAN (WGAN) improve on classic GAN training?',
    opts: [
      'By using a deeper Discriminator with more layers',
      'By replacing JS divergence with Wasserstein distance, giving stable gradients even when distributions do not overlap',
      'By training Generator and Discriminator with more steps per epoch',
      'By adding L2 regularization to the Discriminator'
    ],
    ans: 1,
    exp: 'JS divergence saturates when G and D are very different (gradient ≈ 0). WGAN uses Earth Mover distance — it provides informative gradients even when distributions do not overlap, stabilizing training.'
  },
  {
    q: 'What is a Conditional GAN (cGAN)?',
    opts: [
      'A GAN conditioned on an Autoencoder\'s distribution',
      'A GAN where Generator and Discriminator receive additional information (e.g., class label) for controlled generation',
      'A GAN with paired discrimination (image-to-image)',
      'A GAN that uses attention to condition generation on features'
    ],
    ans: 1,
    exp: 'cGAN (Mirza & Osindero, 2014): G receives (z, y) and generates an image of class y. D receives (image, y) and classifies real/fake within the class. Enables generating specific MNIST digits, faces with controlled attributes, etc.'
  },
  {
    q: 'In the Discriminator\'s loss function, what does it maximize?',
    opts: [
      'The probability of real images being classified as fake',
      'E[log D(x)] + E[log(1 - D(G(z)))] — classifying real as real AND fake as fake',
      'Only the probability of rejecting generated images',
      'The norm of the Generator\'s gradients'
    ],
    ans: 1,
    exp: 'L_D = -E[log D(x)] - E[log(1-D(G(z)))]. Maximizing this means D(x)→1 for real images and D(G(z))→0 for generated ones. Equivalent to minimizing binary classification error.'
  }
]);
</script>
