<div id="quiz-vae"></div>
<script>
buildQuiz('vae', 'Variational Autoencoders (VAE)', [
  {
    q: 'What is the "reparameterization trick" in VAEs?',
    opts: [
      'Using a different loss function for the encoder and decoder',
      'Sampling z = μ + σ·ε where ε~N(0,I), separating the stochastic node to enable backpropagation',
      'Replacing the sampler with a deterministic function',
      'Normalizing encoder weights to unit norm'
    ],
    ans: 1,
    exp: 'Sampling directly z~N(μ,σ²) is not differentiable. The trick rewrites z = μ + σ·ε with ε~N(0,I), shifting randomness to an external node. Now ∂z/∂μ=1 and ∂z/∂σ=ε — backprop works through μ and σ.'
  },
  {
    q: 'What does a VAE encoder output, unlike a vanilla autoencoder?',
    opts: [
      'A single deterministic point z in latent space',
      'The parameters μ and σ of a Gaussian distribution q(z|x), not z directly',
      'A categorical distribution over a discrete latent vocabulary',
      'A binary vector selecting latent space dimensions'
    ],
    ans: 1,
    exp: 'Vanilla encoder: z = f(x) (deterministic). VAE encoder: μ(x), σ(x) (distribution statistics). This forces the latent space to be continuous and structured, enabling interpolation and new sample generation.'
  },
  {
    q: 'The ELBO loss of a VAE has two terms. What do they measure?',
    opts: [
      'Reconstruction loss (MSE) + KL divergence between q(z|x) and p(z)=N(0,I)',
      'Adversarial loss + pixel-level reconstruction loss',
      'Cross-entropy + contrastive loss',
      'Encoder loss + decoder loss computed separately'
    ],
    ans: 0,
    exp: 'ELBO = E[log p(x|z)] - KL(q(z|x)||p(z)). The first term maximizes reconstruction quality; the second regularizes the latent space to be close to N(0,I), ensuring structure and the ability to sample new instances.'
  },
  {
    q: 'How do you generate new samples from a trained VAE?',
    opts: [
      'Pass an image through the encoder and reconstruct it with the decoder',
      'Sample z~N(0,I) directly and decode: x̂ = Decoder(z)',
      'Interpolate between two dataset points in pixel space',
      'Use the inverted encoder to map from output to latent space'
    ],
    ans: 1,
    exp: 'The prior p(z)=N(0,I) is the sampling distribution. KL divergence during training aligns q(z|x) with it. At inference: sample z~N(0,I), apply the decoder, obtain a new plausible image without any input.'
  },
  {
    q: 'Why do VAE-generated images tend to be blurrier than GAN outputs?',
    opts: [
      'Because VAEs use fewer parameters than GANs',
      'Because reconstruction loss (e.g., MSE) penalizes all frequencies equally, leading to averages that appear blurry',
      'Because the VAE latent space is smaller than a GAN\'s',
      'Because VAEs do not use convolutional networks'
    ],
    ans: 1,
    exp: 'MSE approximates log-likelihood assuming pixel-wise Gaussian distribution — ignoring high-frequency perceptual structure. GANs learn an implicit perceptual loss via the discriminator. VAE-GAN hybrids or perceptual losses mitigate this.'
  }
]);
</script>
