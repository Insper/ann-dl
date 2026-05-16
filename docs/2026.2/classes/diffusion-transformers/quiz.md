<div id="quiz-dit"></div>
<script>
buildQuiz('dit', 'Diffusion Transformers (DiT)', [
  {
    q: 'What does DiT replace compared to classic Stable Diffusion?',
    opts: [
      'Replaces the VAE with a purely convolutional encoder',
      'Replaces the U-Net with pure Transformer blocks, keeping diffusion/FM in latent space',
      'Replaces the text encoder with LSTM recurrent networks',
      'Replaces the diffusion process with explicit Flow Matching'
    ],
    ans: 1,
    exp: 'DiT (Peebles & Xie, 2023) keeps the VAE, text encoder, and diffusion process. The change is in the denoising network: from U-Net (conv + skip connections) to pure Transformer (attention + FFN). Result: better scaling with more parameters.'
  },
  {
    q: 'What is the "Patchify" operation in DiT?',
    opts: [
      'Splitting the image into patches before passing through the VAE',
      'Dividing the latent into p×p patches and projecting each to a token of dimension d_model',
      'Applying dropout to random patches of the latent during training',
      'Compressing the latent by a factor of 2× before the Transformer'
    ],
    ans: 1,
    exp: 'Patchify(latent H×W×C, patch p): divide into (H/p)×(W/p) patches → flatten → linear projection → N tokens of dim d_model. With p=2, FLUX processes 4096 tokens for 1024px images.'
  },
  {
    q: 'What is AdaLN (Adaptive Layer Normalization) in DiT?',
    opts: [
      'LayerNorm with fixed parameters learned during training',
      'LayerNorm whose γ and β parameters are dynamically predicted by the timestep and conditioning, enabling step-wise modulation',
      'Adaptive normalization to the batch size',
      'Batch Normalization adapted for variable-length sequences'
    ],
    ans: 1,
    exp: 'AdaLN(h, c) = γ(c) · norm(h) + β(c), where c = MLP(embed(t) + embed(class)). Unlike fixed LayerNorm weights, γ and β are generated dynamically — the model "knows" which timestep t it is at and adjusts normalization accordingly.'
  },
  {
    q: 'What distinguishes MMDiT (FLUX/SD3) from the original DiT?',
    opts: [
      'MMDiT uses convolutions while DiT uses attention',
      'MMDiT has separate streams for image and text tokens that share bidirectional attention — text sees image and image sees text',
      'MMDiT is significantly smaller than DiT in parameter count',
      'MMDiT replaces self-attention with pure cross-attention'
    ],
    ans: 1,
    exp: 'DiT injects text via cross-attention or concatenation. MMDiT (SD3, FLUX) processes text and image in parallel streams with separate weights, but combines Q, K, V from both in the same attention operation — bidirectional and much richer conditioning.'
  },
  {
    q: 'Why does DiT scale better than U-Net with more parameters?',
    opts: [
      'Because DiT uses fewer floating-point operations per parameter',
      'Because global attention from the first layer can fully utilize all additional capacity, while U-Net has hierarchical bottlenecks',
      'Because DiT does not require batch normalization and is more stable',
      'Because DiT processes fewer tokens than U-Net'
    ],
    ans: 1,
    exp: 'U-Net has pooling/upsampling creating information bottlenecks. Skip connections help but there is a hierarchical limit. DiT: full global attention (O(N²)) from block 1, no resolution hierarchy. Every new block/dimension adds fully exploitable capacity.'
  }
]);
</script>
