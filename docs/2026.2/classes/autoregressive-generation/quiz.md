<div id="quiz-ar-gen"></div>
<script>
buildQuiz('ar-gen', 'Autoregressive Image Generation', [
  {
    q: 'What does VQ-GAN (Vector Quantized GAN) do?',
    opts: [
      'Generates high-resolution images adversarially without a quantization step',
      'Learns to compress images into discrete tokens via a codebook of K vectors, enabling images to be treated as sequences of indices',
      'Applies weight quantization to reduce model size',
      'Uses gradient quantization to speed up GAN training'
    ],
    ans: 1,
    exp: 'VQ-GAN: Encoder → feature map → quantization (each feature mapped to the nearest codebook vector) → integer index. Decoder reconstructs from quantized vectors. A 256×256 image → 16×16 = 256 integer tokens.'
  },
  {
    q: 'How does autoregressive image generation work after VQ-GAN?',
    opts: [
      'All tokens are generated simultaneously in parallel',
      'A Transformer generates codebook indices one by one, the same way GPT generates text',
      'The VQ-GAN discriminator generates tokens in decreasing order of importance',
      'A beam search algorithm selects the most likely token sequence'
    ],
    ans: 1,
    exp: 'With a trained codebook, token sequences represent images. A GPT-like Transformer learns p(t_i | t_1,...,t_{i-1}) — the next token given previous ones. Generation: sample token by token and decode with the VQ-GAN decoder.'
  },
  {
    q: 'How does MaskGIT accelerate generation compared to pure autoregressive?',
    opts: [
      'By using a smaller model in parallel',
      'By generating all masked tokens in parallel iteratively, revealing the most confident ones each step',
      'By skipping less important tokens based on attention',
      'By using a smaller codebook with fewer tokens per image'
    ],
    ans: 1,
    exp: 'Pure AR: N steps for N tokens. MaskGIT: starts with all tokens masked, predicts ALL simultaneously, reveals top-k most confident ones, repeats ~8-12 times. N=1024 tokens in 8-12 passes vs 1024 passes in pure AR.'
  },
  {
    q: 'What characterizes "any-to-any" models like Gemini and Chameleon?',
    opts: [
      'Can generate any file format from any text input',
      'Process text and image as a single mixed token sequence through the same Transformer, without modality-specific architectures',
      'Use multiple specialized models and combine outputs by voting',
      'Run on any hardware without special acceleration'
    ],
    ans: 1,
    exp: 'Any-to-any: text and image tokens coexist in the same sequence and pass through the same Transformer. [text][img_tok][text][img_tok] — the model learns to attend to any pattern. Gemini 2.0 Flash can interleave generated text and images natively.'
  },
  {
    q: 'What is the main disadvantage of pure autoregressive generation compared to diffusion for images?',
    opts: [
      'Lower visual quality than diffusion images in all cases',
      'Slow generation: N tokens = N sequential forward passes, while diffusion updates all pixels in parallel per step',
      'Does not support text conditioning',
      'Requires training a separate adversarial discriminator'
    ],
    ans: 1,
    exp: 'Pure AR: for 16×16 = 256 tokens, requires 256 passes. Diffusion: at each ODE step, all pixels/latents are updated in parallel (one U-Net/DiT forward pass). For large N, AR is much slower — hence MaskGIT and hybrid approaches.'
  }
]);
</script>
