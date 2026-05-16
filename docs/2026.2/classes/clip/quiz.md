<div id="quiz-clip"></div>
<script>
buildQuiz('clip', 'CLIP — Contrastive Language-Image Pretraining', [
  {
    q: 'What is CLIP\'s training objective?',
    opts: [
      'Generating images from text descriptions',
      'Maximizing cosine similarity of matched image-text pairs and minimizing it for mismatched pairs (contrastive learning)',
      'Classifying images into 1000 ImageNet categories',
      'Segmenting objects using text descriptions as prompts'
    ],
    ans: 1,
    exp: 'CLIP trains on 400M (image, text) pairs from the internet. For a batch of N pairs, it maximizes the diagonal (correct pairs) and minimizes the N²-N off-diagonal entries in the similarity matrix, aligning both embedding spaces.'
  },
  {
    q: 'What is zero-shot classification with CLIP?',
    opts: [
      'Classifying images with zero training examples using only textual descriptions of classes',
      'Using CLIP to classify text without fine-tuning',
      'Training a classifier with 0 positive examples per class',
      'Using CLIP without the text encoder'
    ],
    ans: 0,
    exp: 'To classify "cat" vs "dog" with CLIP: encode the image and compare with embeddings of "a photo of a cat" and "a photo of a dog". The highest cosine similarity defines the class — no task-specific classifier training needed.'
  },
  {
    q: 'What are the two encoders in CLIP?',
    opts: [
      'Image encoder (CNN/ViT) and text encoder (Transformer), both mapping to the same embedding space',
      'Image encoder and text decoder, for caption generation',
      'A shared encoder for both image and text',
      'Patch encoder and token encoder for local alignment'
    ],
    ans: 0,
    exp: 'Image Encoder: ViT or modified ResNet (e.g., ViT-L/14). Text Encoder: GPT-like Transformer. Both project to a shared d-dimensional space (e.g., 512 or 768) where cosine similarity is computed.'
  },
  {
    q: 'What is a known limitation of CLIP for fine-grained classification?',
    opts: [
      'CLIP does not work with color images',
      'CLIP struggles with fine-grained subcategories (car models, flower species) and requires careful prompt engineering',
      'CLIP only works in English',
      'CLIP cannot be used without a GPU'
    ],
    ans: 1,
    exp: 'CLIP generalizes well to common categories (animals, objects) but performs near-random for fine-grained distinctions (e.g., car model variants, aircraft types). Also sensitive to prompt wording: "a photo of a cat" vs "cat" can differ by 10+ percentage points.'
  },
  {
    q: 'How is CLIP used in Stable Diffusion?',
    opts: [
      'As an image generator using only the text encoder',
      'CLIP\'s text encoder converts the prompt into embeddings that condition the diffusion process via cross-attention in the U-Net/DiT',
      'CLIP is used as an adversarial discriminator to evaluate generated image quality',
      'CLIP compresses images to the VAE\'s latent space'
    ],
    ans: 1,
    exp: 'In SD: text → CLIP text encoder → conditioning embeddings → U-Net receives them via cross-attention at each resolution block. The embeddings guide which features are denoised. FLUX uses T5-XXL + CLIP for richer conditioning.'
  }
]);
</script>
