### Vanilla Python Implementation

Below is a basic "vanilla" Python implementation of a toy CLIP-like model using only NumPy (no Torch or other frameworks for the core logic, keeping it simple and from-scratch). This simulates the contrastive loss computation and a minimal training loop on dummy data. In a real scenario, you'd need full encoders, but here we use random vectors as "embeddings" for illustration.

```python
import numpy as np

def compute_cosine_similarity(images, texts):
    # Normalize to unit length
    images = images / np.linalg.norm(images, axis=1, keepdims=True)
    texts = texts / np.linalg.norm(texts, axis=1, keepdims=True)
    return np.dot(images, texts.T)

def clip_loss(images, texts, temperature=0.07):
    N = images.shape[0]
    logits = compute_cosine_similarity(images, texts) / temperature
    
    # Image-to-text loss
    labels = np.arange(N)
    log_probs_i2t = logits - np.max(logits, axis=1, keepdims=True)
    log_probs_i2t = np.exp(log_probs_i2t) / np.sum(np.exp(log_probs_i2t), axis=1, keepdims=True)
    loss_i2t = -np.mean(np.log(log_probs_i2t[np.arange(N), labels]))
    
    # Text-to-image loss (symmetric)
    logits_t = logits.T
    log_probs_t2i = logits_t - np.max(logits_t, axis=1, keepdims=True)
    log_probs_t2i = np.exp(log_probs_t2i) / np.sum(np.exp(log_probs_t2i), axis=1, keepdims=True)
    loss_t2i = -np.mean(np.log(log_probs_t2i[np.arange(N), labels]))
    
    return (loss_i2t + loss_t2i) / 2

# Dummy data: 4 pairs, 5D embeddings
np.random.seed(42)
image_embeds = np.random.randn(4, 5)
text_embeds = image_embeds + np.random.randn(4, 5) * 0.1  # Slightly perturb texts to simulate matches

# "Train" by nudging text embeds toward images (toy gradient descent)
learning_rate = 0.01
for epoch in range(10):
    loss = clip_loss(image_embeds, text_embeds)
    print(f"Epoch {epoch}: Loss = {loss:.4f}")
    
    # Toy update: Move texts closer (not real backprop)
    similarity = compute_cosine_similarity(image_embeds, text_embeds)
    grad = (text_embeds - image_embeds) / np.linalg.norm(text_embeds - image_embeds, axis=1, keepdims=True)
    text_embeds -= learning_rate * grad
```

#### Explanation of the Code:

- **compute_cosine_similarity**: Computes dot products after normalization (cosine sim).
- **clip_loss**: Implements the symmetric contrastive loss. It calculates logits, applies stable softmax, and averages the cross-entropy losses for both directions.
- **Dummy Training Loop**: Starts with random embeddings, computes loss, and naively updates text embeddings to align better (not real optimization; in practice, use SGD on encoder params).
- To arrive at this: Start with the InfoNCE formula, derive the symmetric version from CLIP's paper, implement softmax carefully to avoid overflow (subtract max), and average losses.

If you run this, the loss decreases over epochs as alignments improve. For a full CLIP, replace dummy embeds with actual encoder outputs and train on real data. Let me know if you'd like expansions!

** This is a simplified simulation; real CLIP handles large batches (e.g., 32k) and uses distributed training.
