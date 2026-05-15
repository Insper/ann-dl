### Introduction to Stable Diffusion

Stable Diffusion is a type of generative AI model based on **diffusion models**, specifically a Latent Diffusion Model (LDM). It generates images from text prompts by learning to reverse a noise-adding process. The core idea comes from Denoising Diffusion Probabilistic Models (DDPMs), where data (e.g., images) is gradually corrupted with noise (forward process), and a neural network learns to reverse this by predicting and removing noise (backward process). This allows sampling new data from noise.

Key components in Stable Diffusion:

- **VAE (Variational Autoencoder)**: Compresses images to a lower-dimensional latent space for efficiency (e.g., from 512x512 pixels to 64x64 latents).
- **U-Net**: A CNN-like architecture (with attention for text conditioning) that predicts noise at each step.
- **Text Encoder** (e.g., CLIP): Converts prompts to embeddings for conditioning.
- **Scheduler**: Controls the noise addition/removal schedule (e.g., linear beta schedule).

The "forward pass" refers to the diffusion (noising) process during training. The "backward pass" is the reverse diffusion (denoising) for generation, but training involves backpropagation to update the model. I'll focus on the math for the core DDPM, then note Stable Diffusion's extensions. Assume images as vectors \( \mathbf{x} \in \mathbb{R}^D \) (flattened), time steps \( T \) (e.g., 1000).

### Forward Pass in Diffusion Models (Noising Process)

The forward pass is a Markov chain that progressively adds Gaussian noise to the data until it's pure noise. This is non-learnable; it's fixed.

#### Key Notations:
- Clean data: \( \mathbf{x}_0 \sim q(\mathbf{x}_0) \) (from dataset).
- Time step: \( t = 1 \) to \( T \).
- Noise schedule: \( \beta_t \in (0,1) \) (variance, often increasing linearly from ~0.0001 to 0.02).
- Cumulative: \( \alpha_t = 1 - \beta_t \), \( \bar{\alpha}_t = \prod_{s=1}^t \alpha_s \).
- Noise: \( \mathbf{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I}) \).

#### Forward Transition:
At each step:

\[
q(\mathbf{x}_t | \mathbf{x}_{t-1}) = \mathcal{N}(\mathbf{x}_t; \sqrt{\alpha_t} \mathbf{x}_{t-1}, \beta_t \mathbf{I})
\]

Direct sampling from \( \mathbf{x}_0 \) to any \( \mathbf{x}_t \) (key for training):

\[
\mathbf{x}_t = \sqrt{\bar{\alpha}_t} \mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t} \mathbf{\epsilon}
\]

- **How to arrive at this**: This closed-form derives from reparameterizing the Gaussian transitions. Starting from \( \mathbf{x}_1 = \sqrt{\alpha_1} \mathbf{x}_0 + \sqrt{\beta_1} \mathbf{\epsilon}_1 \), inductively, the mean scales by \( \sqrt{\bar{\alpha}_t} \), and variance accumulates to \( 1 - \bar{\alpha}_t \). At \( t=T \), \( \mathbf{x}_T \approx \mathcal{N}(\mathbf{0}, \mathbf{I}) \).

In Stable Diffusion, this happens in latent space: First, encode image \( \mathbf{z}_0 = \text{Encoder}(\mathbf{x}_0) \), then diffuse \( \mathbf{z}_t \).

### Backward Pass in Diffusion Models (Denoising Process)

The backward pass learns to reverse the forward process, starting from noise \( \mathbf{x}_T \) and iteratively denoising to \( \mathbf{x}_0 \). A neural network \( \mathbf{\epsilon}_\theta(\mathbf{x}_t, t) \) (e.g., U-Net) predicts the noise \( \mathbf{\epsilon} \) added at step t.

#### Key Notations:
- Reverse transition: \( p_\theta(\mathbf{x}_{t-1} | \mathbf{x}_t) = \mathcal{N}(\mathbf{x}_{t-1}; \mathbf{\mu}_\theta(\mathbf{x}_t, t), \sigma_t^2 \mathbf{I}) \).
- Predicted mean: \( \mathbf{\mu}_\theta(\mathbf{x}_t, t) = \frac{1}{\sqrt{\alpha_t}} \left( \mathbf{x}_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}} \mathbf{\epsilon}_\theta(\mathbf{x}_t, t) \right) \).
- Variance: \( \sigma_t^2 = \frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t} \beta_t \) (or simplified in DDPM).

#### Training Objective:
Minimize the difference between predicted and actual noise. Loss (simplified variational bound):

\[
L = \mathbb{E}_{t, \mathbf{x}_0, \mathbf{\epsilon}} \left[ \| \mathbf{\epsilon} - \mathbf{\epsilon}_\theta(\mathbf{x}_t, t) \|^2 \right]
\]

Where \( \mathbf{x}_t = \sqrt{\bar{\alpha}_t} \mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t} \mathbf{\epsilon} \), t uniform in [1,T].

- Backpropagation: Standard gradient descent on \( L \) w.r.t. \( \theta \) (U-Net params). The "backward pass" here means both the reverse sampling and the autograd backward for training.

#### Inference (Generation):
Start from \( \mathbf{x}_T \sim \mathcal{N}(\mathbf{0}, \mathbf{I}) \), then for t=T to 1:

\[
\mathbf{x}_{t-1} = \mathbf{\mu}_\theta(\mathbf{x}_t, t) + \sigma_t \mathbf{z}, \quad \mathbf{z} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})
\]

- **How to arrive at this**: The reverse is derived by matching the forward Gaussian posterior. The mean \( \mathbf{\mu}_\theta \) is obtained by Bayes' rule on the forward process, approximating the reverse with the noise predictor. The loss is a reweighted ELBO from the diffusion model's probabilistic framework.

In Stable Diffusion:
- Conditioning: \( \mathbf{\epsilon}_\theta(\mathbf{z}_t, t, \mathbf{c}) \), where \( \mathbf{c} \) is text embedding.
- After denoising \( \mathbf{z}_0 \), decode to image: \( \mathbf{x}_0 = \text{Decoder}(\mathbf{z}_0) \).
- Schedulers like DDIM allow fewer steps (e.g., 50 instead of 1000) by deterministic sampling.

### Example Code: Simple DDPM in PyTorch

Here's a minimal PyTorch example of a DDPM (basis for Stable Diffusion). It trains a small MLP (instead of U-Net for simplicity) to denoise 1D data (e.g., toy Gaussians). For images, replace with a U-Net and 2D tensors. This shows forward (noising), training loss, and backward (denoising inference + autograd).

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Simple noise predictor (MLP instead of U-Net for toy example)
class NoisePredictor(nn.Module):
    def __init__(self, dim=32):
        super(NoisePredictor, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(dim + 1, 64),  # Input: x_t + t (embedded)
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, dim)  # Output: predicted epsilon
        )

    def forward(self, x, t):
        t = t.unsqueeze(1) / 1000.0  # Normalize t
        input = torch.cat([x, t], dim=1)
        return self.fc(input)

# Diffusion parameters
T = 1000
beta = torch.linspace(1e-4, 0.02, T)
alpha = 1 - beta
alpha_bar = torch.cumprod(alpha, dim=0)

def forward_diffusion(x0, t, noise):
    sqrt_alpha_bar = torch.sqrt(alpha_bar[t-1]).view(-1, 1)  # t starts from 1
    sqrt_one_minus_alpha_bar = torch.sqrt(1 - alpha_bar[t-1]).view(-1, 1)
    return sqrt_alpha_bar * x0 + sqrt_one_minus_alpha_bar * noise

# Training loop snippet
model = NoisePredictor(dim=32)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Toy data: batch of 1D vectors from N(0,1)
batch_size = 64
x0 = torch.randn(batch_size, 32)  # Example data

for epoch in range(100):  # Simplified
    t = torch.randint(1, T+1, (batch_size,))
    noise = torch.randn_like(x0)
    xt = forward_diffusion(x0, t, noise)
    
    pred_noise = model(xt, t.float())
    loss = nn.MSELoss()(pred_noise, noise)
    
    optimizer.zero_grad()
    loss.backward()  # Backward pass (autograd)
    optimizer.step()

print("Training Loss:", loss.item())

# Inference: Denoise from pure noise
def denoise(model, shape):
    x = torch.randn(shape)  # Start from x_T
    for t in range(T, 0, -1):
        t_tensor = torch.full((shape[0],), t, dtype=torch.long)
        pred_noise = model(x, t_tensor.float())
        alpha_t = alpha[t-1]
        beta_t = beta[t-1]
        alpha_bar_t = alpha_bar[t-1]
        
        mu = (1 / torch.sqrt(alpha_t)) * (x - (beta_t / torch.sqrt(1 - alpha_bar_t)) * pred_noise)
        sigma = torch.sqrt(beta_t)  # Simplified
        x = mu + sigma * torch.randn_like(x) if t > 1 else mu
    return x

generated = denoise(model, (1, 32))
print("Generated Sample:", generated[0][:5])  # First 5 elements
```

#### Running this Code:
This trains on random data and generates new samples. Outputs might look like:
- Training Loss: 0.85 (decreases over epochs)
- Generated Sample: tensor([0.1234, -0.5678, 0.9101, ...])

For Stable Diffusion, use libraries like `diffusers` from Hugging Face for real implementation (e.g., `StableDiffusionPipeline`). The math scales up: U-Net predicts noise on latents, with cross-attention for text.

<!-- source: https://grok.com/chat/aeb2b344-5ad6-4b03-8ce1-a7afd6fbf018 -->


https://github.com/openai/CLIP
https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/multimodal/vlm/clip.html
https://openai.com/index/clip/
https://huggingface.co/docs/transformers/model_doc/clip
https://en.wikipedia.org/wiki/Contrastive_Language-Image_Pre-training
https://en.wikipedia.org/wiki/U-Net


[^3]: [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/){:target="_blank"}