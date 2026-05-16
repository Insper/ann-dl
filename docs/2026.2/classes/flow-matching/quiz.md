<div id="quiz-flow-matching"></div>
<script>
buildQuiz('flow-matching', 'Flow Matching', [
  {
    q: 'What does Flow Matching train the model to predict?',
    opts: [
      'The noise ε added at each diffusion step',
      'The velocity field v_θ(x_t, t) that moves particles from p_0 (noise) to p_1 (data)',
      'Token probabilities in language models',
      'The distribution of latents z in the VAE space'
    ],
    ans: 1,
    exp: 'dx/dt = v_θ(x, t). FM trains the velocity field defining the transformation trajectory. With the OT (Optimal Transport) path, the target velocity is simply x_1 - x_0 — constant along the straight trajectory.'
  },
  {
    q: 'What is the simplest conditional path in Flow Matching (CFM)?',
    opts: [
      'A stochastic Brownian trajectory as in DDPM',
      'Linear interpolation (straight line): x_t = (1-t)x_0 + t·x_1',
      'A Bezier curve between x_0 and x_1',
      'A great-circle arc in hyperspherical space'
    ],
    ans: 1,
    exp: 'The OT path in CFM is a straight line between noise (x_0~N(0,I)) and real data (x_1). The target velocity vector is constant: u_t(x|x_0,x_1) = x_1 - x_0. Far simpler than DDPM\'s β_t noise schedule.'
  },
  {
    q: 'What is the main inference advantage of Flow Matching over DDPM?',
    opts: [
      'FM generates higher-resolution images',
      'FM has straighter trajectories, needing 10-50 ODE steps vs 500-1000 for DDPM',
      'FM requires no GPU for inference',
      'FM can generate in real-time with no iterative steps'
    ],
    ans: 1,
    exp: 'Straight trajectories = less curvature = ODE integrators need fewer steps for good approximation. Empirical DDPM uses 50-1000 steps (DDIM). FM with Euler integration needs ~20-50 steps with comparable quality.'
  },
  {
    q: 'Which state-of-the-art image generation model uses Flow Matching with a Diffusion Transformer?',
    opts: ['Stable Diffusion 1.5', 'Midjourney v5', 'FLUX.1 (Black Forest Labs, 2024)', 'DALL-E 2'],
    ans: 2,
    exp: 'FLUX.1 (2024) combines: Flow Matching as training objective + MMDiT (Multi-Modal Diffusion Transformer) as architecture + VAE latent space. State-of-the-art in open-source generation with 12B parameters.'
  },
  {
    q: 'How is inference (generation) performed in Flow Matching?',
    opts: [
      'Sampling directly from the model without iteration',
      'Sample z~N(0,I) and integrate the ODE dx/dt = v_θ from t=0 to t=1',
      'Reverse denoising from t=T to t=0 as in DDPM',
      'Sample tokens autoregressively and decode'
    ],
    ans: 1,
    exp: 'FM goes from t=0 (noise) to t=1 (data). Each Euler step: x_{t+Δt} = x_t + Δt·v_θ(x_t, t). Opposite direction to DDPM (which goes T→0 in noise space). Heun or RK4 solvers need fewer steps.'
  }
]);
</script>
