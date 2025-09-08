## Comparison

Here's a tabular comparison to highlight key differences:

| | Gradient Descent (GD) | Stochastic Gradient Descent (SGD) | Momentum | ADAM |
|--------------------|-----------------------|-------------------------------|-----------------------|-----------------------|
| **Computational Cost per Update** | High (full dataset) | Low (mini-batch) | Low (similar to SGD) | Medium (stores moments) |
| **Convergence Speed** | Slow (few updates) | Medium (many noisy updates) | Fast (accelerates in consistent directions) | Fast (adaptive + momentum) |
| **Stability/Noise** | High stability, low noise | Low stability, high noise | Medium stability, reduced oscillations | High stability, low effective noise |
| **Adaptivity** | None (fixed η) | None (fixed η, but can schedule) | Low (momentum helps indirectly) | High (per-parameter adaptation) |
| **Best For** | Small datasets, convex problems | Large datasets, online learning | Noisy gradients, ravine-like landscapes | Complex MLPs, sparse data |
| **Common Hyperparameters** | Learning rate (η) | Learning rate (η), batch size | Learning rate (η), momentum (β ~0.9) | Learning rate (η ~0.001), β1 (0.9), β2 (0.999), ε |
| **Limitations** | Scalability issues; stuck in local minima | Oscillations; requires tuning | Can overshoot; still fixed η | Potential overfitting; higher memory |

In summary,

- GD is basic but inefficient;
- SGD adds speed at the cost of noise; 
- Momentum smooths SGD; 
- and, ADAM offers the most automation and efficiency for MLPs, though SGD/Momentum may edge out in generalization for fine-tuned tasks.

Choice depends on dataset size, computational resources, and problem complexity—experiment with validation sets for best results.