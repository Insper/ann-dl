
Variational Autoencoders (VAEs) are generative models that learn to encode data into a lower-dimensional latent space and then decode it back to the original space. They consist of an encoder, which maps input data \( \mathbf{x} \) to a latent representation \( \mathbf{z} \), and a decoder, which reconstructs \( \mathbf{x} \) from \( \mathbf{z} \). VAEs are trained to maximize the evidence lower bound (ELBO), balancing reconstruction accuracy and latent space regularization.

The ELBO can be expressed as:

\[
\text{ELBO} = \mathbb{E}_{q(\mathbf{z}|\mathbf{x})}[\log p(\mathbf{x}|\mathbf{z})] - D_{KL}(q(\mathbf{z}|\mathbf{x}) || p(\mathbf{z}))
\]

Where:
- \( q(\mathbf{z}|\mathbf{x}) \) is the encoder's output (approximate posterior).
- \( p(\mathbf{x}|\mathbf{z}) \) is the decoder's output (likelihood).
- \( D_{KL} \) is the Kullback-Leibler divergence, measuring how much the approximate posterior diverges from the prior \( p(\mathbf{z}) \).

During training, VAEs optimize the ELBO using stochastic gradient descent, often employing reparameterization tricks to allow backpropagation through the stochastic layers.

### Backward Pass in VAEs