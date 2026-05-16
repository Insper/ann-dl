<div id="quiz-deep-learning"></div>
<script>
buildQuiz('deep-learning', 'Deep Learning Layers', [
  {
    q: 'What does a Dense (Fully Connected) layer do?',
    opts: [
      'Connects only adjacent neurons, preserving spatial locality',
      'Connects each input neuron to each output neuron via weight matrix W and bias b',
      'Applies 2D convolution over image data',
      'Normalizes activations per batch during training'
    ],
    ans: 1,
    exp: 'Dense(x) = activation(Wx + b). Each output depends on ALL inputs. Effective for capturing global relationships but inefficient for spatially structured data (images) since it ignores locality.'
  },
  {
    q: 'What does Pooling (e.g., Max Pooling) accomplish?',
    opts: [
      'Increases spatial resolution via interpolation',
      'Reduces spatial dimensions by selecting the maximum value per region, providing translation invariance',
      'Normalizes activations per channel',
      'Increases the number of feature map channels'
    ],
    ans: 1,
    exp: 'Max Pooling 2×2 with stride 2 halves the resolution by selecting the maximum value in each 2×2 block. Reduces parameters and memory, and provides invariance to small translations.'
  },
  {
    q: 'What is the purpose of skip connections (residual connections) in ResNets?',
    opts: [
      'To speed up inference by skipping unnecessary layers',
      'To allow gradients to flow directly through layers, mitigating vanishing gradients in very deep networks',
      'To reduce the number of model parameters',
      'To apply selective dropout on intermediate layers'
    ],
    ans: 1,
    exp: 'F(x) + x: the skip connection adds the input directly to the block output. Gradients can flow through the addition without passing through convolutions, solving the vanishing gradient problem and enabling 100+ layer networks.'
  },
  {
    q: 'What is an Embedding in deep learning?',
    opts: [
      'A lossless image compression technique',
      'A mapping of discrete entities (words, users, items) to dense continuous vectors of fixed dimension',
      'The flatten operation that linearizes 2D tensors to 1D',
      'A type of regularization that projects gradients'
    ],
    ans: 1,
    exp: 'Embeddings map discrete indices to dense vectors of dimension d (e.g., 256). Similar entities are close in embedding space. Essential in NLP (word2vec, BERT) and recommendation systems.'
  },
  {
    q: 'What does the Flatten layer do in a CNN before the final dense layer?',
    opts: [
      'Applies global normalization of feature maps',
      'Converts the multi-dimensional tensor (e.g., 7×7×512) into a 1D vector to feed dense layers',
      'Reduces the number of channels via global pooling',
      'Applies dropout over the feature maps'
    ],
    ans: 1,
    exp: 'Flatten(tensor shape [B, C, H, W]) → [B, C×H×W]. Required to connect convolutional layers to the dense classification head. Alternative: Global Average Pooling, which computes the mean per channel.'
  }
]);
</script>
