<div id="quiz-cnn"></div>
<script>
buildQuiz('cnn', 'Convolutional Neural Networks', [
  {
    q: 'A conv layer with 32×32 input, 3×3 kernel, padding=1, stride=1 produces output of what size?',
    opts: ['30×30', '32×32', '34×34', '16×16'],
    ans: 1,
    exp: 'H_out = (H_in + 2p - K) / s + 1 = (32 + 2 - 3)/1 + 1 = 32. Padding=1 preserves spatial dimensions with a 3×3 kernel — known as "same padding".'
  },
  {
    q: 'What is weight sharing in CNNs?',
    opts: [
      'Sharing weights between different networks on the same task',
      'The same kernel is applied at all spatial positions of the input, drastically reducing the number of parameters',
      'Initializing CNN weights with dense network weights',
      'Using dropout so neurons share representations'
    ],
    ans: 1,
    exp: 'A convolutional filter (e.g., 3×3×3 = 27 parameters) slides over the entire image. Applied separately to each position (64×64 = 4096 regions) would be 4096×27 parameters. Weight sharing is the key to CNN efficiency.'
  },
  {
    q: 'Why are CNNs more efficient than MLPs for images?',
    opts: [
      'CNNs use faster activation functions to compute',
      'CNNs exploit spatial locality and weight sharing — nearby pixels are correlated; the same pattern can occur anywhere',
      'CNNs do not require GPUs for training',
      'CNNs do not need backpropagation'
    ],
    ans: 1,
    exp: 'A 224×224×3 image has 150k inputs → an MLP with 1k neurons = 150M parameters. A CNN with 3×3 filters learns reusable local feature detectors at any position, with far fewer parameters.'
  },
  {
    q: 'What are feature maps in a CNN?',
    opts: [
      'A map of pixel positions with high intensity',
      'The activation outputs of each convolutional filter, representing the filter\'s response at each spatial position',
      'The weight matrix of a convolutional layer',
      'The gradient map during backpropagation'
    ],
    ans: 1,
    exp: 'Each convolutional filter produces a feature map: a 2D grid indicating where in the image a particular pattern (edge, texture, etc.) was detected. With C_out filters, we get C_out feature maps per layer.'
  },
  {
    q: 'Which architecture introduced residual (skip) connections to enable 100+ layer networks?',
    opts: ['AlexNet', 'VGGNet', 'ResNet (He et al., 2015)', 'Inception/GoogLeNet'],
    ans: 2,
    exp: 'ResNet (Deep Residual Learning, He et al., 2015) introduced F(x)+x skip connections, enabling training of 152+ layer networks that surpassed all benchmarks of the time and inspired virtually all subsequent deep architectures.'
  }
]);
</script>
