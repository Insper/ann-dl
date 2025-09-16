### Introduction to Convolutional Neural Networks (CNNs)

Convolutional Neural Networks (CNNs) are a class of deep neural networks commonly used for image recognition, video analysis, and other tasks involving grid-like data (e.g., pixels in images). Unlike fully connected networks, CNNs exploit spatial hierarchies through **convolutions**, which apply learnable filters (kernels) to local regions of the input. This reduces parameters, enables translation invariance, and captures features like edges, textures, or objects.

A typical CNN architecture includes:

- **Convolutional layers**: Extract features via convolution.
- **Activation functions** (e.g., ReLU): Introduce non-linearity.
- **Pooling layers** (e.g., max pooling): Downsample to reduce spatial dimensions and computational load.
- **Fully connected layers**: For final classification or regression.
- **Output layer**: Often softmax for classification.

Training involves the **forward pass** (computing predictions) and **backward pass** (backpropagation to update weights via gradients). Below, I'll focus on the math for the convolutional layer, as it's the core of CNNs. I'll assume 2D convolution for images (input shape: batch_size × channels × height × width).


### Forward Pass in a Convolutional Layer

The forward pass computes the output feature map by sliding a kernel over the input.

#### Key Notations:

- Input: \( X \in \mathbb{R}^{B \times C_{in} \times H_{in} \times W_{in}} \) (batch size \( B \), input channels \( C_{in} \), height \( H_{in} \), width \( W_{in} \)).
- Kernel (weights): \( W \in \mathbb{R}^{C_{out} \times C_{in} \times K_h \times K_w} \) (output channels \( C_{out} \), kernel height \( K_h \), width \( K_w \)).
- Bias: \( b \in \mathbb{R}^{C_{out}} \) (one per output channel).
- Stride: \( s \) (step size for sliding the kernel).
- Padding: \( p \) (zeros added to borders to control output size).
- Output: \( Y \in \mathbb{R}^{B \times C_{out} \times H_{out} \times W_{out}} \), where \( H_{out} = \lfloor \frac{H_{in} + 2p - K_h}{s} \rfloor + 1 \), similarly for \( W_{out} \).

#### Convolution Operation:

For each position \( (i, j) \) in the output feature map, for batch item \( b \) and output channel \( c_{out} \):

\[
Y_{b, c_{out}, i, j} = \sum_{c_{in}=0}^{C_{in}-1} \sum_{m=0}^{K_h-1} \sum_{n=0}^{K_w-1} W_{c_{out}, c_{in}, m, n} \cdot X_{b, c_{in}, s \cdot i + m - p, s \cdot j + n - p} + b_{c_{out}}
\]

This is essentially a dot product between the kernel and a local patch of the input, summed over input channels, plus bias.

- **How to arrive at this**: The formula derives from cross-correlation (convolution is flipped cross-correlation, but in ML, we often use cross-correlation for simplicity). The indices ensure the kernel slides without going out of bounds (padding handles edges). For valid padding (\( p=0 \)), output shrinks; for same padding, \( p = \lfloor \frac{K-1}{2} \rfloor \) keeps size.

After convolution, apply activation: \( Y' = f(Y) \), e.g., ReLU: \( f(x) = \max(0, x) \).

Pooling (e.g., max pooling) over a window (size \( k \), stride \( s \)) takes the max value in each patch, reducing dimensions.

### Backward Pass in a Convolutional Layer

The backward pass computes gradients for weights, biases, and inputs using chain rule, to minimize loss \( L \) via gradient descent.

#### Key Notations:
- Incoming gradient from next layer: \( \frac{\partial L}{\partial Y} \in \mathbb{R}^{B \times C_{out} \times H_{out} \times W_{out}} \) (gradient w.r.t. output).
- We need:
  - \( \frac{\partial L}{\partial W} \) (weight gradient).
  - \( \frac{\partial L}{\partial b} \) (bias gradient).
  - \( \frac{\partial L}{\partial X} \) (input gradient, passed to previous layer).

#### Bias Gradient:
Simple sum over spatial dimensions and batch:

\[
\frac{\partial L}{\partial b_{c_{out}}} = \sum_{b=0}^{B-1} \sum_{i=0}^{H_{out}-1} \sum_{j=0}^{W_{out}-1} \frac{\partial L}{\partial Y_{b, c_{out}, i, j}}
\]

- **How to arrive at this**: Bias is added to every position, so its gradient is the sum of all output gradients per channel.

#### Weight Gradient:
Correlate input with output gradient:

\[
\frac{\partial L}{\partial W_{c_{out}, c_{in}, m, n}} = \sum_{b=0}^{B-1} \sum_{i=0}^{H_{out}-1} \sum_{j=0}^{W_{out}-1} \frac{\partial L}{\partial Y_{b, c_{out}, i, j}} \cdot X_{b, c_{in}, s \cdot i + m - p, s \cdot j + n - p}
\]

- **How to arrive at this**: By chain rule, \( \frac{\partial L}{\partial W} = \frac{\partial L}{\partial Y} \cdot \frac{\partial Y}{\partial W} \). Since \( Y \) is linear in \( W \), this is like convolving the input with the output gradient (but actually cross-correlation).

#### Input Gradient:
"Full" convolution of rotated kernel with output gradient (to propagate error back):

First, pad the output gradient if needed. Then:

\[
\frac{\partial L}{\partial X_{b, c_{in}, k, l}} = \sum_{c_{out}=0}^{C_{out}-1} \sum_{m=0}^{K_h-1} \sum_{n=0}^{K_w-1} \frac{\partial L}{\partial Y_{b, c_{out}, i, j}} \cdot W_{c_{out}, c_{in}, m, n}
\]

Where \( i = \lfloor \frac{k + p - m}{s} \rfloor \), \( j = \lfloor \frac{l + p - n}{s} \rfloor \), and only if the division is integer (i.e., aligns with stride).

- **How to arrive at this**: Chain rule: \( \frac{\partial L}{\partial X} = \frac{\partial L}{\partial Y} \cdot \frac{\partial Y}{\partial X} \). This requires "deconvolving" or transposing the convolution. In practice, it's implemented as convolving the output gradient with a rotated (180°) kernel, with appropriate padding.

If there's activation, multiply by its derivative (e.g., ReLU derivative: 1 if >0, else 0).

For pooling, backward pass upsamples the gradient (e.g., for max pooling, place gradient only at max position).



### Example



https://colab.research.google.com/drive/1AOONNT2DS0xP6k3thCq8pog1PoJUKTe6#scrollTo=b2tjGQ76GbeD