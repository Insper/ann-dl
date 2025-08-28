### Introduction to Convolutional Neural Networks (CNNs)

Convolutional Neural Networks (CNNs) are a class of deep neural networks commonly used for image recognition, video analysis, and other tasks involving grid-like data (e.g., pixels in images). Unlike fully connected networks, CNNs exploit spatial hierarchies through **convolutions**, which apply learnable filters (kernels) to local regions of the input. This reduces parameters, enables translation invariance, and captures features like edges, textures, or objects.

A typical CNN architecture includes:
- **Convolutional layers**: Extract features via convolution.
- **Activation functions** (e.g., ReLU): Introduce non-linearity.
- **Pooling layers** (e.g., max pooling): Downsample to reduce spatial dimensions and computational load.
- **Fully connected layers**: For final classification or regression.
- **Output layer**: Often softmax for classification.

Training involves the **forward pass** (computing predictions) and **backward pass** (backpropagation to update weights via gradients). Below, I'll focus on the math for the convolutional layer, as it's the core of CNNs. I'll assume 2D convolution for images (input shape: batch_size × channels × height × width).

<!-- 
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

### Example Code: Simple CNN in PyTorch

Here's a minimal PyTorch example demonstrating a CNN with one convolutional layer. It includes forward pass, computes a dummy loss, and shows backward pass gradients. (PyTorch handles the math automatically via autograd, but this illustrates the concepts.)

```python
import torch
import torch.nn as nn
import torch.optim as optim

# Define a simple CNN with one conv layer
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=3, stride=1, padding=1)  # 1 input channel (grayscale), 2 output channels
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc = nn.Linear(2 * 2 * 2, 2)  # Assuming input 4x4, after pool: 2x2 per channel

    def forward(self, x):
        x = self.conv(x)  # Convolution
        x = self.relu(x)  # Activation
        x = self.pool(x)  # Pooling
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)  # Fully connected
        return x

# Create model, input (batch=1, channel=1, 4x4 image), target
model = SimpleCNN()
input_tensor = torch.randn(1, 1, 4, 4, requires_grad=True)  # Random input
target = torch.tensor([[1.0, 0.0]])  # Dummy target for binary classification

# Forward pass
output = model(input_tensor)
print("Forward Pass Output:", output)

# Loss and backward pass
criterion = nn.MSELoss()
loss = criterion(output, target)
loss.backward()

# Print gradients (e.g., for conv weights and input)
print("Conv Weight Gradient:", model.conv.weight.grad[0, 0])  # Sample from first kernel
print("Input Gradient:", input_tensor.grad[0, 0])  # Gradient w.r.t. input
```

#### Running this Code:
If executed (e.g., in a Python environment with PyTorch), it would output something like:

- Forward Pass Output: tensor([[0.1234, -0.5678]], grad_fn=<AddmmBackward0>)  (random values)
- Conv Weight Gradient: tensor([[0.0100, 0.0200, 0.0300], [0.0400, 0.0500, 0.0600], [0.0700, 0.0800, 0.0900]])  (example gradients)
- Input Gradient: tensor([[0.0012, 0.0034, 0.0056, 0.0078], ...])  (propagated errors)

This shows how gradients are computed automatically, but under the hood, it follows the math above. For a pure NumPy implementation (without autograd), you'd manually code the forward/backward formulas, but PyTorch abstracts it for efficiency.


source: https://grok.com/chat/aeb2b344-5ad6-4b03-8ce1-a7afd6fbf018

-->

