Deep learning is a subset of machine learning (which itself is part of artificial intelligence) that focuses on training artificial neural networks with multiple layers to learn and make predictions from complex data. These networks are inspired by the human brain's structure, where "neurons" process information and pass it along.

Unlike traditional machine learning algorithms (e.g., linear regression or decision trees), which often require manual feature engineering (hand-picking important data characteristics), deep learning models automatically extract features from raw data through layers of processing. This makes them powerful for tasks like image recognition, natural language processing, speech synthesis, and more.

Deep learning excels with large datasets and high computational power (e.g., GPUs), but it can be "black-box" in nature—meaning it's sometimes hard to interpret why a model makes a specific decision.

The core building block is the **artificial neural network (ANN)**, which consists of interconnected nodes (neurons) organized into layers. Data flows from the input layer, through hidden layers (where the "deep" part comes in, with many layers stacked), to the output layer. Training involves adjusting weights (connections between neurons) using algorithms like backpropagation to minimize errors.

## Key Components

A typical neural network has three main parts:

- **Input Layer**: The entry point where raw data (e.g., pixel values from an image) is fed into the network. It doesn't perform computations; it just passes data forward.
- **Hidden Layers**: The "depth" of deep learning. These are where the magic happens—multiple stacked layers that transform the data through mathematical operations. Each layer learns increasingly abstract representations (e.g., from edges in an image to full objects).
- **Output Layer**: The final layer that produces the prediction or classification (e.g., "cat" or "dog" in an image classifier).

## Different Types of Layers

Deep learning models use various specialized layers depending on the task and architecture. Here's an overview of common layer types, grouped by their typical use. The following table summarizes their characteristics:

| Layer Type              | Description | Common Use Cases | How It Works |
|-------------------------|-------------|-----------------|--------------|
| [**Dense (Fully Connected)**](#a-dense-fully-connected) | Every neuron in this layer is connected to every neuron in the previous layer. It's the most basic type. | General-purpose networks, like simple classifiers or regressors. Often used in the final stages of more complex models. | Applies a linear transformation (weights * inputs + bias) followed by an activation function (e.g., ReLU) to introduce non-linearity. |
| [**Convolutional**](#b-convolutional) | Uses filters (kernels) to scan input data, detecting local patterns like edges or textures. Key to "convolutional neural networks" (CNNs). | Image and video processing, computer vision (e.g., object detection in photos). | Slides filters over the input, computing dot products to create feature maps. Reduces spatial dimensions while preserving important features. |
| [**Pooling**](#c-pooling-max-pooling) | Downsamples the output from convolutional layers, reducing computational load and preventing overfitting. Types include max pooling (takes the maximum value) and average pooling. | Follows convolutional layers in CNNs to summarize features. | Aggregates values in small regions (e.g., 2x2 grid) into a single value, making the model more robust to variations like translations. |
| [**Recurrent**](#d-recurrent-lstm) (e.g., RNN, LSTM, GRU) | Handles sequential data by maintaining a "memory" of previous inputs via loops. LSTM (Long Short-Term Memory) and GRU (Gated Recurrent Unit) are advanced variants that address vanishing gradient issues. | Time-series forecasting, natural language processing (e.g., machine translation), speech recognition. | Processes inputs one step at a time, using hidden states to carry information forward. Good for sequences but can struggle with long dependencies. |
| [**Embedding**](#e-embedding) | Converts categorical data (e.g., words) into dense vectors of fixed size, capturing semantic relationships. | NLP tasks like word embeddings (e.g., Word2Vec). Often the first layer in text-based models. | Maps high-dimensional sparse data (e.g., one-hot encoded words) to lower-dimensional continuous space. |
| [**Attention**](#f-attention-scaled-dot-product) (used in Transformers) | Allows the model to focus on relevant parts of the input dynamically, weighing their importance. Self-attention computes relationships between all elements. | Modern NLP (e.g., GPT models), machine translation, and even vision tasks. | Uses queries, keys, and values to compute attention scores, enabling parallel processing of sequences (unlike RNNs). |
| [**Normalization**](#g-normalization-batch-normalization) (e.g., Batch Normalization, Layer Normalization) | Stabilizes training by normalizing activations within a layer, reducing internal covariate shift. | Almost all deep networks to speed up training and improve performance. | Adjusts and scales activations (e.g., mean to 0, variance to 1) across mini-batches or individual layers. |
| [**Dropout**](#h-dropout) | Randomly "drops out" (ignores) a fraction of neurons during training to prevent overfitting. | Regularization in any network, especially dense or convolutional ones. | Temporarily removes connections, forcing the network to learn redundant representations. Inactive during inference. |
| [**Flatten**](#i-flatten) | Converts multi-dimensional data (e.g., from convolutional layers) into a 1D vector for dense layers. | Transitioning from feature extraction (CNN) to classification. | Reshapes tensors without changing values, e.g., turning a 2D feature map into a flat array. |
| [**Activation**](#j-activation) | Applies a non-linear function to the output of other layers (though often built into them). Common ones: ReLU (Rectified Linear Unit), Sigmoid, Tanh, Softmax. | Everywhere, to add non-linearity and control output ranges (e.g., Softmax for probabilities). | Transforms linear outputs; e.g., ReLU sets negative values to 0 for faster training. |

## Common Deep Learning Architectures

These layers are combined into architectures tailored to specific problems:

- **Feedforward Neural Networks (FNN)**: Basic stack of dense layers for simple tasks.
- **Convolutional Neural Networks (CNN)**: Convolutional + pooling layers for spatial data like images (e.g., ResNet, VGG).
- **Recurrent Neural Networks (RNN)**: Recurrent layers for sequences (e.g., LSTM for text generation).
- **Transformers**: Attention layers for handling long-range dependencies (e.g., BERT for NLP, Vision Transformers for images).
- **Autoencoders**: Encoder (convolutional/dense) + decoder layers for unsupervised learning like denoising.
- **Generative Adversarial Networks (GANs)**: Combines generator and discriminator networks (often convolutional) for generating realistic data.

## Forward and Backward Pass for Each Layer

The forward pass computes the output of each layer given the input, while the backward pass computes gradients for learning.

Backpropagation computes the gradient of the loss with respect to the layer's inputs and parameters (e.g., weights, biases) to update them via optimizers like gradient descent. Assume a scalar loss \( L \), and upstream gradient \( \displaystyle \frac{\partial L}{\partial y} \) (where \( y \) is the layer's output) is provided from the next layer.

---

### A. Dense (Fully Connected)

--8<-- "docs/classes/deep-learning/dense.md"

---

### B. Convolutional

--8<-- "docs/classes/deep-learning/convolutional.md"

---

### C. Pooling (Max Pooling)

Downsamples the output from convolutional layers, reducing computational load and preventing overfitting. Types include max pooling (takes the maximum value) and average pooling. Follows convolutional layers in CNNs to summarize features. Aggregates values in small regions (e.g., 2x2 grid) into a single value, making the model more robust to variations like translations.

**Forward Pass**:

<div class="grid cards" markdown>

-   \( Y[i,j] = \max(X[i:i+k, j:j+k]) \) for pool size \( k \).

-   \( X = \begin{bmatrix} 1 & 2 & 3 & 4 \\ 5 & 6 & 7 & 8 \\ 9 & 10 & 11 & 12 \\ 13 & 14 & 15 & 16 \end{bmatrix} \), pool=2, stride=2,

    \( Y = \begin{bmatrix} 6 & 8 \\ 14 & 16 \end{bmatrix} \).

    - Max positions: e.g., 6 from X[1,1]=6, etc.

</div>

**Backward Pass**:

<div class="grid cards" markdown>

-   Distribute upstream gradient \( \displaystyle \frac{\partial L}{\partial Y} \) to the max position in each window; 0 elsewhere.

-   \( \displaystyle \frac{\partial L}{\partial Y} = \begin{bmatrix} 0.5 & -0.5 \\ 1 & 0 \end{bmatrix} \).

    - \( \displaystyle \frac{\partial L}{\partial X} \):
    
        - 0.5 to pos of 6 (1,1),
        - -0.5 to pos of 8 (1,3),
        - 1 to pos of 14 (3,1),
        - 0 to pos of 16 (3,3).
        - Other positions 0.

</div>

**Implementation**:

```pyodide install="pandas,scikit-learn" exec="on" html="1"
--8<-- "docs/classes/deep-learning/pooling.py"
```

---

### D. Recurrent (LSTM)

--8<-- "docs/classes/deep-learning/lstm.md"

---

### E. Embedding 

**Forward Pass**:

<div class="grid cards" markdown>

-   \( y = E[i] \),
    
    where \( E \) is the embedding matrix,
    
    \( i \) is the input index.

-   Index 1,

    \( E = \begin{bmatrix} 0.1 & 0.2 & 0.3 \\ 0.4 & 0.5 & 0.6 \end{bmatrix}\),

    \( y = [0.4,0.5,0.6] \).

</div>

**Backward Pass**:

<div class="grid cards" markdown>

-   Gradient \( \displaystyle \frac{\partial L}{\partial E[i]} += \frac{\partial L}{\partial y} \); other rows 0. (Sparse update).

-   \( \displaystyle \frac{\partial L}{\partial y} = [0.1, -0.1, 0.2] \),

    so add to E[1].

</div>

**Implementation**:

```pyodide install="pandas,scikit-learn" exec="on" html="1"
--8<-- "docs/classes/deep-learning/embedding.py"
```

---

### F. Attention (Scaled Dot-Product)

**Forward Pass**: \( \text{Attention} = \text{softmax}\left(\frac{QK^T}{\sqrt{d}}\right)V \).

**Backward Pass**: Gradients for Q, K, V via chain rule on softmax and matmuls.

**Numerical Example**: Use previous. Backward is matrix derivs; code handles.

**Implementation**:

```pyodide install="pandas,scikit-learn" exec="on" html="1"
--8<-- "docs/classes/deep-learning/attention.py"
```

---

### G. Normalization (Batch Normalization)

**Forward Pass**: Normalize, scale, shift.

**Backward Pass**: Gradients for input, gamma, beta via chain rule on mean/var.

**Numerical Example**: Previous forward. Backward computes dx, dgamma, dbeta.

**Implementation**:

```pyodide install="pandas,scikit-learn" exec="on" html="1"
--8<-- "docs/classes/deep-learning/normalization.py"
```

---

### H. Dropout

**Forward Pass**: Mask during training.

**Backward Pass**: Same mask applied to upstream gradient (scale by 1/(1-p)).

**Numerical Example**: Same as forward; backward passes dy through mask.

**Implementation**:

```pyodide install="pandas,scikit-learn" exec="on" html="1"
--8<-- "docs/classes/deep-learning/dropout.py"
```

---

### I. Flatten

**Forward Pass**: Reshape to 1D.

**Backward Pass**: Reshape upstream gradient back to original shape.

**Numerical Example**:
- Forward: 2x2 to [1,2,3,4].
- Backward: dy = [0.1,0.2,0.3,0.4] -> reshape to 2x2.

**Implementation**:

```pyodide install="pandas,scikit-learn" exec="on" html="1"
--8<-- "docs/classes/deep-learning/flatten.py"
```

### J. Activation (ReLU)

**Forward Pass**:
<div class="grid cards" markdown>
-   \( y = \max(0, x) \). 
-   \( x = [-1, 0, 2, -3] \),
    
    \( y = [0, 0, 2, 0] \).
</div>

**Backward Pass**:
<div class="grid cards" markdown>
-   \( \displaystyle \frac{\partial L}{\partial x} = \frac{\partial L}{\partial y} \cdot (x > 0) \).

-   \( dy = [0.5,-0.5,1,0] \),

    \( dx = [0,0,1,0] \) (masked).
</div>

**Implementation**:

```pyodide install="pandas,scikit-learn" exec="on" html="1"
--8<-- "docs/classes/deep-learning/activation.py"
```




[^1]: Mohd Halim Mohd Noor, Ayokunle Olalekan Ige: [A Survey on State-of-the-art Deep Learning Applications and Challenges](https://arxiv.org/pdf/2403.17561){target='_blank'}, 2025.

[^2]: Aston Zhang, Zachary C. Lipton, Mu Li, and Alexander J. Smola: [Dive into Deep Learning](https://d2l.ai/){target='_blank'}, 2020.

[^3]: Ian Goodfellow, Yoshua Bengio, Aaron Courville: [Deep Learning](https://www.deeplearningbook.org/){target='_blank'}, 2016.

[^4]: Johannes Schneider, Michalis Vlachos: [A Survey of Deep Learning: From Activations to Transformers](https://arxiv.org/abs/2302.00722){target='_blank'}, 2024.

[^5]: Goodfellow, I., Bengio, Y., & Courville, A. (2016). [Deep Learning](https://www.deeplearningbook.org/){target='_blank'}. MIT Press.

[^6]: [Adit Deshpande](https://adeshpande3.github.io/){target='_blank'} is a comprehensive resource for learning about deep learning, covering various topics such as neural networks, convolutional neural networks (CNNs), recurrent neural networks (RNNs), and more. It provides detailed explanations, code examples, and practical applications of deep learning concepts, making it suitable for both beginners and advanced learners.

[^7]: [François Fleuret](https://fleuret.org/francois/lbdl.html){target='_blank'} offers a collection of deep learning resources, including lectures, tutorials, and research papers, aimed at helping learners understand and apply deep learning techniques effectively.