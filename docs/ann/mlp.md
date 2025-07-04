

<center>
``` mermaid
flowchart LR
    classDef default fill:#fff,stroke:#333,stroke-width:1px;
    subgraph input
        x1(["x<sub>1</sub>"])
        x2(["x<sub>2</sub>"])
        x3(["x<sub>3</sub>"])
    end
    subgraph hidden
        h1(["h<sub>1</sub>"])
        h2(["h<sub>2</sub>"])
    end
    subgraph output
        y1(["y<sub>1</sub>"])
    end
    in1@{ shape: circle, label: " " } --> x1
    in2@{ shape: circle, label: " " } --> x2
    in3@{ shape: circle, label: " " } --> x3
    x1 -->|"w<sub>11</sub>"|h1
    x1 -->|"w<sub>12</sub>"|h2
    x2 -->|"w<sub>21</sub>"|h1
    x2 -->|"w<sub>22</sub>"|h2
    x3 -->|"w<sub>31</sub>"|h1
    x3 -->|"w<sub>32</sub>"|h2
    h1 -->|"v<sub>11</sub>"|y1
    h2 -->|"v<sub>21</sub>"|y1
    y1 --> out1@{ shape: dbl-circ, label: " " }
    style input fill:#fff,stroke:#666,stroke-width:1px
    style hidden fill:#fff,stroke:#666,stroke-width:1px
    style output fill:#fff,stroke:#666,stroke-width:1px
```
<i>Multi-Layer Perceptron (MLP) Architecture.</i>
</center>

- Limitations of Perceptrons, such as their inability to solve non-linearly separable problems.
- Introduction to Multi-Layer Perceptrons (MLPs) as an extension of the Perceptron model.
- Structure of MLPs, including input, hidden, and output layers.
- Activation functions used in MLPs, such as sigmoid, tanh, and ReLU.
- The concept of feedforward and backpropagation in MLPs.
- The role of weights and biases in MLPs and how they are adjusted during training.
- The training process of MLPs, including the use of gradient descent and backpropagation.
- The importance of loss functions in MLP training, such as mean squared error and cross-entropy.

## Regularization techniques to prevent overfitting in MLPs, such as dropout and L2 regularization. https://grok.com/chat/0e1af7da-0d92-4603-84a6-99f2aa1b8686


### Dropout
What it is: Dropout is a regularization technique where, during training, a random subset of neurons (or their connections) is "dropped" (set to zero) in each forward and backward pass. This prevents the network from relying too heavily on specific neurons.
How it works:

During training, each neuron has a probability $ p $ (typically 0.2 to 0.5) of being dropped.
This forces the network to learn redundant representations, making it more robust and less likely to memorize the training data.
At test time, all neurons are active, but their weights are scaled by $ 1-p $ to account for the reduced activation during training.

Why it prevents overfitting:

Dropout acts like training an ensemble of smaller subnetworks, reducing co-dependency between neurons.
It introduces noise, making the model less sensitive to specific patterns in the training data.

Practical tips:

Common dropout rates: 20–50% for hidden layers, lower (10–20%) for input layers.
Use in deep networks, especially in fully connected layers or convolutional neural networks (CNNs).
Avoid dropout in the output layer or when the network is small (it may hurt performance).


### L2 Regularization (Weight Decay)
What it is: L2 regularization adds a penalty term to the loss function based on the magnitude of the model’s weights, discouraging large weights that can lead to complex, overfitted models.
How it works:

The loss function is modified to include an L2 penalty:
$$\text{Loss} = \text{Original Loss} + \lambda \sum w_i^2$$
where $ w_i $ are the model’s weights, and $ \lambda $ (regularization strength) controls the penalty’s impact.
During optimization, this penalty encourages smaller weights, simplifying the model.

Why it prevents overfitting:

Large weights amplify small input changes, leading to overfitting. L2 regularization constrains weights, making the model smoother and less sensitive to noise.
It effectively reduces the model’s capacity to memorize training data.

Practical tips:

Common $ \lambda $: $ 10^{-5} $ to $ 10^{-2} $, tuned via cross-validation.
Works well in linear models, fully connected NNs, and CNNs.
Combine with other techniques (e.g., dropout) for better results.



- Optimization algorithms used in MLP training, such as stochastic gradient descent (SGD), Adam, and RMSprop.
- Evaluation metrics for MLP performance, such as accuracy, precision, recall, and F1 score.
- Common challenges in training MLPs, such as overfitting, underfitting, the vanishing gradient problem and the need for large datasets.
- Real-world applications of MLPs in various fields, including computer vision, natural language processing, and time series forecasting.
- Applications of MLPs in various domains, including image recognition, natural language processing, and time series prediction.


- Backpropagation: ./ann/backpropagation.md
- Regularization: ./ann/regularization.md
- Optimization: ./ann/optimization.md
- Comparison of MLPs with other neural network architectures, such as convolutional neural networks (CNNs) and recurrent neural networks (RNNs).

<iframe width="100%" height="470" src="https://www.youtube.com/embed/aircAruvnKk" allowfullscreen></iframe>


### Training and Optimization

Algorithms for training ANNs involve adjusting the weights of the connections between neurons to minimize a loss function, which quantifies the difference between the predicted output and the true output. The most common optimization algorithm used in training ANNs is stochastic gradient descent (SGD), which iteratively updates the weights based on the gradient of the loss function with respect to the weights.


## Additional Resources

- [TensorFlow Playground](https://playground.tensorflow.org/){target="_blank"} is an interactive platform that allows users to visualize and experiment with neural networks. It provides a user-friendly interface to create, train, and test simple neural networks, making it an excellent tool for understanding the concepts of neural networks and their behavior. Users can adjust parameters such as the number of layers, activation functions, and learning rates to see how these changes affect the network's performance on various datasets.


[^1]: Haykin, S. (1994). Neural Networks: A Comprehensive Foundation. Prentice Hall.
[:fontawesome-brands-amazon:](https://www.amazon.com/Neural-Networks-Comprehensive-Foundation-2nd/dp/0132733501){target="_blank"}

[^2]: Bishop, C. M. (2006). Pattern Recognition and Machine Learning. Springer.
[:fontawesome-brands-amazon:](https://www.amazon.com/Pattern-Recognition-Learning-Information-Statistics/dp/0387310738){target="_blank"}
[:octicons-download-24:](https://www.microsoft.com/en-us/research/wp-content/uploads/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf){target="_blank"}

[^3]: Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep Learning. MIT Press.
[:fontawesome-brands-amazon:](https://www.amazon.com/Deep-Learning-Adaptive-Computation-Machine/dp/0262035618){target="_blank"}
[:octicons-download-24:](https://www.deeplearningbook.org/){target="_blank"}

