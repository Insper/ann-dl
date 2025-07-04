
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
</center>

- Limitations of Perceptrons, such as their inability to solve non-linearly separable problems.
- Introduction to Multi-Layer Perceptrons (MLPs) as an extension of the Perceptron model.
- Structure of MLPs, including input, hidden, and output layers.
- Activation functions used in MLPs, such as sigmoid, tanh, and ReLU.
- The concept of feedforward and backpropagation in MLPs.
- The role of weights and biases in MLPs and how they are adjusted during training.
- The training process of MLPs, including the use of gradient descent and backpropagation.
- The importance of loss functions in MLP training, such as mean squared error and cross-entropy.
- Regularization techniques to prevent overfitting in MLPs, such as dropout and L2 regularization.
- Optimization algorithms used in MLP training, such as stochastic gradient descent (SGD), Adam, and RMSprop.
- Evaluation metrics for MLP performance, such as accuracy, precision, recall, and F1 score.
- Common challenges in training MLPs, such as overfitting, underfitting, the vanishing gradient problem and the need for large datasets.
- Real-world applications of MLPs in various fields, including computer vision, natural language processing, and time series forecasting.
- Applications of MLPs in various domains, including image recognition, natural language processing, and time series prediction.


- Backpropagation: ./ann/backpropagation.md
- Regularization: ./ann/regularization.md
- Optimization: ./ann/optimization.md
- Comparison of MLPs with other neural network architectures, such as convolutional neural networks (CNNs) and recurrent neural networks (RNNs).
