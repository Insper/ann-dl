
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






TODO: improve Perceptron description, add more math, and explain the learning rule. Herbian learning rule, etc.

The Perceptron learning rule (Hebbian learning rule[^4]) can be expressed mathematically as follows:

$$
w_i(t+1) = w_i(t) + \eta (y - \hat{y}) x_i
$$

where:

- \(w_i(t)\) is the weight of the \(i\)-th input at time \(t\),
- \(\eta\) is the learning rate,
- \(y\) is the true label,
- \(\hat{y}\) is the predicted output,
- \(x_i\) is the \(i\)-th input feature.

This equation updates the weights based on the difference between the true label and the predicted output, scaled by the learning rate and the input feature. The learning rate \(\eta\) controls how much the weights are adjusted during each iteration, balancing the speed of learning and stability of convergence.

This simple model can operate as a linear classifier, but it is limited to linearly separable data. 

Minsky and Papert's work in the 1960s highlighted the limitations of the Perceptron, particularly its inability to solve problems like the XOR problem, which are not linearly separable. This led to a temporary decline in interest in neural networks, often referred to as the "AI winter." However, the development of multi-layer networks and backpropagation in the 1980s revived interest in ANNs, leading to the powerful deep learning models we see today.

TODO: draw the XOR problem, explain it, and how the Perceptron cannot solve it.
```python exec="on" html="1"
--8<-- "docs/ann/xor-problem.py"
```


- Activation Functions: ./ann/activation-functions.md

The input domain of ANNs is typically represented as a vector of features, where each feature corresponds to a specific aspect of the input data. The output domain can vary depending on the task, such as classification (discrete labels) or regression (continuous values). The architecture of an ANN consists of layers of neurons, where each layer transforms the input data through weighted connections and activation functions. The connections between neurons are represented by weights, which are adjusted during the training process to minimize the error in predictions.


### Training and Optimization

Algorithms for training ANNs involve adjusting the weights of the connections between neurons to minimize a loss function, which quantifies the difference between the predicted output and the true output. The most common optimization algorithm used in training ANNs is stochastic gradient descent (SGD), which iteratively updates the weights based on the gradient of the loss function with respect to the weights.
