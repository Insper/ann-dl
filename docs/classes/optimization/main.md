
Gradient Descent is an optimization algorithm used to minimize the loss function in machine learning models. It works by iteratively adjusting the model parameters in the direction of the steepest descent of the loss function, as defined by the negative gradient.

The main idea behind gradient descent is to update the model parameters in the opposite direction of the gradient of the loss function with respect to the parameters. This is done using the following update rule:

\[
\theta = \theta - \eta \nabla J(\theta)
\]

where:

- \(\theta\) represents the model parameters,
- \(\eta\) is the learning rate (a hyperparameter that controls the step size),
- \(\nabla J(\theta)\) is the gradient of the loss function with respect to the parameters.

In the standard Supervised Learning paradigm, the loss (per sample) is simply the output of the cost function. Machine Learning is mostly about optimizing functions (usually minimizing them). It could also involve finding Nash Equilibria between two functions like with GANs. This is done using Gradient Based Methods, though not necessarily Gradient Descent[^1].

A **Gradient Based Method** is a method/algorithm that finds the minima of a function, assuming that one can easily compute the gradient of that function. It assumes that the function is continuous and differentiable almost everywhere (it need not be differentiable everywhere)[^1].

**Gradient Descent Intuition** - Imagine being in a mountain in the middle of a foggy night. Since you want to go down to the village and have only limited vision, you look around your immediate vicinity to find the direction of steepest descent and take a step in that direction[^1].

To visualize the gradient descent process, we can create a simple 3D plot that shows how the parameters of a model are updated over iterations to minimize a loss function. Below is an example code using Python with Matplotlib to create such a visualization.


``` python exec="on" html="on"
--8<-- "docs/classes/optimization/gradient-descent-path.py"
```

## Gradient Descent Variants

There are several variants of gradient descent, each with its own characteristics:

1. **Batch Gradient Descent**: Computes the gradient using the entire dataset. It provides a stable estimate of the gradient but can be slow for large datasets. Formula is given by:

    ``` python
    for epoch in range(num_epochs):
        gradients = compute_gradients(X, y, model)
        model.parameters -= learning_rate * gradients
    ```

2. **Stochastic Gradient Descent (SGD)**: Computes the gradient using a single data point at a time. It introduces noise into the optimization process, which can help escape local minima but may lead to oscillations.  Formula is given by:

    ``` python
    for epoch in range(num_epochs):
        for i in range(len(X)):
            gradients = compute_gradients(X[i], y[i], model)
            model.parameters -= learning_rate * gradients
    ```

3. **Mini-Batch Gradient Descent**: Combines the advantages of batch and stochastic gradient descent by using a small random subset of the data (mini-batch) to compute the gradient. It balances the stability of batch gradient descent with the speed of SGD.

    ``` python
    for epoch in range(num_epochs):
        for batch in get_mini_batches(X, y, batch_size):
            gradients = compute_gradients(batch.X, batch.y, model)
            model.parameters -= learning_rate * gradients
    ```


## Momentum

Momentum is a variant of gradient descent that helps accelerate the optimization process by using the past gradients to smooth out the updates. It introduces a "momentum" term that accumulates the past gradients and adds it to the current gradient update. The update rule with momentum is given by:

In Momentum, we have two iterates ($p$ and $\theta$) instead of just one. The updates are as follows:

$$
p_{k+1} = \beta p_k + (1 - \beta) \nabla f_i(\theta_k)
$$

$$
\theta_{k+1} = \theta_k - \eta p_{k+1}
$$

$p$ is called the SGD momentum. At each update step we add the stochastic gradient to the old value of the momentum, after dampening it by a factor $\beta$ (value between 0 and 1). $p$ can be thought of as a running average of the gradients. Finally we move $\theta$ in the direction of the new momentum $p$ [^1].

Alternate Form: Stochastic Heavy Ball Method

\[
\theta_{k+1} = \theta_k - \eta \nabla f_i(\theta_k) + \beta( \theta_k - \theta_{k-1} ) \quad 0 \leq \beta < 1
\]

This form is mathematically equivalent to the previous form. Here, the next step is a combination of previous step’s direction and the new negative gradient.

The momentum term helps to dampen oscillations and can lead to faster convergence, especially in scenarios with noisy gradients or ravines in the loss landscape.

## ADAM Optimizer

ADAM (Adaptive Moment Estimation) is an advanced optimization algorithm that combines the benefits of both AdaGrad and RMSProp. It maintains two moving averages for each parameter: the first moment (mean) and the second moment (uncentered variance). The update rules are as follows:

1. Compute the gradients:

\[
g_t = \nabla J(\theta_t)
\]

2. Update the first moment estimate:

\[
m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t
\]

3. Update the second moment estimate:

\[
v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2
\]

4. Compute bias-corrected estimates:

\[
\hat{m}_t = \frac{m_t}{1 - \beta_1^t}
\]

\[
\hat{v}_t = \frac{v_t}{1 - \beta_2^t}
\]

5. Update the parameters:

\[
\theta_{t+1} = \theta_t - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
\]

Where:

- \(m_t\) is the first moment (the mean of the gradients),
- \(v_t\) is the second moment (the uncentered variance of the gradients),
- \(\beta_1\) and \(\beta_2\) are hyperparameters that control the decay rates of the moving averages,
- \(\epsilon\) is a small constant added for numerical stability.

ADAM is widely used in practice due to its adaptive learning rate properties and is particularly effective for training deep learning models.



---8<-- "docs/classes/optimization/comparison.md"


[^1]: [Introduction to Gradient Descent and Backpropagation Algorithm](https://atcold.github.io/NYU-DLSP20/en/week02/02-1/){:target="_blank"}, LeCun, Y.

[^2]: [ADAM: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980){:target="_blank"}, Kingma, D. P., & Ba, J.

[^3]: [Dive into Deep Learning](https://d2l.ai){:target="_blank"}, Zhang, A., & Lipton, Z. C.
