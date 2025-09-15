Every neuron in this layer is connected to every neuron in the previous layer. It's the most basic type. General-purpose networks, like simple classifiers or regressors. Often used in the final stages of more complex models.

![](https://docscontent.nvidia.com/dita/00000189-949d-d46e-abe9-bcdf9f8c0000/deeplearning/performance/dl-performance-fully-connected/graphics/fc-layer.svg){width="100%"}
/// caption
A sample of a small fully-connected layer with four input and eight output neurons. Source: [Linear/Fully-Connected Layers User's Guide](https://docs.nvidia.com/deeplearning/performance/dl-performance-fully-connected/index.html){target='_blank'}
///

**Parameters**:

<div class="grid cards" markdown>
-   x: Input vector.
    
    W: Weight matrix.
    
    b: Bias vector.

-   \( x = [2, 3] \)

    \( W = \begin{bmatrix} 1 & 2 \\ 0 & -1 \end{bmatrix} \)
    
    \( b = [1, -1] \)

</div>

**Forward Pass**:

<div class="grid cards" markdown>

-   \( y = Wx + b \),
    
    then apply activation (e.g., ReLU: \( y = \max(0, y) \)).

-   \( y = [9, -4] \),
    
    ReLU: [9, 0].

</div>

**Backward Pass**:

<div class="grid cards" markdown>

-   1. Gradient w.r.t. input:
    
        \( \displaystyle \frac{\partial L}{\partial x} = W^T \cdot \frac{\partial L}{\partial y'} \)
        
        where \( y' \) is post-activation,
        
        and \( \displaystyle \frac{\partial L}{\partial y'} \) is adjusted for activation,
        
        e.g., for ReLU: 1 if \( y > 0 \), else 0).

    1. Gradient w.r.t. weights:
    
        \( \displaystyle  \frac{\partial L}{\partial W} = \frac{\partial L}{\partial y'} \cdot x^T \).

    1. Gradient w.r.t. bias:
    
        \( \displaystyle \frac{\partial L}{\partial b} = \sum \frac{\partial L}{\partial y'} \).

-   Assume loss gradient \( \frac{\partial L}{\partial y'} = [0.5, -0.2] \) (post-ReLU). For ReLU: mask = [1, 0], so \( \frac{\partial L}{\partial y} = [0.5, 0] \).

    1. \( \begin{align*}
        \displaystyle \frac{\partial L}{\partial x} &= W^T \cdot [0.5, 0] \\
        & = \begin{bmatrix} 1 & 0 \\ 2 & -1 \end{bmatrix} \begin{bmatrix} 0.5 \\ 0 \end{bmatrix} \\
        & = [0.5, 1.0]
        \end{align*} \).

    1. \( \begin{align*}
        \displaystyle \frac{\partial L}{\partial W} &= [0.5, 0]^T \cdot [2, 3] \\
        & = \begin{bmatrix} 1 & 1.5 \\ 0 & 0 \end{bmatrix}
        \end{align*} \).

    1. \( \displaystyle \frac{\partial L}{\partial b} = [0.5, 0] \).

</div>

**Implementation**:

```pyodide install="pandas,scikit-learn" exec="on" html="1"
--8<-- "docs/classes/deep-learning/dense.py"
```