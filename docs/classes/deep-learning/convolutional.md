Uses filters (kernels) to scan input data, detecting local patterns like edges or textures. Key to "convolutional neural networks" (CNNs). Image and video processing, computer vision (e.g., object detection in photos). Slides filters over the input, computing dot products to create feature maps. Reduces spatial dimensions while preserving important features.

![](https://developer-blogs.nvidia.com/wp-content/uploads/2015/11/convolution.png)
/// caption
Convolution of an image with an edge detector convolution kernel. Sources: [Deep Learning in a Nutshell: Core Concepts](https://developer.nvidia.com/blog/deep-learning-nutshell-core-concepts/#convolutional_deep_learning){target='_blank'}
///

![](https://developer.nvidia.com/blog/wp-content/uploads/2015/11/Convolution_schematic.gif)
/// caption
Calculating convolution by sliding image patches over the entire image. One image patch (yellow) of the original image (green) is multiplied by the kernel (red numbers in the yellow patch), and its sum is written to one feature map pixel (red cell in convolved feature). Image source: [Deep Learning in a Nutshell: Core Concepts](https://developer.nvidia.com/blog/deep-learning-nutshell-core-concepts/#convolutional_deep_learning){target='_blank'}
///

**Parameters**:

<div class="grid cards" markdown>
-   X: Input matrix (e.g., image).

    K: Convolution kernel (filter).

    b: Bias term.

-   (2D, stride=1, no padding):

    \( X = \begin{bmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \\ 7 & 8 & 9 \end{bmatrix} \)
    
    \( K = \begin{bmatrix} 1 & 0 \\ -1 & 1 \end{bmatrix} \)
    
    \( b=1 \)

</div>

**Forward Pass**:

<div class="grid cards" markdown>

-   Convolution:

    \( \displaystyle Y[i,j] = \sum_{m,n} X[i+m, j+n] \cdot K[m,n] + b \).    

-   Convolution:

    \( \begin{bmatrix}
        \begin{array}{ll}
        =& 1 \times 1 + 2 \times 0 \\
        &+ 4 \times (-1) + 5 \times 1 \\
        &+ 1 \end{array} &
        \begin{array}{ll}
        =& 2 \times 1 + 3 \times 0 \\
        &+ 5 \times (-1) + 6 \times 1 \\
        &+ 1 \end{array} \\
        \begin{array}{ll}
        =& 4 \times 1 + 5 \times 0 \\
        &+ 7 \times (-1) + 8 \times 1 \\
        &+ 1 \end{array} &
        \begin{array}{ll}
        =& 5 \times 1 + 6 \times 0 \\
        &+ 8 \times (-1) + 9 \times 1 \\
        &+ 1 \end{array}
    \end{bmatrix} \)

    \( Y = \begin{bmatrix} 3 & 3 \\ -1 & -1 \end{bmatrix} \)

</div>

**Backward Pass**:

<div class="grid cards" markdown>

-   1. Gradient w.r.t. input:

        Convolve upstream gradient \( \displaystyle \frac{\partial L}{\partial Y} \) with rotated kernel (full convolution).

    1. Gradient w.r.t. kernel:

        Convolve input \( X \) with \( \displaystyle \frac{\partial L}{\partial Y} \).

    1. Gradient w.r.t. bias:
    
        Sum of \( \displaystyle \frac{\partial L}{\partial Y} \).

-   \( \displaystyle \frac{\partial L}{\partial Y} = \begin{bmatrix} 0.5 & -0.5 \\ 1 & 0 \end{bmatrix} \).

    1. \( \displaystyle \frac{\partial L}{\partial X} \):
    
        Full conv with rotated K (\( \begin{bmatrix} 1 & -1 \\ 0 & 1 \end{bmatrix} \)) and padded \( dY \), approx. \( \begin{bmatrix} 0.5 & -0.5 & -0.5 \\ 0 & 1.5 & 0 \\ 1 & 0 & 0 \end{bmatrix} \) (simplified calc).

    1. \( \displaystyle \frac{\partial L}{\partial K} = \) Conv X with dY:
    
        \( \begin{bmatrix} 0.5*1 + (-0.5)*2 + 1*4 + 0*5 \\ \ldots \end{bmatrix} \)
        
        (detailed in code).

    1. \( \displaystyle \frac{\partial L}{\partial b} = 0.5 -0.5 +1 +0 = 1 \).

</div>

**Implementation**:

```pyodide install="pandas,scikit-learn" exec="on" html="1"
--8<-- "docs/classes/deep-learning/convolutional.py"
```