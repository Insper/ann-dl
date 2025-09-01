!!! success inline end "Deadline and Submission"

    :date: 19.sep (friday)
    
    :clock1: Commits until 23:59

    :material-account: Individual

    :simple-github: Submission the GitHub Pages' Link (yes, **only** the link for pages) via [insper.blackboard.com](https://insper.blackboard.com){:target="_blank"}.

**Activity: Understanding Multi-Layer Perceptrons (MLPs)**

This activity is designed to test your skills in Multi-Layer Perceptrons (MLPs).

!!! failure "Usage of Toolboxes"

    You may use toolboxes (e.g., NumPy) ==ONLY for matrix operations and calculations== during this activity. All other computations, including activation functions, loss calculations, gradients, and the forward pass, ==**MUST BE IMPLEMENTED** within your MLP== (Multi-Layer Perceptron) code. The use of ==third-party libraries for the MLP implementation **IS STRICTLY PROHIBITED**==.

    **Failure to comply with these instructions will result in your submission being rejected.**

***

## Exercise 1: Manual Calculation of MLP Steps

Consider a simple MLP with 2 input features, 1 hidden layer containing 2 neurons, and 1 output neuron. Use the hyperbolic tangent (tanh) function as the activation for both the hidden layer and the output layer. The loss function is mean squared error (MSE): \( L = \frac{1}{N} (y - \hat{y})^2 \), where \( \hat{y} \) is the network's output.

For this exercise, use the following specific values:

- Input and output vectors:

    \( \mathbf{x} = [0.5, -0.2] \)

    \( y = 1.0 \)

- Hidden layer weights:

    \( \mathbf{W}^{(1)} = \begin{bmatrix} 0.3 & -0.1 \\ 0.2 & 0.4 \end{bmatrix} \)

- Hidden layer biases:

    \( \mathbf{b}^{(1)} = [0.1, -0.2] \)

- Output layer weights:

    \( \mathbf{W}^{(2)} = [0.5, -0.3] \)

- Output layer bias:

    \( b^{(2)} = 0.2 \)

- Learning rate: \( \eta = 0.3 \)

- Activation function: \( \tanh \)

Perform the following steps explicitly, showing all mathematical derivations and calculations with the provided values:

1. **Forward Pass**:

    - Compute the hidden layer pre-activations: \( \mathbf{z}^{(1)} = \mathbf{W}^{(1)} \mathbf{x} + \mathbf{b}^{(1)} \).
    - Apply tanh to get hidden activations: \( \mathbf{a}^{(1)} = \tanh(\mathbf{z}^{(1)}) \).
    - Compute the output pre-activation: \( z^{(2)} = \mathbf{W}^{(2)} \mathbf{a}^{(1)} + b^{(2)} \).
    - Compute the final output: \( \hat{y} = \tanh(z^{(2)}) \).

2. **Loss Calculation**:

    - Compute the MSE loss:

        \( L = \frac{1}{N} (y - \hat{y})^2 \).

3. **Backward Pass (Backpropagation)**: Compute the gradients of the loss with respect to all weights and biases. Start with \( \frac{\partial L}{\partial \hat{y}} \), then compute:

    - \( \frac{\partial L}{\partial z^{(2)}} \) (using the tanh derivative: \( \frac{d}{dz} \tanh(z) = 1 - \tanh^2(z) \)).
    - Gradients for output layer: \( \frac{\partial L}{\partial \mathbf{W}^{(2)}} \), \( \frac{\partial L}{\partial b^{(2)}} \).
    - Propagate to hidden layer: \( \frac{\partial L}{\partial \mathbf{a}^{(1)}} \), \( \frac{\partial L}{\partial \mathbf{z}^{(1)}} \).
    - Gradients for hidden layer: \( \frac{\partial L}{\partial \mathbf{W}^{(1)}} \), \( \frac{\partial L}{\partial \mathbf{b}^{(1)}} \).
    
    Show all intermediate steps and calculations.

4. **Parameter Update**: Using the learning rate \( \eta = 0.1 \), update all weights and biases via gradient descent:

    - \( \mathbf{W}^{(2)} \leftarrow \mathbf{W}^{(2)} - \eta \frac{\partial L}{\partial \mathbf{W}^{(2)}} \)
    - \( b^{(2)} \leftarrow b^{(2)} - \eta \frac{\partial L}{\partial b^{(2)}} \)
    - \( \mathbf{W}^{(1)} \leftarrow \mathbf{W}^{(1)} - \eta \frac{\partial L}{\partial \mathbf{W}^{(1)}} \)
    - \( \mathbf{b}^{(1)} \leftarrow \mathbf{b}^{(1)} - \eta \frac{\partial L}{\partial \mathbf{b}^{(1)}} \)

    Provide the numerical values for all updated parameters.

**Submission Requirements**: Show all mathematical steps explicitly, including intermediate calculations (e.g., matrix multiplications, tanh applications, gradient derivations). Use exact numerical values throughout and avoid rounding excessively to maintain precision (at least 4 decimal places).

***

## Exercise 2: Binary Classification with Synthetic Data and Scratch MLP

Using the `make_classification` function from scikit-learn ([documentation](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_classification.html){target='_blank'}), generate a synthetic dataset with the following specifications:

- Number of samples: 1000
- Number of classes: 2
- Number of clusters per class: Use the `n_clusters_per_class` parameter creatively to achieve 1 cluster for one class and 2 for the other (hint: you may need to generate subsets separately and combine them, as the function applies the same number of clusters to all classes by default).
- Other parameters: Set `n_features=2` for easy visualization, `n_informative=2`, `n_redundant=0`, `random_state=42` for reproducibility, and adjust `class_sep` or `flip_y` as needed for a challenging but separable dataset.

Implement an MLP from scratch (without using libraries like TensorFlow or PyTorch for the model itself; you may use NumPy for array operations) to classify this data. You have full freedom to choose the architecture, including:

- Number of hidden layers (at least 1)
- Number of neurons per layer
- Activation functions (e.g., sigmoid, ReLU, tanh)
- Loss function (e.g., binary cross-entropy)
- Optimizer (e.g., gradient descent, with a chosen learning rate)

Steps to follow:

1. Generate and split the data into training (80%) and testing (20%) sets.
2. Implement the forward pass, loss computation, backward pass, and parameter updates in code.
3. Train the model for a reasonable number of epochs (e.g., 100-500), tracking training loss.
4. Evaluate on the test set: Report accuracy, and optionally plot decision boundaries or confusion matrix.
5. Submit your code and results, including any visualizations.


***

## Exercise 3: Multi-Class Classification with Synthetic Data and Reusable MLP

Similar to Exercise 2, but with increased complexity.

Use `make_classification` to generate a synthetic dataset with:

- Number of samples: 1500
- Number of classes: 3
- Number of clusters per class: Achieve 2 clusters for one class, 3 for another, and 4 for the last (again, you may need to generate subsets separately and combine them, as the function doesn't directly support varying clusters per class).
- Other parameters: `n_features=2`, `n_informative=2`, `n_redundant=0`, `random_state=42`.

Implement an MLP from scratch to classify this data. You may choose the architecture freely, but for an extra point (bringing this exercise to 4 points), reuse the exact same MLP implementation code from Exercise 2, modifying only hyperparameters (e.g., output layer size for 3 classes, loss function to categorical cross-entropy if needed) without changing the core structure.

Steps:

1. Generate and split the data (80/20 train/test).
2. Train the model, tracking loss.
3. Evaluate on test set: Report accuracy, and optionally visualize (e.g., scatter plot of data with predicted labels).
4. Submit code and results.


***

## Exercise 4: Multi-Class Classification with Deeper MLP

Repeat Exercise 3 exactly, but now ensure your MLP has **at least 2 hidden layers**. You may adjust the number of neurons per layer as needed for better performance. Reuse code from Exercise 3 where possible, but the focus is on demonstrating the deeper architecture. Submit updated code, training results, and test evaluation.

***


## **Evaluation Criteria**

The deliverable for this activity consists of a **report** that includes:


**Important Notes:**

- The deliverable must be submitted in the format specified: **GitHub Pages**. **No other formats will be accepted.** - there exists a template for the course that you can use to create your GitHub Pages - [template](https://hsandmann.github.io/documentation.template/){target='_blank'};

- There is a **strict policy against plagiarism**. Any form of plagiarism will result in a zero grade for the activity and may lead to further disciplinary actions as per the university's academic integrity policies;

- **The deadline for each activity is not extended**, and it is expected that you complete them within the timeframe provided in the course schedule - **NO EXCEPTIONS** will be made for late submissions.

- **AI Collaboration is allowed**, but each student **MUST UNDERSTAND** and be able to explain all parts of the code and analysis submitted. Any use of AI tools must be properly cited in your report. **ORAL EXAMS** may require you to explain your work in detail.

- All deliverables for individual activities should be submitted through the course platform [insper.blackboard.com](http://insper.blackboard.com/){:target="_blank"}.

**Grade Criteria:**

- **Exercise 1 (2 points)**:
    - Forward pass fully explicit (0.5 points)
    - Loss and backward pass with all gradients derived (1 point)
    - Parameter updates shown correctly (0.5 point)
    - Deductions for missing steps or incorrect math.

- **Exercise 2 (3 points)**:
    - Correct data generation and splitting (0.5 points)
    - Functional MLP implementation from scratch (2 point)
    - Training, evaluation, and results reported (0.5 points)
    - Deductions for using forbidden libraries in the model core or poor performance due to errors.

- **Exercise 3 (2 points + 1 extra)**:
    - Correct data generation and splitting (0.5 points)
    - Functional MLP for multi-class (1.5 points)
    - Training, evaluation, and results (1 point)
    - Extra point: Exact reuse of Exercise 2's MLP code structure (1 point, optional)
    - Deductions similar to Exercise 2; extra point only if reuse is verbatim in core logic.

- **Exercise 4 (2 points)**:
    - Successful adaptation of Exercise 3 with at least 2 hidden layers (1 point)
    - Training and evaluation results showing functionality (1 point)
    - Deductions if architecture doesn't meet the depth requirement or if results are not provided.

**Overall**: Submissions must be clear, well-documented (code comments, explanations), and reproducible.
