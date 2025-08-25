
!!! success inline end "Deadline and Submission"

    :date: 21.sep (sunday)
    
    :clock1: Commits until 23:59

    :material-account-group: [Team (2-3 members) form](https://forms.gle/Rrb9b3dJcHTUHbsK6){:target="_blank"}

    :simple-github: Submission the GitHub Pages' Link via [insper.blackboard.com](https://insper.blackboard.com){:target="_blank"}.


In this project, you will tackle a real-world classification task using a Multi-Layer Perceptron (MLP) neural network. The goal is to deepen your understanding of neural networks by handling data preparation, model implementation, training strategies, and evaluation without relying on high-level deep learning libraries. You will select a public dataset suitable for classification, process it, build and train your MLP, and analyze the results.

!!! example "Competition Bonus"

    This project encourages creativity in dataset selection and rewards ambition—bonus points will be awarded if you submit your solution to a relevant online competition (e.g., on platforms like [Kaggle](https://www.kaggle.com){:target="_blank"}, [DrivenData](https://www.drivendata.org){:target="_blank"}, or [Zindi](https://zindi.africa){:target="_blank"}). Submissions must be documented in your report, including a link to your entry and any leaderboard position (if applicable). Bonus points:
    
    | Points | Description |
    |:--------:|-------------|
    | +0.5 | Valid submission to a recognized competition (proof required, e.g., link, screenshot). |
    | +0.5 | Valid submission ranking in the top 50% of the leaderboard (proof required). |

    If selecting from a competition platform, note the competition rules and ensure your work complies.

!!! danger "Important Constraints"

    - **DO NOT USE** the Titanic, Iris, Wine or others **classical** datasets. These are overused and will result in a zero score for the dataset selection portion.
    - The task must be classification (e.g., binary, multi-class, or multi-label).
    - You may implement the MLP yourself or use high-level libraries like TensorFlow, PyTorch, Keras, or scikit-learn's neural network modules, these are **ALLOWED**. But, you **HAVE TO** understand and explain all parts of the code and analysis submitted. You may use NumPy (or similar for matrix operations), Matplotlib/Seaborn for plotting, and Pandas/SciPy for data cleaning/normalization.
    - The dataset must have at least 1,000 samples and multiple features (at least 5) to ensure the MLP is meaningful.


## Project Steps

Follow these steps in your work. Your report must address each one explicitly.

### 1. **Dataset Selection**

- Choose a public dataset for a classification problem. Sources include:
    - Kaggle (e.g., datasets for digit recognition, spam detection, or medical diagnosis).
    - UCI Machine Learning Repository (e.g., Banknote Authentication, Adult Income, or Covertype).
    - Other open sources like OpenML, Google Dataset Search, or government data portals (e.g., data.gov).
    - Also, consider datasets from LOTS, here you have direct access to business problems.

- Ensure the dataset has at least 1,000 samples and multiple features (at least 5) to make the MLP meaningful.
- If selecting from a competition platform, note the competition rules and ensure your work complies.
- In your report: Provide the dataset name, source URL, size (rows/columns), and why you chose it (e.g., relevance to real-world problems, complexity).

### 2. **Dataset Explanation**

- Describe the dataset in detail: What does it represent? What are the features (inputs) and their types (numerical, categorical)? What is the target variable (classes/labels)?
- Discuss any domain knowledge: E.g., if it's a medical dataset, explain key terms.
- Identify potential issues: Imbalanced classes, missing values, outliers, or noise.
- In your report: Include summary statistics (e.g., mean, std dev, class distribution) and visualizations (e.g., histograms, correlation matrices).

### 3. **Data Cleaning and Normalization**

- Clean the data: Handle missing values (impute or remove), remove duplicates, detect and treat outliers.
- Preprocess: Encode categorical variables (e.g., one-hot encoding), normalize/scale numerical features (e.g., min-max scaling or z-score standardization).
- You may use libraries like Pandas for loading/cleaning and SciPy/NumPy for normalization.
- In your report: Explain each step, justify choices (e.g., "I used median imputation for missing values to avoid skew from outliers"), and show before/after examples (e.g., via tables or plots).

### 4. **MLP Implementation**

- Code an MLP from scratch using only NumPy (or equivalent) for operations like matrix multiplication, activation functions, and gradients.
- Architecture: At minimum, include an input layer, one hidden layer, and output layer. Experiment with more layers/nodes for better performance.
- Activation functions: Use sigmoid, ReLU, or tanh.
- Loss function: Cross-entropy for classification.
- Optimizer: Stochastic Gradient Descent (SGD) or a variant like mini-batch GD.
- Pre-built neural network libraries allowed, but you must understand and explain all parts of the code and analysis submitted.
- In your report: Provide code or key code snippets (the full code). Explain hyperparameters (e.g., learning rate, number of epochs, hidden units).

### 5. **Model Training**

- Train your MLP on the prepared data.
- Implement the training loop: Forward propagation, loss calculation, backpropagation, and parameter updates.
- Handle initialization (e.g., random weights) and regularization if needed (e.g., L2 penalty, but optional).
- In your report: Describe the training process, including any challenges (e.g., vanishing gradients) and how you addressed them.

### 6. **Training and Testing Strategy**

- Split the data: Use train/validation/test sets (e.g., 70/15/15 split) or k-fold cross-validation.
- Training mode: Choose batch, mini-batch, or online (stochastic) training; explain why (e.g., "Mini-batch for balance between speed and stability").
- Early stopping or other techniques to prevent overfitting.
- In your report: Detail the split ratios, random seeds for reproducibility, and rationale. Discuss validation's role in hyperparameter tuning.

### 7. **Error Curves and Visualization**

- Plot training and validation loss/accuracy curves over epochs.
- Use Matplotlib or similar for plots.
- Analyze: Discuss convergence, overfitting/underfitting, and adjustments made.
- In your report: Include at least two plots (e.g., loss vs. epochs, accuracy vs. epochs). Interpret trends (e.g., "Loss plateaus after 50 epochs, indicating convergence").

### 8. **Evaluation Metrics**

- Apply classification metrics on the test set: Accuracy, precision, recall, F1-score, confusion matrix (for multi-class).
- If imbalanced, include ROC-AUC or precision-recall curves.
- Compare to baselines (e.g., majority class predictor).
- In your report: Present results in tables (e.g., metric values) and visualizations (e.g., confusion matrix heatmap). Discuss strengths/weaknesses (e.g., "High recall on class A but low on B due to imbalance").

***

## **Evaluation Criteria**

The deliverable for this activity consists of a comprehensive **report** that includes:

- **Sections**: One for each step above (1-8).
- **Conclusion**: Overall findings, limitations (e.g., MLP vs. more advanced models), future improvements.
- **References**: Cite dataset sources, any papers on MLPs, etc.

**Important Notes:**

- The deliverable must be submitted in the format specified: **GitHub Pages**. **No other formats will be accepted.** - there exists a template for the course that you can use to create your GitHub Pages - [template](https://hsandmann.github.io/documentation.template/){target='_blank'};

- There is a **strict policy against plagiarism**. Any form of plagiarism will result in a zero grade for the activity and may lead to further disciplinary actions as per the university's academic integrity policies;

- **The deadline for each activity is not extended**, and it is expected that you complete them within the timeframe provided in the course schedule - **NO EXCEPTIONS** will be made for late submissions.

- **AI Collaboration is allowed**, but each student **MUST UNDERSTAND** and be able to explain all parts of the code and analysis submitted. Any use of AI tools must be properly cited in your report. ^^**ORAL EXAMS**^^ may require you to explain your work in detail.

- All deliverables for individual activities should be submitted through the course platform [insper.blackboard.com](http://insper.blackboard.com/){:target="_blank"}.


**Grading Rubric** (out of 10 points):

| Criteria | Description |
|:--------:|-------------|
| **2 pts** | Dataset Selection and Explanation: 1 point<br>Data Cleaning/Normalization: 1 point |
| **6 pts** | MLP Implementation: 2 points (correctness and originality);<br>Training and Strategy: 1.5 points;<br>Error Curves: 1 point;<br>Metrics and Analysis: 1.5 points |
| **2 pts** | Report Quality (clarity, structure, visuals): 1 point;<br>Bonus: Up to +1 for competition submission (as described). |

This project will test your end-to-end machine learning skills. If you have questions, ask during office hours.
