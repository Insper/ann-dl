Below is a detailed list of metrics commonly used to evaluate the accuracy and performance of classification and regression models in machine learning, including neural networks. The metrics are categorized based on their applicability to classification or regression tasks, with explanations of their purpose and mathematical formulations where relevant.

## Classification Metrics

Classification tasks involve predicting discrete class labels. The following metrics assess the accuracy and effectiveness of such models:

1. **Accuracy**
   - **Purpose**: Measures the proportion of correct predictions across all classes.
   - **Formula**: \( \text{Accuracy} = \frac{\text{Number of Correct Predictions}}{\text{Total Number of Predictions}} = \frac{TP + TN}{TP + TN + FP + FN} \)
     - \( TP \): True Positives, \( TN \): True Negatives, \( FP \): False Positives, \( FN \): False Negatives.
   - **Use Case**: Suitable for balanced datasets but misleading for imbalanced ones.

2. **Precision**
   - **Purpose**: Evaluates the proportion of positive predictions that are actually correct.
   - **Formula**: \( \text{Precision} = \frac{TP}{TP + FP} \)
   - **Use Case**: Important when false positives are costly (e.g., spam detection).

3. **Recall (Sensitivity or True Positive Rate)**
   - **Purpose**: Measures the proportion of actual positives correctly identified.
   - **Formula**: \( \text{Recall} = \frac{TP}{TP + FN} \)
   - **Use Case**: Critical when false negatives are costly (e.g., disease detection).

4. **F1-Score**
   - **Purpose**: Harmonic mean of precision and recall, balancing both metrics.
   - **Formula**: \( \text{F1-Score} = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}} \)
   - **Use Case**: Useful for imbalanced datasets where both precision and recall matter.

5. **Area Under the ROC Curve (AUC-ROC)**
   - **Purpose**: Measures the model’s ability to distinguish between classes across all thresholds.
   - **Formula**: Area under the curve plotting True Positive Rate (Recall) vs. False Positive Rate (\( \frac{FP}{FP + TN} \)).
   - **Use Case**: Effective for binary classification and assessing model robustness.

6. **Area Under the Precision-Recall Curve (AUC-PR)**
   - **Purpose**: Focuses on precision and recall trade-off, especially for imbalanced datasets.
   - **Formula**: Area under the curve plotting Precision vs. Recall.
   - **Use Case**: Preferred when positive class is rare (e.g., fraud detection).

7. **Confusion Matrix**
   - **Purpose**: Provides a tabular summary of prediction outcomes (TP, TN, FP, FN).
   - **Use Case**: Offers detailed insights into class-specific performance, especially for multi-class problems.

8. **Log Loss (Logarithmic Loss or Cross-Entropy Loss)**
   - **Purpose**: Penalizes incorrect predictions based on predicted probabilities.
   - **Formula**: \( \text{Log Loss} = -\frac{1}{N} \sum_{i=1}^N [y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i)] \)
     - \( y_i \): True label, \( \hat{y}_i \): Predicted probability.
   - **Use Case**: Common in probabilistic classifiers like neural networks with softmax outputs.

9. **Matthews Correlation Coefficient (MCC)**
   - **Purpose**: Balances all four confusion matrix quadrants, robust for imbalanced data.
   - **Formula**: \( \text{MCC} = \frac{TP \cdot TN - FP \cdot FN}{\sqrt{(TP + FP)(TP + FN)(TN + FP)(TN + FN)}} \)
   - **Use Case**: Preferred for a single, comprehensive metric in binary classification.

10. **Cohen’s Kappa**
    - **Purpose**: Measures agreement between predicted and true labels, adjusted for chance.
    - **Formula**: \( \kappa = \frac{p_o - p_e}{1 - p_e} \)
      - \( p_o \): Observed agreement, \( p_e \): Expected agreement by chance.
    - **Use Case**: Useful for multi-class problems or when chance agreement is a concern.
