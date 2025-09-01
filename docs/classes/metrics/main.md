## Considerations for Neural Networks

- **Classification**: Metrics like log loss and AUC-ROC are particularly relevant for neural networks, as they align with probabilistic outputs (e.g., softmax) and gradient-based optimization. For imbalanced datasets, F1-score or AUC-PR are preferred over accuracy.
- **Regression**: MSE and RMSE are commonly used as loss functions in neural networks, but MAE or Huber loss may be chosen for robustness to outliers. R² is useful for post-training evaluation but not typically as a training objective.
- **Domain-Specific Nuances**: In multi-class or multi-label classification (e.g., in CNNs for image tasks), metrics like macro/micro-averaged F1-scores are used. For time-series regression with RNNs, metrics like RMSE or MAPE are adapted to temporal dependencies.
