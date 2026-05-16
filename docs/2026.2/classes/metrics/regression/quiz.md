<div id="quiz-metrics-reg"></div>
<script>
buildQuiz('metrics-reg', 'Regression Metrics', [
  {
    q: 'What is the difference between MAE and MSE?',
    opts: [
      'MAE uses means; MSE uses medians',
      'MAE averages absolute errors; MSE averages squared errors — penalizing large errors more strongly',
      'MSE is always larger than MAE',
      'MAE is for classification; MSE is for regression'
    ],
    ans: 1,
    exp: 'MAE = mean(|y - ŷ|): robust to outliers. MSE = mean((y-ŷ)²): amplifies large errors. RMSE = √MSE has the same unit as the target, making it more interpretable than MSE.'
  },
  {
    q: 'What is the R² (coefficient of determination)?',
    opts: [
      'The Pearson correlation between predictions and labels',
      'The proportion of target variance explained by the model (R²=1: perfect, R²=0: equal to the mean)',
      'The root mean squared error',
      'The L2 regularization coefficient'
    ],
    ans: 1,
    exp: 'R² = 1 - SS_res/SS_tot. SS_res is residual variance (model errors); SS_tot is total data variance. R²=0.85 means the model explains 85% of the variability in the target.'
  },
  {
    q: 'When is RMSE preferable to MAE?',
    opts: [
      'When the dataset contains many outliers',
      'When large errors are disproportionately worse — RMSE penalizes extreme deviations more',
      'When we want a metric in the same scale as the target without extra penalization',
      'RMSE is always preferable to MAE'
    ],
    ans: 1,
    exp: 'RMSE penalizes large errors more than MAE due to squaring. Use RMSE when an error of 10 is much worse than ten errors of 1 (e.g., peak demand forecasting). Use MAE when all errors have equal cost.'
  },
  {
    q: 'A regression model has R² = -0.3 on the test set. What does this mean?',
    opts: [
      'The model has 30% accuracy',
      'The model performs WORSE than simply predicting the mean of the data',
      'The model has 70% error',
      'The model is experiencing slight underfitting'
    ],
    ans: 1,
    exp: 'Negative R² means SS_res > SS_tot — the model makes more error than simply predicting the mean. Can indicate inverted data leakage, irrelevant features, or a bug in the implementation.'
  },
  {
    q: 'Which metric is most appropriate for predicting house prices, where the magnitude of error matters?',
    opts: ['R²', 'RMSE (in dollars)', 'Log-Loss', 'F1-Score'],
    ans: 1,
    exp: 'RMSE in dollars provides direct interpretation: "average error of $X". R² is dimensionless and does not indicate error magnitude. Log-Loss and F1 are for classification.'
  }
]);
</script>
