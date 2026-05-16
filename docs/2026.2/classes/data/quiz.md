<div id="quiz-data"></div>
<script>
buildQuiz('data', 'Data', [
  {
    q: 'What are features in a machine learning dataset?',
    opts: [
      'The output variables the model must predict',
      'The input variables used to make predictions',
      'The number of samples in the dataset',
      'The model hyperparameters'
    ],
    ans: 1,
    exp: 'Features are the input columns of a dataset. For example, in house price prediction: area, number of rooms, location. The model learns how they relate to the target variable.'
  },
  {
    q: 'What is class imbalance?',
    opts: [
      'When the model has more parameters than training samples',
      'When some classes have far more samples than others',
      'When input data is not normalized',
      'When there are missing values in the dataset'
    ],
    ans: 1,
    exp: 'Class imbalance occurs when the class distribution is very unequal (e.g., 95% negative, 5% positive). Models tend to predict the majority class, making accuracy metrics misleading. Use F1, AUC-ROC, or SMOTE resampling.'
  },
  {
    q: 'What is the purpose of train/validation/test splits?',
    opts: [
      'Train: weight updates; Validation: hyperparameter tuning; Test: final unbiased evaluation',
      'Train: evaluation; Validation: training; Test: tuning',
      'All three sets are used to train the model',
      'The split is only used to speed up training'
    ],
    ans: 0,
    exp: 'The training set adjusts weights. Validation monitors performance during development and guides hyperparameter choices. The test set evaluates the final model once, simulating real-world performance.'
  },
  {
    q: 'What is a Gaussian (normal) distribution in data?',
    opts: [
      'Data uniformly distributed over an interval',
      'Data concentrated around a mean, with symmetric spread described by standard deviation',
      'Data with only two possible values (0 or 1)',
      'Data sorted in ascending order'
    ],
    ans: 1,
    exp: 'The normal distribution (bell curve) has mean μ and standard deviation σ. ~68% of data lies within ±1σ, ~95% within ±2σ. Many natural phenomena follow this distribution.'
  },
  {
    q: 'What is Data Leakage?',
    opts: [
      'Losing data during training due to disk failure',
      'Including test set information in training, artificially inflating metrics',
      'Incorrect normalization of features',
      'Using irrelevant features in the model'
    ],
    ans: 1,
    exp: 'Data leakage occurs when the model improperly accesses future or test-set information during training, resulting in optimistic metrics that do not hold in production.'
  }
]);
</script>
