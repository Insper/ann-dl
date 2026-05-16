<div id="quiz-preprocessing"></div>
<script>
buildQuiz('preprocessing', 'Preprocessing', [
  {
    q: 'What is the difference between Min-Max normalization and Z-score standardization?',
    opts: [
      'Normalization maps to [0,1]; standardization transforms to mean 0 and standard deviation 1',
      'They are equivalent and produce the same result',
      'Normalization is for continuous data; standardization for categorical',
      'Standardization maps to [0,1]; normalization to mean 0'
    ],
    ans: 0,
    exp: 'Min-Max: x\' = (x-min)/(max-min) → [0,1]. Z-score: x\' = (x-μ)/σ → mean 0, std 1. Use standardization when data has outliers; normalization when bounds are known and meaningful.'
  },
  {
    q: 'What is Principal Component Analysis (PCA) used for?',
    opts: [
      'Classifying samples into groups',
      'Reducing dimensionality while preserving maximum data variance',
      'Artificially increasing the number of training samples',
      'Normalizing data to [0,1]'
    ],
    ans: 1,
    exp: 'PCA finds the axes of greatest variance (principal components) and projects data into that reduced space. Useful for visualization, compression, and removing correlated features.'
  },
  {
    q: 'What is One-Hot Encoding?',
    opts: [
      'A regularization technique for neural networks',
      'Converting categorical variables into binary vectors with a single active bit per category',
      'Normalizing numerical data to [0,1]',
      'A strategy for handling missing values'
    ],
    ans: 1,
    exp: 'One-Hot transforms "color: [red, blue, green]" into three binary columns: red=[1,0,0], blue=[0,1,0], green=[0,0,1]. Prevents the model from assuming ordinal relationships between categories.'
  },
  {
    q: 'What strategy is most appropriate for filling missing values (NaN) in a numerical feature?',
    opts: [
      'Always delete rows with NaN',
      'Replace with the column median or mean (imputation)',
      'Always replace with zero',
      'Missing values require no treatment'
    ],
    ans: 1,
    exp: 'Median imputation is robust to outliers. Mean works for symmetric distributions. Deleting rows loses information and can introduce bias if data is not Missing At Random (MAR).'
  },
  {
    q: 'Why must a scaler be fit ONLY on the training set?',
    opts: [
      'To save processing time',
      'To avoid data leakage — the scaler must not see statistics from the test set',
      'The scaler should be computed on all data for greater accuracy',
      'Because the test set does not need normalization'
    ],
    ans: 1,
    exp: 'Computing min/max or μ/σ using the test set is data leakage. The preprocessor should be fit only on training data and then applied (transform) to train and test separately.'
  }
]);
</script>
