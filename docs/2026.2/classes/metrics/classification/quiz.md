<div id="quiz-metrics-clf"></div>
<script>
buildQuiz('metrics-clf', 'Classification Metrics', [
  {
    q: 'True Positives (TP) = 90, False Negatives (FN) = 10. What is the Recall?',
    opts: ['90%', '80%', '75%', 'Cannot calculate without FP'],
    ans: 0,
    exp: 'Recall = TP/(TP+FN) = 90/(90+10) = 0.90 = 90%. It measures the ability to find all actual positives — critical in medical diagnostics where false negatives are costly.'
  },
  {
    q: 'In fraud detection with 99% negatives and 1% positives, why is accuracy a misleading metric?',
    opts: [
      'Accuracy does not work for binary problems',
      'A model that always predicts "no fraud" achieves 99% accuracy but zero utility for detecting fraud',
      'Accuracy only works with balanced and normalized data',
      'Accuracy cannot be computed without the full confusion matrix'
    ],
    ans: 1,
    exp: 'With extreme imbalance, predicting the majority class gives high accuracy. Use F1-Score, AUC-ROC, or Precision-Recall curves that explicitly account for both classes.'
  },
  {
    q: 'F1-Score is defined as:',
    opts: [
      'Arithmetic mean of Precision and Recall',
      'Harmonic mean of Precision and Recall: 2·P·R/(P+R)',
      'Precision divided by Recall',
      'Accuracy weighted by number of classes'
    ],
    ans: 1,
    exp: 'F1 = 2PR/(P+R) is the harmonic mean, which penalizes imbalances between P and R. If P=1 and R=0, F1=0 (no favorable trade-off). Useful when both metrics matter equally.'
  },
  {
    q: 'What does the Area Under the ROC Curve (AUC-ROC) represent?',
    opts: [
      'The maximum achievable accuracy of the model',
      'The probability that the model ranks a random positive higher than a random negative',
      'The optimal classification threshold',
      'The average loss over the validation set'
    ],
    ans: 1,
    exp: 'AUC-ROC measures discriminative ability independent of threshold. AUC=0.5 is random; AUC=1.0 is perfect. Useful for comparing models with different precision/recall trade-offs.'
  },
  {
    q: 'When is high Precision preferable over high Recall?',
    opts: [
      'Never — Recall is always more important',
      'In email spam filtering: false positives (spam in inbox) are more annoying than false negatives (missed spam)',
      'In cancer diagnosis, where false negatives are critical',
      'When the dataset is balanced'
    ],
    ans: 1,
    exp: 'High Precision reduces false positives — important when a false accusation is costly (e.g., flagging legitimate emails as spam, content recommendation). High Recall reduces false negatives — vital when missing a case is dangerous (e.g., cancer, fraud).'
  }
]);
</script>
