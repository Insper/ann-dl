<div id="quiz-metrics-clf"></div>
<script>
buildQuiz('metrics-clf', 'Métricas de Classificação', [
  {
    q: 'Verdadeiros Positivos (VP) = 90, Falsos Negativos (FN) = 10. Qual é o Recall?',
    opts: ['90%', '80%', '75%', 'Não é possível calcular sem FP'],
    ans: 0,
    exp: 'Recall = VP/(VP+FN) = 90/(90+10) = 0,90 = 90%. Mede a capacidade de encontrar todos os positivos reais — crítico em diagnósticos médicos onde falsos negativos são custosos.'
  },
  {
    q: 'Na detecção de fraudes com 99% de negativos e 1% de positivos, por que a acurácia é uma métrica enganosa?',
    opts: [
      'A acurácia não funciona para problemas binários',
      'Um modelo que sempre prevê "sem fraude" atinge 99% de acurácia mas tem utilidade zero para detectar fraudes',
      'A acurácia só funciona com dados balanceados e normalizados',
      'A acurácia não pode ser calculada sem a matriz de confusão completa'
    ],
    ans: 1,
    exp: 'Com desequilíbrio extremo, prever a classe majoritária dá alta acurácia. Use F1-Score, AUC-ROC ou curvas Precisão-Recall que consideram explicitamente ambas as classes.'
  },
  {
    q: 'O F1-Score é definido como:',
    opts: [
      'Média aritmética de Precisão e Recall',
      'Média harmônica de Precisão e Recall: 2·P·R/(P+R)',
      'Precisão dividida pelo Recall',
      'Acurácia ponderada pelo número de classes'
    ],
    ans: 1,
    exp: 'F1 = 2PR/(P+R) é a média harmônica, que penaliza desequilíbrios entre P e R. Se P=1 e R=0, F1=0 (sem trade-off favorável). Útil quando ambas as métricas importam igualmente.'
  },
  {
    q: 'O que a Área Sob a Curva ROC (AUC-ROC) representa?',
    opts: [
      'A acurácia máxima alcançável do modelo',
      'A probabilidade de o modelo classificar um positivo aleatório acima de um negativo aleatório',
      'O limiar de classificação ótimo',
      'A perda média sobre o conjunto de validação'
    ],
    ans: 1,
    exp: 'AUC-ROC mede a capacidade discriminativa independente do limiar. AUC=0,5 é aleatório; AUC=1,0 é perfeito. Útil para comparar modelos com diferentes trade-offs de precisão/recall.'
  },
  {
    q: 'Quando a alta Precisão é preferível ao alto Recall?',
    opts: [
      'Nunca — o Recall é sempre mais importante',
      'Na filtragem de spam: falsos positivos (spam na caixa de entrada) são mais inconvenientes que falsos negativos (spam perdido)',
      'No diagnóstico de câncer, onde falsos negativos são críticos',
      'Quando o dataset é balanceado'
    ],
    ans: 1,
    exp: 'Alta Precisão reduz falsos positivos — importante quando uma falsa acusação é custosa (ex: marcar e-mails legítimos como spam, recomendação de conteúdo). Alto Recall reduz falsos negativos — vital quando perder um caso é perigoso (ex: câncer, fraude).'
  }
]);
</script>
