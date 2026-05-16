<div id="quiz-metrics-reg"></div>
<script>
buildQuiz('metrics-reg', 'Métricas de Regressão', [
  {
    q: 'Qual é a diferença entre MAE e MSE?',
    opts: [
      'MAE usa médias; MSE usa medianas',
      'MAE calcula a média dos erros absolutos; MSE calcula a média dos erros ao quadrado — penalizando erros grandes com mais intensidade',
      'MSE é sempre maior que MAE',
      'MAE é para classificação; MSE é para regressão'
    ],
    ans: 1,
    exp: 'MAE = média(|y - ŷ|): robusto a outliers. MSE = média((y-ŷ)²): amplifica erros grandes. RMSE = √MSE tem a mesma unidade do alvo, tornando-o mais interpretável que o MSE.'
  },
  {
    q: 'O que é R² (coeficiente de determinação)?',
    opts: [
      'A correlação de Pearson entre predições e rótulos',
      'A proporção da variância do alvo explicada pelo modelo (R²=1: perfeito, R²=0: igual à média)',
      'A raiz do erro quadrático médio',
      'O coeficiente de regularização L2'
    ],
    ans: 1,
    exp: 'R² = 1 - SS_res/SS_tot. SS_res é a variância residual (erros do modelo); SS_tot é a variância total dos dados. R²=0,85 significa que o modelo explica 85% da variabilidade no alvo.'
  },
  {
    q: 'Quando o RMSE é preferível ao MAE?',
    opts: [
      'Quando o dataset contém muitos outliers',
      'Quando erros grandes são desproporcionalmente piores — RMSE penaliza desvios extremos mais fortemente',
      'Quando queremos uma métrica na mesma escala do alvo sem penalização extra',
      'RMSE é sempre preferível ao MAE'
    ],
    ans: 1,
    exp: 'RMSE penaliza erros grandes mais que o MAE por causa do quadrado. Use RMSE quando um erro de 10 é muito pior que dez erros de 1 (ex: previsão de demanda de pico). Use MAE quando todos os erros têm custo igual.'
  },
  {
    q: 'Um modelo de regressão tem R² = -0,3 no conjunto de teste. O que isso significa?',
    opts: [
      'O modelo tem 30% de acurácia',
      'O modelo tem desempenho PIOR do que simplesmente prever a média dos dados',
      'O modelo tem 70% de erro',
      'O modelo está sofrendo leve underfitting'
    ],
    ans: 1,
    exp: 'R² negativo significa SS_res > SS_tot — o modelo comete mais erro do que simplesmente prever a média. Pode indicar vazamento de dados invertido, features irrelevantes ou um bug na implementação.'
  },
  {
    q: 'Qual métrica é mais adequada para prever preços de casas, onde a magnitude do erro importa?',
    opts: ['R²', 'RMSE (em reais)', 'Log-Loss', 'F1-Score'],
    ans: 1,
    exp: 'RMSE em reais fornece interpretação direta: "erro médio de R$ X". R² não tem dimensão e não indica a magnitude do erro. Log-Loss e F1 são para classificação.'
  }
]);
</script>
