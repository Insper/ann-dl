
Tarefas de regressão preveem valores contínuos. As seguintes métricas avaliam a acurácia dos valores previstos em relação aos valores reais:

| Métrica | Propósito | Caso de Uso |
|--------|---------|----------|
| **Erro Absoluto Médio (MAE)** <br> \( \displaystyle \frac{1}{N} \sum_{i=1}^N \vert y_i - \hat{y}_i \vert \) | Mede a diferença absoluta média entre previsões e valores reais | Robusto a outliers, interpretável como erro médio |
| **Erro Quadrático Médio (MSE)** <br> \( \displaystyle \frac{1}{N} \sum_{i=1}^N (y_i - \hat{y}_i)^2 \) | Mede a diferença quadrática média entre previsões e valores reais | Sensível a outliers, comumente usado em funções de perda de redes neurais |
| **Raiz do Erro Quadrático Médio (RMSE)** <br> \( \displaystyle \sqrt{\frac{1}{N} \sum_{i=1}^N (y_i - \hat{y}_i)^2} \) | Raiz quadrada do MSE, fornecendo o erro nas mesmas unidades que o alvo | Preferido para magnitude de erro interpretável, amplamente usado em previsão |
| **Erro Percentual Absoluto Médio (MAPE)** <br> \( \displaystyle \frac{1}{N} \sum_{i=1}^N \left \vert \frac{y_i - \hat{y}_i}{y_i} \right \vert \cdot 100 \) | Mede o erro percentual médio relativo aos valores reais | Útil quando erros relativos importam (ex: previsões financeiras), mas sensível a valores reais próximos de zero |
| **$R^2$ (Coeficiente de Determinação)** <br> \( \displaystyle 1 - \frac{\sum_{i=1}^N (y_i - \hat{y}_i)^2}{\sum_{i=1}^N (y_i - \bar{y})^2} \) | Mede a proporção da variância na variável dependente explicada pelo modelo | Indica o ajuste do modelo, com valores mais próximos de 1 indicando melhor ajuste |
| **$R^2$ Ajustado** <br> \( \displaystyle 1 - \left( \frac{(1 - R^2)(N - 1)}{N - k - 1} \right) \) | Ajusta o R² para o número de preditores, penalizando modelos excessivamente complexos | Útil ao comparar modelos com diferentes números de features |
| **Erro Absoluto Mediano ($\text{MedAE}$)** <br> \( \displaystyle \text{mediana}(\vert y_1 - \hat{y}_1 \vert, \dots, \vert y_N - \hat{y}_N \vert) \) | Mede a mediana das diferenças absolutas, altamente robusto a outliers | Preferido em datasets com valores extremos ou erros não-gaussianos |


---

--8<-- "docs/2026.2/classes/metrics/regression/quiz.pt.md"
