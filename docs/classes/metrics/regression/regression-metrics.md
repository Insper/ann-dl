
Regression tasks predict continuous values. The following metrics evaluate the accuracy of predicted values against true values:

1. **Mean Absolute Error (MAE)**
   - **Purpose**: Measures the average absolute difference between predictions and true values.
   - **Formula**: \( \text{MAE} = \frac{1}{N} \sum_{i=1}^N |y_i - \hat{y}_i| \)
     - \( y_i \): True value, \( \hat{y}_i \): Predicted value, \( N \): Number of samples.
   - **Use Case**: Robust to outliers, interpretable as average error.

2. **Mean Squared Error (MSE)**
   - **Purpose**: Measures the average squared difference between predictions and true values.
   - **Formula**: \( \text{MSE} = \frac{1}{N} \sum_{i=1}^N (y_i - \hat{y}_i)^2 \)
   - **Use Case**: Sensitive to outliers, commonly used in neural network loss functions.

3. **Root Mean Squared Error (RMSE)**
   - **Purpose**: Square root of MSE, providing error in the same units as the target.
   - **Formula**: \( \text{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^N (y_i - \hat{y}_i)^2} \)
   - **Use Case**: Preferred for interpretable error magnitude, widely used in forecasting.

4. **Mean Absolute Percentage Error (MAPE)**
   - **Purpose**: Measures average percentage error relative to true values.
   - **Formula**: \( \text{MAPE} = \frac{1}{N} \sum_{i=1}^N \left| \frac{y_i - \hat{y}_i}{y_i} \right| \cdot 100 \)
   - **Use Case**: Useful when relative errors matter (e.g., financial predictions), but sensitive to zero or near-zero true values.

5. **R-Squared (Coefficient of Determination)**
   - **Purpose**: Measures the proportion of variance in the dependent variable explained by the model.
   - **Formula**: \( R^2 = 1 - \frac{\sum_{i=1}^N (y_i - \hat{y}_i)^2}{\sum_{i=1}^N (y_i - \bar{y})^2} \)
     - \( \bar{y} \): Mean of true values.
   - **Use Case**: Indicates model fit, with values closer to 1 indicating better fit.

6. **Adjusted R-Squared**
   - **Purpose**: Adjusts R² for the number of predictors, penalizing overly complex models.
   - **Formula**: \( \text{Adjusted } R^2 = 1 - \left( \frac{(1 - R^2)(N - 1)}{N - k - 1} \right) \)
     - \( k \): Number of predictors.
   - **Use Case**: Useful when comparing models with different numbers of features.

7. **Median Absolute Error**
   - **Purpose**: Measures the median of absolute differences, highly robust to outliers.
   - **Formula**: \( \text{MedAE} = \text{median}(|y_1 - \hat{y}_1|, \dots, |y_N - \hat{y}_N|) \)
   - **Use Case**: Preferred in datasets with extreme values or non-Gaussian errors.

8. **Huber Loss**
   - **Purpose**: Combines MSE and MAE, less sensitive to outliers than MSE.
   - **Formula**: 
     \[
     L_\delta(y_i, \hat{y}_i) = 
     \begin{cases} 
     \frac{1}{2}(y_i - \hat{y}_i)^2 & \text{if } |y_i - \hat{y}_i| \leq \delta \\
     \delta |y_i - \hat{y}_i| - \frac{1}{2}\delta^2 & \text{otherwise}
     \end{cases}
     \]
   - **Use Case**: Used in robust regression tasks, often as a loss function in neural networks.
