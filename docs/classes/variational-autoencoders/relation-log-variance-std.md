
#### 1. Definitions

For a random variable ( x ) that follows a normal distribution:

\[
x \sim \mathcal{N}(\mu, \sigma^2)
\]

where:

* \( \mu \): mean
* \( \sigma^2 \): variance
* \( \sigma \): standard deviation

---

#### 2. Log variance

Often, instead of directly predicting or storing the variance \( \sigma^2 \) or standard deviation \( \sigma \), models work with the **log variance**:

\[
\displaystyle \text{log_var} = \log(\sigma^2)
\]

---

#### 3. Relationship between log variance and std

From the above definition:

\[
\displaystyle \sigma^2 = e^{\text{log_var}}
\]

Taking the square root to get the standard deviation:

\[
\displaystyle \sigma = \displaystyle \sqrt{e^{\text{log_var}}} = \displaystyle e^{\frac{1}{2}\text{log_var}}
\]

**So:**

\[
\displaystyle \boxed{\sigma = \exp\left(\frac{1}{2} \cdot \text{log_var}\right)}
\]

and conversely,

\[
\displaystyle \boxed{\text{log_var} = 2 \cdot \log(\sigma)}
\]

---

#### 4. Why use log variance?

It’s common in neural nets because:

* It ensures the variance is always **positive** (since \( e^x > 0 \)).
* It’s numerically **more stable** when optimizing.
* It allows unconstrained outputs from the network (no need to force positivity).

---

#### Summary

| Quantity       | Expression         | In terms of log_var               |
| -------------- | ------------------ | --------------------------------- |
| Variance       | \( \sigma^2 \)       | \( e^{\text{log_var}} \)            |
| Std. deviation | \( \sigma \)         | \( e^{\frac{1}{2}\text{log_var}} \) |
| Log variance   | \( \text{log_var} \) | \( 2 \log(\sigma) \)                |
