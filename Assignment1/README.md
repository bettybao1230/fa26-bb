# Assignment 1 README

## Overview

This repository contains the Python code used to produce the numerical results and figures reported in the written responses for the assignment.

The analysis is organized into five Python scripts:

-   `problem1.py` --- Reading the Shape of a Sample
-   `problem2.py` --- A Regression Whose Errors Are Not Normal
-   `problem3.py` --- Pearson Against Spearman
-   `problem4.py` --- Conditional Distributions
-   `problem5.py` --- Identifying an AR or MA Order and Fitting AR/MA Models

Each script reads the corresponding CSV data file and reproduces the numerical results and figures used in the written responses.

------------------------------------------------------------------------

## Requirements

The code requires **Python 3** and the following packages:

-   `numpy`
-   `pandas`
-   `scipy`
-   `matplotlib`
-   `statsmodels`

Install all required packages with:

``` bash
pip install numpy pandas scipy matplotlib statsmodels
```

A virtual environment may also be used:

``` bash
python -m venv .venv
```

Activate it on macOS/Linux with:

``` bash
source .venv/bin/activate
```

Then install the dependencies:

``` bash
pip install numpy pandas scipy matplotlib statsmodels
```

------------------------------------------------------------------------

## Directory Structure

Place the scripts and data files in the same directory:

``` text
Assignment1/
├── README.md
├── problem1.py
├── problem1.csv
├── problem2.py
├── problem2.csv
├── problem3.py
├── problem3.csv
├── problem4.py
├── problem4.csv
├── problem5.py
├── problem5.csv
└── responses.pdf
```

The scripts use relative file paths such as:

``` python
pd.read_csv("problem1.csv")
```

Therefore, the scripts should be run from the directory containing the corresponding CSV files.

------------------------------------------------------------------------

# Problem 1 --- Reading the Shape of a Sample

## Files

``` text
problem1.py
problem1.csv
```

## Run

``` bash
python problem1.py
```

## What the Script Calculates

The script:

1.  Loads `problem1.csv`.

2.  Extracts the `x` series.

3.  Removes missing observations.

4.  Computes the first four sample moments:

    -   mean;
    -   variance;
    -   skewness;
    -   excess kurtosis.

5.  Fits a Normal distribution using the sample mean and sample standard deviation.

6.  Calculates the fitted Normal 1% lower-tail quantile.

7.  Counts the observed values below this quantile.

8.  Calculates the expected number of observations below the 1% quantile.

The script prints all of these numerical results.

## Statistical Conventions

Missing observations are removed using:

``` python
x = data1["x"].dropna().to_numpy()
```

The variance is calculated using:

``` python
np.var(x, ddof=1)
```

Therefore, the sample variance uses $n-1$ in the denominator.

Skewness is calculated using the bias-corrected sample version:

``` python
stats.skew(x, bias=False)
```

Excess kurtosis is calculated using:

``` python
stats.kurtosis(
    x,
    fisher=True,
    bias=False
)
```

The option `fisher=True` reports excess kurtosis, so a Normal distribution has excess kurtosis equal to zero.

The fitted Normal distribution uses the sample mean and sample standard deviation as its estimated parameters.

The 1% lower-tail threshold is

$$
q_{0.01} = F^{-1}(0.01),
$$

where $F$ is the CDF of the fitted Normal distribution.

------------------------------------------------------------------------

# Problem 2 --- A Regression Whose Errors Are Not Normal

## Files

``` text
problem2.py
problem2.csv
```

## Run

``` bash
python problem2.py
```

## What the Script Calculates

The script:

1.  Plots $y$ against $x$.
2.  Fits the regression using OLS.
3.  Estimates the model under Normal errors using maximum likelihood.
4.  Estimates the model under Student's $t$ errors using maximum likelihood.
5.  Calculates AICc for both models.
6.  Compares the two AICc values.
7.  Reports the preferred model.

The script prints:

-   OLS $\alpha$ and $\beta$;
-   OLS standard errors;
-   OLS residual standard deviation;
-   Normal MLE estimates;
-   Normal AICc;
-   Student's $t$ MLE estimates;
-   Student's $t$ degrees of freedom;
-   Student's $t$ AICc;
-   preferred model.

## OLS Convention

The regression model is

$$
y_i = \alpha + \beta x_i + \epsilon_i.
$$

The intercept is included using:

``` python
X = sm.add_constant(x)
```

The residual standard deviation is calculated as

$$\boxed{\hat{\sigma}=\sqrt{\frac{\mathrm{RSS}}{n-2}}}$$

The denominator is $n-2$ because two regression coefficients, $\alpha$ and $\beta$, are estimated.

## Normal MLE Convention

The Normal error model assumes

$$
\epsilon_i \sim N(0,\sigma^2).
$$

The estimated parameters are

$$
\alpha,\quad \beta,\quad \sigma.
$$

The optimization is performed over $\log(\sigma)$, with

$$
\sigma = e^{\log \sigma},
$$

which guarantees that $\sigma > 0$.

The OLS estimates are used as starting values because the OLS estimates of $\alpha$ and $\beta$ coincide with their Normal-error MLEs.

The Normal model therefore has

$$
k = 3
$$

estimated parameters.

## Student's $t$ MLE Convention

The Student's $t$ error model assumes

$$
\epsilon_i \sim t_\nu(0,\sigma).
$$

The estimated parameters are

$$
\alpha,\quad \beta,\quad \sigma,\quad \nu.
$$

The code optimizes transformed parameters:

$$
\sigma = e^{\log \sigma}
$$

and

$$
\nu = 2 + e^{\log(\nu-2)}.
$$

Therefore,

$$
\sigma > 0
$$

and

$$
\nu > 2.
$$

The restriction $\nu > 2$ ensures that the fitted Student's $t$ distribution has finite variance.

The starting value for the degrees of freedom is

$$
\nu = 10.
$$

The OLS estimates are used as starting values for $\alpha$, $\beta$, and $\sigma$.

The Student's $t$ model therefore has

$$
k = 4
$$

estimated parameters.

## AICc Convention

The models are compared using

$$\mathrm{AIC}=2k-2\log L$$

and

$$\mathrm{AICc}=\mathrm{AIC}+\frac{2k(k+1)}{n-k-1}.$$

The parameter count includes the regression coefficients and the error-scale parameter.

Thus,

$$
k_{\mathrm{Normal}} = 3
$$

and

$$
k_t = 4.
$$

The model with the smaller AICc is preferred.

------------------------------------------------------------------------

# Problem 3 --- Pearson Against Spearman

## Files

``` text
problem3.py
problem3.csv
```

## Run

``` bash
python problem3.py
```

## What the Script Calculates

The script computes Pearson and Spearman correlations for all six unique pairs among

$$
x_1,x_2,x_3,x_4.
$$

The six pairs are:

``` text
x1 vs x2
x1 vs x3
x1 vs x4
x2 vs x3
x2 vs x4
x3 vs x4
```

For each pair, the script reports:

-   Pearson correlation;
-   Spearman correlation;
-   absolute Pearson-Spearman difference.

It also produces scatterplots for all six pairs and prints the Pearson and Spearman correlation matrices.

Finally, it identifies the pair with the largest Pearson-Spearman gap.

## Correlation Conventions

Pearson correlation is calculated with:

``` python
pearsonr(x, y).statistic
```

Pearson correlation measures linear association.

Spearman correlation is calculated with:

``` python
spearmanr(x, y).statistic
```

Spearman correlation measures monotonic association using ranks.

The reported difference is

$$\left|\rho_{\mathrm{Pearson}}-\rho_{\mathrm{Spearman}}\right|.$$

A substantial difference between Pearson and Spearman correlations can indicate that the relationship is not well described by a simple linear relationship.

------------------------------------------------------------------------

# Problem 4 --- Conditional Distributions

## Files

``` text
problem4.py
problem4.csv
```

## Run

``` bash
python problem4.py
```

## What the Script Calculates

The script:

1.  Loads `problem4.csv`.

2.  Computes the sample means of $X_1$ and $X_2$.

3.  Computes the sample covariance matrix.

4.  Partitions the covariance matrix into scalar blocks.

5.  Computes the conditional expectation $E[X_2 \mid X_1=x_1]$.

6.  Computes the conditional variance $\mathrm{Var}(X_2 \mid X_1)$.

7.  Constructs the 95% conditional band.

8.  Plots the observations, conditional expectation, and conditional band.

9.  Computes overall coverage.

10. Computes coverage within three regions of $X_1$:

    -   within 1 standard deviation;
    -   between 1 and 2 standard deviations;
    -   beyond 2 standard deviations.

## Covariance Convention

The covariance matrix is calculated using:

``` python
np.cov(
    np.column_stack((x1, x2)),
    rowvar=False
)
```

`np.cov()` uses the sample covariance estimator, dividing by $n-1$.

The covariance matrix is partitioned as

$$\Sigma =\begin{pmatrix}
\sigma_{11} & \sigma_{12} \\
\sigma_{21} & \sigma_{22}
\end{pmatrix}.
$$

Because $X_1$ and $X_2$ are scalars,

$$\sigma_{11} = \mathrm{Var}(X_1),$$

$$\sigma_{22} = \mathrm{Var}(X_2),$$

and

$$\sigma_{21}=\sigma_{12}=\mathrm{Cov}(X_2,X_1).$$

## Conditional Expectation

Under the joint Normal assumption,

$$E[X_2 \mid X_1=x_1]=\mu_2+\frac{\sigma_{21}}{\sigma_{11}}(x_1-\mu_1).$$

The conditional-expectation slope is therefore

$$\beta_{\mathrm{cond}}=\frac{\sigma_{21}}{\sigma_{11}}.$$

## Conditional Variance

Under the joint Normal assumption,

$$\mathrm{Var}(X_2 \mid X_1)=\sigma_{22}-\frac{\sigma_{21}\sigma_{12}}{\sigma_{11}}.$$

Since $X_1$ and $X_2$ are scalars, this is equivalent to

$$\mathrm{Var}(X_2 \mid X_1)=\sigma_{22}-\frac{\sigma_{21}^2}{\sigma_{11}}.$$

An important consequence is that the conditional variance does not depend on the value of $x_1$. Therefore, the resulting conditional band has constant width.

## 95% Conditional Band

The code uses

$$
z_{0.95} = 1.96
$$

and constructs

$$
E[X_2 \mid X_1=x_1]
\pm
1.96
\sqrt{
\mathrm{Var}(X_2 \mid X_1)
}.
$$

This is a 95% conditional interval under the joint Normal assumption.

The value $1.96$ is explicitly used in the code rather than obtained from a library function.

## Plotting Convention

The observations are not necessarily ordered by $x_1$. Therefore, the code sorts the $x_1$ values before plotting the conditional expectation and the upper and lower boundaries.

This allows the lines to be displayed continuously from low to high $x_1$, without changing the underlying calculations.

## Coverage Calculation

An observation is considered covered when

$$
\mathrm{lower}_i \leq x_{2,i} \leq \mathrm{upper}_i
$$

The overall coverage is the fraction of observations satisfying this condition.

The code also calculates coverage by standardized distance from the sample mean of $x_1$:

$$z_i=\left|\frac{x_{1,i}-\mu_1}{\mathrm{SD}(X_1)}\right|.$$

The three regions are

$$
z_i \leq 1,
$$

$$
1 < z_i \leq 2,
$$

and

$$
z_i > 2.
$$

The reported coverage values are calculated separately within these three groups.

------------------------------------------------------------------------

# Problem 5 --- Identifying an AR or MA Order

## Files

``` text
problem5.py
problem5.csv
```

## Run

``` bash
python problem5.py
```

## What the Script Calculates

The script:

1.  Loads the time series.
2.  Removes missing observations.
3.  Plots the original time series.
4.  Plots the ACF.
5.  Plots the PACF.
6.  Fits AR(1), AR(2), and AR(3).
7.  Fits MA(1), MA(2), and MA(3).
8.  Calculates AICc for all six models.
9.  Prints the AICc comparison.

## ACF and PACF Convention

The ACF is calculated using a 95% confidence level:

``` python
plot_acf(
    x,
    lags=max_lag,
    alpha=0.05,
    ax=axes[1]
)
```

The PACF is calculated using:

``` python
plot_pacf(
    x,
    lags=max_lag,
    alpha=0.05,
    method="ywm",
    ax=axes[2]
)
```

The PACF method is explicitly set to `"ywm"` rather than relying on the library default. This makes the calculation reproducible and specifies exactly which PACF estimator is being used.

The maximum number of plotted lags is

$$
\min(20,\lfloor n/2 \rfloor - 1).
$$

This limits the plot to at most 20 lags while avoiding an excessive number of lags for a small sample.

## AR Model Convention

The AR models are fitted using:

``` python
AutoReg(
    x,
    lags=p,
    trend="c"
).fit()
```

The option `trend="c"` includes a constant/intercept in the AR model.

For an AR($p$) model, the parameter count used for AICc is

$$
k = p + 2,
$$

because the model includes:

-   $p$ AR coefficients;
-   one intercept;
-   one error variance.

## MA Model Convention

The MA models are fitted using:

``` python
ARIMA(
    x,
    order=(0, 0, q),
    trend="c"
).fit()
```

The option `trend="c"` includes a constant/intercept.

For an MA($q$) model, the parameter count used for AICc is

$$
k = q + 2,
$$

because the model includes:

-   $q$ MA coefficients;
-   one intercept;
-   one error variance.

The constant-term convention is therefore consistent across the AR and MA models.

## AICc Convention

The corrected Akaike Information Criterion is calculated as

$$\mathrm{AICc}=\mathrm{AIC}+\frac{2k(k+1)}{n-k-1}.$$

The code uses the effective number of observations reported by the fitted model:

``` python
n_obs = model.nobs
```

rather than automatically using the original length of the raw series.

This matters for lagged models because the number of observations used in estimation can differ from the original sample size.

For the candidate models,

$$k_{\mathrm{AR}(p)} = p + 2$$

and

$$k_{\mathrm{MA}(q)} = q + 2.$$

The model with the smallest AICc is preferred among the candidate models.

------------------------------------------------------------------------

# Reproducing Every Numerical Result

The complete set of numerical results can be reproduced with:

``` bash
python problem1.py
python problem2.py
python problem3.py
python problem4.py
python problem5.py
```

| Problem | Command              | Main Numerical Results                          |
|-----------------|-----------------|---------------------------------------|
| 1       | `python problem1.py` | Mean, variance, skewness, kurtosis, tail counts |
| 2       | `python problem2.py` | OLS, MLE, AICc, preferred model                 |
| 3       | `python problem3.py` | Pearson/Spearman correlations and largest gap   |
| 4       | `python problem4.py` | Covariance, conditional moments, coverage       |
| 5       | `python problem5.py` | ACF/PACF and AR/MA AICc                         |

The scripts print the numerical results directly to the terminal.

------------------------------------------------------------------------

# Figures

The following figures are generated by the scripts.

## Problem 2

A scatterplot of $y$ against $x$.

## Problem 3

Six scatterplots showing the pairwise relationships among

$$
x_1,x_2,x_3,x_4.
$$

## Problem 4

A scatterplot of $x_2$ against $x_1$, together with:

-   the estimated conditional expectation;
-   the lower limit of the 95% conditional band;
-   the upper limit of the 95% conditional band.

## Problem 5

A three-panel figure containing:

1.  the original time series;
2.  the ACF;
3.  the PACF.

The figures are displayed automatically when the corresponding script is run.

------------------------------------------------------------------------

# Reproducibility Notes

The following conventions are explicitly specified in the code to make the results reproducible:

1.  Missing observations are removed for Problems 1 and 5 using `dropna()`.
2.  Problem 1 uses the sample variance with `ddof=1`.
3.  Problem 1 reports bias-corrected skewness and excess kurtosis.
4.  Problem 2 includes an intercept in the regression.
5.  Problem 2 estimates Normal and Student's $t$ models by maximum likelihood.
6.  Problem 2 optimizes transformed scale and degrees-of-freedom parameters to enforce valid parameter values.
7.  Problem 2 uses $k=3$ for the Normal model and $k=4$ for the Student's $t$ model in AICc.
8.  Problem 3 reports both Pearson and Spearman correlations.
9.  Problem 4 uses the sample covariance matrix with denominator $n-1$.
10. Problem 4 uses $1.96$ for the 95% conditional interval.
11. Problem 4 sorts $x_1$ only for plotting; sorting does not affect the numerical calculations.
12. Problem 5 uses `method="ywm"` for the PACF calculation.
13. Problem 5 includes a constant in both AR and MA models.
14. Problem 5 uses the fitted model's effective observation count for AICc.
15. AICc is calculated consistently across competing models using

$$\mathrm{AICc}=\mathrm{AIC}+\frac{2k(k+1)}{n-k-1}.$$

------------------------------------------------------------------------

# Troubleshooting

## `ModuleNotFoundError`

If Python reports a missing package, install all dependencies with:

``` bash
pip install numpy pandas scipy matplotlib statsmodels
```

## `FileNotFoundError`

If a CSV file cannot be found, make sure the corresponding CSV file is in the same directory as the Python script and run the script from that directory.

For example:

``` bash
cd path/to/Assignment1
python problem4.py
```

## Different Numerical Results

Small differences in the last decimal place may occur across different Python or package versions because of numerical optimization and floating-point calculations.

The code specifies the statistical conventions and optimization setup explicitly so that the results can be reproduced consistently.

------------------------------------------------------------------------

# Complete Reproduction Command

From a clean Python environment, run:

``` bash
pip install numpy pandas scipy matplotlib statsmodels
```

Then:

``` bash
python problem1.py
python problem2.py
python problem3.py
python problem4.py
python problem5.py
```

These commands reproduce the numerical calculations and figures used in the written responses for Problems 1--5.
