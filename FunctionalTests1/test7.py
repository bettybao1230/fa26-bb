import pandas as pd
import numpy as np
from scipy.stats import t
from scipy.optimize import minimize
from scipy.stats import skew, kurtosis
from scipy.stats import norminvgauss



# Test 7.1. Fit Normal Distribution.
data_test7_1 = pd.read_csv('test7_1.csv')

mu = data_test7_1['x1'].mean()
sigma = data_test7_1['x1'].std()  # ddof=1 by default in pandas

musigma = pd.DataFrame(
    [[mu, sigma]],
    columns=["mu", "sigma"]
)

print(musigma)




# Test 7.2. Fit T Distribution.
data_test7_2 = pd.read_csv('test7_2.csv')

# Fit Student's t-distribution: returns (degrees of freedom, loc, scale)
nu, mu, sigma = t.fit(data_test7_2['x1'])

result = pd.DataFrame(
    [[mu, sigma, nu]],
    columns=["mu", "sigma", "nu"]
)

print(result)




# Test 7.3. T Regression.
data_test7_3 = pd.read_csv('test7_3.csv')
y = data_test7_3['y'].values
X = data_test7_3.drop(columns=['y']).values

# Add intercept: X = [1, X1, X2, ...]
X = np.column_stack([np.ones(len(y)), X])

n_params = X.shape[1]  # number of betas (including intercept)


def neg_log_likelihood(params, X, y):
    betas = params[:n_params]
    sigma = params[n_params]
    nu = params[n_params + 1]

    if sigma <= 0 or nu <= 0:
        return np.inf

    resid = y - X @ betas
    # standardized residuals under a scaled t-distribution
    ll = np.sum(t.logpdf(resid / sigma, df=nu) - np.log(sigma))
    return -ll


# Initial guesses: OLS betas, residual std, and nu=5
ols_betas = np.linalg.lstsq(X, y, rcond=None)[0]
resid0 = y - X @ ols_betas
init_params = np.concatenate([ols_betas, [resid0.std(), 5.0]])

result = minimize(
    neg_log_likelihood,
    init_params,
    args=(X, y),
    method='Nelder-Mead',
    options={'xatol': 1e-8, 'fatol': 1e-8, 'maxiter': 10000}
)

betas_fit = result.x[:n_params]
sigma_fit = result.x[n_params]
nu_fit = result.x[n_params + 1]
alpha_fit, coef_fit = betas_fit[0], betas_fit[1:]

# location is fixed at 0 by construction, not estimated
row = np.concatenate([[0.0, sigma_fit, nu_fit, alpha_fit], coef_fit])
columns = ["mu", "sigma", "nu", "Alpha"] + [f"B{i}" for i in range(1, n_params)]

out = pd.DataFrame([row], columns=columns)
print(out)





# Test 7.4. Calculate AICC on fitted T.
# ------------------------------------------------------------
# Fit Student-t distribution using MLE
# ------------------------------------------------------------
x = data_test7_2['x1']

nu, mu, sigma = t.fit(
    x,
    method="MLE",
    optimizer=lambda func, x0, args, disp:
        minimize(
            func,
            x0,
            args=args,
            method="Nelder-Mead",
            options={
                "xatol": 1e-12,
                "fatol": 1e-12,
                "maxiter": 50000
            }
        ).x
)

# ------------------------------------------------------------
# Calculate log-likelihood
# ------------------------------------------------------------
n = len(x)
k = 3

log_likelihood = np.sum(
    t.logpdf(x, df=nu, loc=mu, scale=sigma)
)

# ------------------------------------------------------------
# Calculate AIC
# AIC = 2k - 2 log(L)
# ------------------------------------------------------------
aic = 2 * k - 2 * log_likelihood

# ------------------------------------------------------------
# Calculate AICc
# AICc = AIC + 2k(k+1)/(n-k-1)
# ------------------------------------------------------------
aicc = aic + (2 * k * (k + 1)) / (n - k - 1)

aicc_table = pd.DataFrame({"AICC": [aicc]})
print(aicc_table)




# Test 7.5. Fit a Normal Inverse Gaussian by the method of moments. Returns mu, alpha, beta, delta.
data_test7_5 = pd.read_csv('test7_5.csv')
x = data_test7_5['x1']

# ------------------------------------------------------------
# Fit Normal Inverse Gaussian (NIG) by Method of Moments
#
# Parameters:
#   mu    : location
#   alpha : tail parameter
#   beta  : skewness parameter
#   delta : scale parameter
# ------------------------------------------------------------

# Sample moments
mean = np.mean(x)
variance = np.var(x, ddof=1)
skewness = skew(x, bias=True)
excess_kurtosis = kurtosis(x, fisher=True, bias=True)

# ------------------------------------------------------------
# NIG moment equations
#
# gamma = sqrt(alpha^2 - beta^2)
#
# Mean:
#   E[X] = mu + delta * beta / gamma
#
# Variance:
#   Var[X] = delta * alpha^2 / gamma^3
#
# Skewness:
#   S = 3 * beta / (alpha * sqrt(delta * gamma))
#
# Excess kurtosis:
#   K = 3 * (1 + 4 * beta^2 / alpha^2) / (delta * gamma)
# ------------------------------------------------------------

# Step 1: Solve for (beta / alpha)^2
rho2 = skewness**2 / (
    3 * excess_kurtosis - 4 * skewness**2
)

# Step 2: Recover beta / alpha, including its sign
ratio = np.sign(skewness) * np.sqrt(rho2)

# Step 3: Solve for alpha using the variance and skewness
alpha = np.sqrt(
    (9 * ratio**2 / skewness**2)
    / (variance * (1 - ratio**2)**2)
)

# beta
beta = ratio * alpha

# gamma = sqrt(alpha^2 - beta^2)
gamma = np.sqrt(alpha**2 - beta**2)

# delta
delta = variance * gamma**3 / alpha**2

# mu
mu = mean - delta * beta / gamma


output_7_5 = pd.DataFrame(
    [[mu, alpha, beta, delta]],
    columns=["mu", "alpha", "beta", "delta"]
)

print(output_7_5)





# Test 7.6. Fit the same NIG by maximum likelihood. Returns mu, alpha, beta, delta.
"""
Fit a Normal Inverse Gaussian (NIG) distribution by Maximum Likelihood.

scipy's norminvgauss uses parameters (a, b, loc, scale) where the
distribution is standardized: a = alpha*delta, b = beta*delta,
loc = mu, scale = delta. Converting back to the (mu, alpha, beta, delta)
parametrization:
    mu    = loc
    delta = scale
    alpha = a / scale
    beta  = b / scale
"""
data_test7_5 = pd.read_csv('test7_5.csv')
x = data_test7_5["x1"].values

a, b, loc, scale = norminvgauss.fit(x)

mu = loc
delta = scale
alpha = a / scale
beta = b / scale

output_7_6 = pd.DataFrame([[mu, alpha, beta, delta]], columns=["mu", "alpha", "beta", "delta"])
print(output_7_6)




