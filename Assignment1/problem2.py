# ==================================================
# Problem 2: A Regression Whose Errors Are Not Normal.
# ==================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.optimize import minimize
from scipy.special import gammaln


# ==================================================
# 1. Load the data
# ==================================================

data2 = pd.read_csv("problem2.csv")

x = data2["x"].to_numpy()
y = data2["y"].to_numpy()

n = len(y)


# ==================================================
# 2. Plot y against x
# ==================================================

# Plot the data before fitting the models to visualize
# the relationship between x and y.
plt.scatter(x, y)
plt.xlabel("x")
plt.ylabel("y")
plt.title("y against x")
plt.show()


# ==================================================
# 3. Ordinary Least Squares (OLS)
# ==================================================

# The regression model is
#
#     y_i = alpha + beta * x_i + epsilon_i.
#
# The intercept is estimated by adding a column of ones
# to the design matrix.

X = sm.add_constant(x)

ols = sm.OLS(y, X).fit()


# Extract the estimated intercept and slope.
alpha_ols = ols.params[0]
beta_ols = ols.params[1]


# Standard errors of the estimated coefficients.
se_alpha_ols = ols.bse[0]
se_beta_ols = ols.bse[1]


# Estimate the residual standard deviation:
#
#     sigma_hat = sqrt(RSS / (n - 2)).
#
# The denominator uses n - 2 because two regression
# coefficients, alpha and beta, are estimated.

sigma_ols = np.sqrt(
    np.sum(ols.resid ** 2) / ols.df_resid
)


# ==================================================
# 4. Maximum likelihood under Normal errors
# ==================================================

# Assume the regression errors are normally distributed:
#
#     epsilon_i ~ N(0, sigma^2).
#
# The parameters estimated by maximum likelihood are
# alpha, beta, and sigma.
#
# We optimize log(sigma) rather than sigma directly
# to guarantee that sigma remains positive.

def normal_neg_loglikelihood(parameters):

    alpha, beta, log_sigma = parameters

    sigma = np.exp(log_sigma)

    # Regression residuals:
    #
    #     epsilon_i = y_i - (alpha + beta * x_i)

    residuals = y - (alpha + beta * x)

    # Negative log-likelihood under Normal errors.

    nll = (
        n * np.log(sigma)
        + np.sum(residuals ** 2) / (2 * sigma ** 2)
        + n * np.log(2 * np.pi) / 2
    )

    return nll


# Use the OLS estimates as starting values.
# Under Normal errors, the MLEs of alpha and beta
# coincide with the OLS estimates.

initial_normal_parameters = [
    alpha_ols,
    beta_ols,
    np.log(sigma_ols)
]


# Minimize the negative log-likelihood to obtain
# the maximum likelihood estimates.

normal_mle_fit = minimize(
    normal_neg_loglikelihood,
    initial_normal_parameters,
    method="BFGS"
)


# Extract the MLEs and transform log(sigma) back
# to the original scale.

alpha_normal_mle = normal_mle_fit.x[0]
beta_normal_mle = normal_mle_fit.x[1]
sigma_normal_mle = np.exp(normal_mle_fit.x[2])


# The optimizer returns the negative log-likelihood.
# Negating it gives the maximized log-likelihood.

normal_loglikelihood = -normal_mle_fit.fun


# The Normal model estimates three parameters:
# alpha, beta, and sigma.

k_normal = 3


# Calculate AICc:
#
#     AIC  = 2k - 2 log(L)
#
#     AICc = AIC + 2k(k + 1) / (n - k - 1)

normal_AIC = (
    2 * k_normal
    - 2 * normal_loglikelihood
)

normal_AICc = (
    normal_AIC
    + 2 * k_normal * (k_normal + 1)
    / (n - k_normal - 1)
)


# ==================================================
# 5. Maximum likelihood under Student's t errors
# ==================================================

# Assume the regression errors follow a Student's t
# distribution with degrees of freedom nu and scale
# parameter sigma:
#
#     epsilon_i ~ t_nu(0, sigma).
#
# The parameters estimated are:
#
#     alpha, beta, sigma, and nu.
#
# We optimize log(sigma) and log(nu - 2) rather than
# sigma and nu directly. This guarantees:
#
#     sigma > 0
#     nu > 2
#
# The restriction nu > 2 ensures that the fitted
# Student's t distribution has finite variance.

def student_t_neg_loglikelihood(parameters):

    alpha, beta, log_sigma, log_nu_minus_2 = parameters

    # Transform the parameters back to their
    # original scale.

    sigma = np.exp(log_sigma)
    nu = 2 + np.exp(log_nu_minus_2)

    # Regression residuals.

    residuals = y - (alpha + beta * x)

    # Standardized residuals.

    z = residuals / sigma

    # Student's t log-likelihood.
    #
    # gammaln() computes log(Gamma(.)) and provides
    # better numerical stability than evaluating the
    # Gamma function directly.

    log_likelihood = np.sum(
        gammaln((nu + 1) / 2)
        - gammaln(nu / 2)
        - 0.5 * np.log(nu * np.pi)
        - np.log(sigma)
        - ((nu + 1) / 2)
        * np.log(1 + z ** 2 / nu)
    )

    # scipy.optimize.minimize() minimizes the objective,
    # so return the negative log-likelihood.

    return -log_likelihood


# ==================================================
# 6. Initial values for the Student's t MLE
# ==================================================

# Use the OLS estimates as starting values for alpha
# and beta, and the OLS residual standard deviation
# as the starting value for sigma.

alpha_start = alpha_ols
beta_start = beta_ols
sigma_start = sigma_ols


# Choose an initial value of 10 for the degrees of
# freedom. The optimizer will estimate nu from the data.

nu_start = 10


# Convert the starting values to the transformed scale
# used by the likelihood function.

initial_t_parameters = [
    alpha_start,
    beta_start,
    np.log(sigma_start),
    np.log(nu_start - 2)
]


# ==================================================
# 7. Estimate the Student's t model by MLE
# ==================================================

t_mle_fit = minimize(
    student_t_neg_loglikelihood,
    initial_t_parameters,
    method="BFGS"
)


# Transform the optimized parameters back to their
# original scale.

alpha_t = t_mle_fit.x[0]
beta_t = t_mle_fit.x[1]

sigma_t = np.exp(t_mle_fit.x[2])
nu_t = 2 + np.exp(t_mle_fit.x[3])


# Convert the minimized negative log-likelihood
# into the maximized log-likelihood.

t_loglikelihood = -t_mle_fit.fun


# The Student's t model estimates four parameters:
# alpha, beta, sigma, and nu.

k_t = 4


# Calculate AICc.

t_AIC = (
    2 * k_t
    - 2 * t_loglikelihood
)

t_AICc = (
    t_AIC
    + 2 * k_t * (k_t + 1)
    / (n - k_t - 1)
)


# ==================================================
# 8. Report model estimates
# ==================================================

# OLS reports coefficient estimates, their standard
# errors, and the estimated residual standard deviation.

print("\nOLS")
print("----------------")
print(f"alpha = {alpha_ols:.6f}")
print(f"beta  = {beta_ols:.6f}")
print(f"SE(alpha) = {se_alpha_ols:.6f}")
print(f"SE(beta)  = {se_beta_ols:.6f}")
print(f"sigma = {sigma_ols:.6f}")


# The Normal MLE reports alpha, beta, sigma, and AICc.

print("\nNormal MLE")
print("----------------")
print(f"alpha = {alpha_normal_mle:.6f}")
print(f"beta  = {beta_normal_mle:.6f}")
print(f"sigma = {sigma_normal_mle:.6f}")
print(f"AICc  = {normal_AICc:.6f}")


# The Student's t MLE additionally estimates the
# degrees of freedom nu.

print("\nStudent's t MLE")
print("----------------")
print(f"alpha = {alpha_t:.6f}")
print(f"beta  = {beta_t:.6f}")
print(f"sigma = {sigma_t:.6f}")
print(f"nu    = {nu_t:.6f}")
print(f"AICc  = {t_AICc:.6f}")


# ==================================================
# 9. Compare models using AICc
# ==================================================

# AICc balances goodness of fit against model
# complexity. The model with the smaller AICc
# is preferred.

if t_AICc < normal_AICc:
    preferred_model = "Student's t"
else:
    preferred_model = "Normal"


print("\nModel comparison")
print("----------------")
print(f"Normal AICc      = {normal_AICc:.6f}")
print(f"Student's t AICc = {t_AICc:.6f}")
print(f"Preferred model  = {preferred_model}")