"""
risk_lib_13.py

Reusable functions for Question 13:

1. Distribution fitting
2. Marginal distribution objects
3. Kendall correlation
4. Multivariate Student-t fitting
5. Gaussian copula log likelihood
6. Student-t copula log likelihood and fitting
7. Copula model selection
8. Lower tail dependence
9. Student-t copula simulation
10. Portfolio VaR and Expected Shortfall

Loss convention
---------------
Returns and P&L are used as inputs.
Negative values represent losses.

VaR and ES are reported as positive losses.
"""

import numpy as np
import pandas as pd

from scipy import stats
from scipy.optimize import minimize, minimize_scalar


# ==================================================================
# 0. Default Settings
# ==================================================================

DEFAULT_ALPHA = 0.05


# ==================================================================
# 1. Basic Distribution Fitting
# ==================================================================

def fit_normal(data):
    """
    Estimate the parameters of a normal distribution.

    Returns
    -------
    mean, standard_deviation
    """
    data = np.asarray(data, dtype=float)

    mean = data.mean()
    standard_deviation = data.std(ddof=1)

    return mean, standard_deviation


def fit_student_t(data):
    """
    Fit a location-scale Student-t distribution using
    maximum likelihood.

    Returns
    -------
    location, scale, degrees_freedom
    """
    data = np.asarray(data, dtype=float)

    def negative_log_likelihood(parameters):
        location, scale, degrees_freedom = parameters

        if scale <= 0 or degrees_freedom <= 2:
            return 1e10

        log_likelihood = stats.t.logpdf(
            data,
            df=degrees_freedom,
            loc=location,
            scale=scale
        ).sum()

        return -log_likelihood

    initial_guess = [
        data.mean(),
        data.std(ddof=1),
        6.0
    ]

    result = minimize(
        negative_log_likelihood,
        initial_guess,
        method="Nelder-Mead",
        options={
            "xatol": 1e-10,
            "fatol": 1e-10,
            "maxiter": 20_000
        }
    )

    if not result.success:
        raise RuntimeError(
            f"Student-t fitting failed: {result.message}"
        )

    location, scale, degrees_freedom = result.x

    return location, scale, degrees_freedom


# ==================================================================
# 2. Marginal Distribution Objects
# ==================================================================

class FittedMarginal:
    """
    Store a fitted marginal distribution for one asset.

    Supported distributions:
        - Normal
        - Student-t

    Methods:
        fit() : Fit a distribution to historical returns.
        cdf() : Convert observations into probabilities.
        ppf() : Convert probabilities into simulated returns.
    """

    def __init__(self, distribution, parameters):
        self.distribution = distribution
        self.parameters = parameters

    @classmethod
    def fit(cls, data, kind):
        """
        Fit the selected distribution to historical returns.

        Parameters
        ----------
        data : array-like
            Historical returns for one asset.
        kind : str
            "Normal" or "T".
        """
        kind = kind.strip().lower()

        if kind == "normal":

            mean, standard_deviation = fit_normal(data)

            return cls(
                stats.norm,
                {
                    "loc": mean,
                    "scale": standard_deviation
                }
            )

        elif kind in ("t", "student", "student-t"):

            location, scale, degrees_freedom = fit_student_t(data)

            return cls(
                stats.t,
                {
                    "df": degrees_freedom,
                    "loc": location,
                    "scale": scale
                }
            )

        else:
            raise ValueError(
                f"Unknown distribution type: {kind}"
            )

    def cdf(self, data):
        """Calculate cumulative probabilities."""

        return self.distribution.cdf(
            data,
            **self.parameters
        )

    def ppf(self, probabilities):
        """Convert probabilities into distribution quantiles."""

        return self.distribution.ppf(
            probabilities,
            **self.parameters
        )


# ==================================================================
# 3. Positive Semi-Definite Correlation Matrix
# ==================================================================

def near_psd(correlation, epsilon=1e-10):
    """
    Adjust a correlation matrix to be positive semi-definite.

    The adjusted matrix is normalized to have diagonal entries
    equal to one.
    """
    correlation = np.asarray(correlation, dtype=float)

    correlation = (
        correlation + correlation.T
    ) / 2

    eigenvalues, eigenvectors = np.linalg.eigh(correlation)

    if eigenvalues.min() >= epsilon:
        return correlation.copy()

    eigenvalues = np.maximum(eigenvalues, epsilon)

    scale = 1.0 / (
        (eigenvectors ** 2) @ eigenvalues
    )

    matrix_b = (
        np.sqrt(scale)[:, None]
        * eigenvectors
        * np.sqrt(eigenvalues)
    )

    adjusted = matrix_b @ matrix_b.T

    diagonal_scale = np.sqrt(np.diag(adjusted))

    adjusted = adjusted / np.outer(
        diagonal_scale,
        diagonal_scale
    )

    return adjusted


# ==================================================================
# 4. Kendall Correlation (Question 13.1)
# ==================================================================

def kendall_correlation(returns):
    """
    Estimate the Gaussian-equivalent correlation matrix
    from Kendall's tau.

    Relationship:
        rho = sin(pi * tau / 2)

    The resulting matrix is adjusted to be positive
    semi-definite.
    """
    returns = np.asarray(returns, dtype=float)

    if returns.ndim != 2:
        raise ValueError("returns must be a 2-D array.")

    n_assets = returns.shape[1]

    kendall_tau = np.eye(n_assets)

    for i in range(n_assets):
        for j in range(i + 1, n_assets):

            tau = stats.kendalltau(
                returns[:, i],
                returns[:, j]
            ).statistic

            kendall_tau[i, j] = tau
            kendall_tau[j, i] = tau

    correlation = np.sin(
        np.pi * kendall_tau / 2
    )

    return near_psd(correlation)


# ==================================================================
# 5. Multivariate Student-t Distribution Fit (Questions 13.2–13.4)
# ==================================================================

def fit_mvt_with_corr(returns, correlation):
    """
    Fit a multivariate Student-t distribution with a fixed
    correlation matrix.

    The sample means and standard deviations are used to
    estimate the location and covariance structure.

    Degrees of freedom are estimated by maximum likelihood.

    Returns
    -------
    mean : ndarray
    scale : ndarray
    degrees_freedom : float
    log_likelihood : float
    """
    returns = np.asarray(returns, dtype=float)
    correlation = np.asarray(correlation, dtype=float)

    if returns.ndim != 2:
        raise ValueError("returns must be a 2-D array.")

    n_observations, n_assets = returns.shape

    if correlation.shape != (n_assets, n_assets):
        raise ValueError(
            "Correlation matrix dimensions do not match returns."
        )

    mean = returns.mean(axis=0)
    standard_deviations = returns.std(axis=0, ddof=1)

    covariance = (
        np.outer(standard_deviations, standard_deviations)
        * correlation
    )

    def negative_log_likelihood(degrees_freedom):

        scale = (
            (degrees_freedom - 2)
            / degrees_freedom
            * covariance
        )

        log_likelihood = stats.multivariate_t.logpdf(
            returns,
            loc=mean,
            shape=scale,
            df=degrees_freedom
        ).sum()

        return -log_likelihood

    optimization = minimize_scalar(
        negative_log_likelihood,
        bounds=(2.001, 100.0),
        method="bounded",
        options={"xatol": 1e-8}
    )

    if not optimization.success:
        raise RuntimeError(
            "Multivariate t fitting failed."
        )

    degrees_freedom = optimization.x

    scale = (
        (degrees_freedom - 2)
        / degrees_freedom
        * covariance
    )

    log_likelihood = -optimization.fun

    return mean, scale, degrees_freedom, log_likelihood


# ==================================================================
# 6. Transform Marginal Returns into Uniform Variables
# ==================================================================

def to_uniforms(returns, marginals):
    """
    Transform each asset's historical returns into uniform
    probabilities using its fitted marginal distribution.
    """
    returns = np.asarray(returns, dtype=float)

    if returns.ndim != 2:
        raise ValueError("returns must be a 2-D array.")

    if returns.shape[1] != len(marginals):
        raise ValueError(
            "Number of marginals must match number of assets."
        )

    uniforms = np.column_stack([
        marginals[j].cdf(returns[:, j])
        for j in range(returns.shape[1])
    ])

    # Avoid infinite quantiles in subsequent transformations.
    return np.clip(uniforms, 1e-12, 1 - 1e-12)


# ==================================================================
# 7. Gaussian Copula Log Likelihood (Question 13.5)
# ==================================================================

def gaussian_copula_loglik(uniforms, correlation):
    """
    Calculate the Gaussian copula log likelihood.

    The likelihood is:
        log c(u) = log f_Z(z) - sum(log phi(z_i))

    where:
        z_i = Phi^(-1)(u_i)
    """
    uniforms = np.asarray(uniforms, dtype=float)
    correlation = near_psd(correlation)

    uniforms = np.clip(
        uniforms,
        1e-12,
        1 - 1e-12
    )

    normal_scores = stats.norm.ppf(uniforms)

    joint_log_density = stats.multivariate_normal.logpdf(
        normal_scores,
        mean=np.zeros(uniforms.shape[1]),
        cov=correlation,
        allow_singular=True
    )

    marginal_log_density = stats.norm.logpdf(
        normal_scores
    ).sum(axis=1)

    return float(
        np.sum(joint_log_density - marginal_log_density)
    )


# ==================================================================
# 8. Student-t Copula Log Likelihood
# ==================================================================

def t_copula_loglik(uniforms, correlation, degrees_freedom):
    """
    Calculate the Student-t copula log likelihood.

    The likelihood is:
        log c(u) = log f_t(z) - sum(log t_nu(z_i))

    where:
        z_i = t_nu^(-1)(u_i)
    """
    uniforms = np.asarray(uniforms, dtype=float)
    correlation = near_psd(correlation)

    uniforms = np.clip(
        uniforms,
        1e-12,
        1 - 1e-12
    )

    t_scores = stats.t.ppf(
        uniforms,
        df=degrees_freedom
    )

    joint_log_density = stats.multivariate_t.logpdf(
        t_scores,
        loc=np.zeros(uniforms.shape[1]),
        shape=correlation,
        df=degrees_freedom
    )

    marginal_log_density = stats.t.logpdf(
        t_scores,
        df=degrees_freedom
    ).sum(axis=1)

    return float(
        np.sum(joint_log_density - marginal_log_density)
    )


# ==================================================================
# 9. Fit Student-t Copula Degrees of Freedom (Question 13.6)
# ==================================================================

def fit_t_copula_nu(uniforms, correlation):
    """
    Estimate the degrees of freedom of a Student-t copula
    by maximizing its log likelihood.

    Returns
    -------
    degrees_freedom : float
    log_likelihood : float
    """
    def negative_log_likelihood(degrees_freedom):

        return -t_copula_loglik(
            uniforms,
            correlation,
            degrees_freedom
        )

    optimization = minimize_scalar(
        negative_log_likelihood,
        bounds=(1.01, 100.0),
        method="bounded",
        options={"xatol": 1e-8}
    )

    if not optimization.success:
        raise RuntimeError(
            "Student-t copula fitting failed."
        )

    degrees_freedom = optimization.x
    log_likelihood = -optimization.fun

    return degrees_freedom, log_likelihood


# ==================================================================
# 10. Model Selection Criteria (Question 13.7)
# ==================================================================

def aicc(log_likelihood, parameters, n_observations):
    """
    Calculate the corrected Akaike Information Criterion.

    AICc = -2 * LL + 2K + 2K(K+1)/(n-K-1)
    """
    if n_observations <= parameters + 1:
        return np.inf

    return (
        -2 * log_likelihood
        + 2 * parameters
        + (
            2 * parameters * (parameters + 1)
            / (n_observations - parameters - 1)
        )
    )


def bic(log_likelihood, parameters, n_observations):
    """
    Calculate the Bayesian Information Criterion.

    BIC = -2 * LL + K * log(n)
    """
    return (
        -2 * log_likelihood
        + parameters * np.log(n_observations)
    )


# ==================================================================
# 11. Student-t Copula Lower Tail Dependence (Question 13.8)
# ==================================================================

def t_tail_dependence(correlation, degrees_freedom):
    """
    Calculate the lower tail dependence coefficient
    of a bivariate Student-t copula.

    Formula:
        lambda_L =
        2 * T_(nu+1)(
            -sqrt((nu+1)*(1-rho)/(1+rho))
        )
    """
    correlation = np.asarray(correlation, dtype=float)

    if np.any(np.abs(correlation) >= 1):
        raise ValueError(
            "Correlation must be strictly between -1 and 1."
        )

    if degrees_freedom <= 0:
        raise ValueError(
            "Degrees of freedom must be positive."
        )

    argument = -np.sqrt(
        (degrees_freedom + 1)
        * (1 - correlation)
        / (1 + correlation)
    )

    return 2 * stats.t.cdf(
        argument,
        df=degrees_freedom + 1
    )


# ==================================================================
# 12. Student-t Copula Simulation (Question 13.9)
# ==================================================================

def t_copula_simulate(
    marginals,
    correlation,
    degrees_freedom,
    n=1_000_000,
    seed=42
):
    """
    Simulate asset returns using a Student-t copula.

    Procedure
    ---------
    1. Generate correlated standard normal variables.
    2. Divide by a common chi-square scaling factor.
    3. Transform the variables into uniform probabilities.
    4. Apply each fitted marginal inverse CDF.

    Returns
    -------
    simulated_returns : ndarray
        Shape (n simulations, number of assets).
    """
    correlation = near_psd(correlation)

    n_assets = len(marginals)

    if correlation.shape != (n_assets, n_assets):
        raise ValueError(
            "Correlation matrix dimensions do not match marginals."
        )

    if degrees_freedom <= 0:
        raise ValueError(
            "Degrees of freedom must be positive."
        )

    if n <= 0:
        raise ValueError("n must be positive.")

    rng = np.random.default_rng(seed)

    # Step 1: Generate correlated standard normal variables.
    normal_simulations = rng.multivariate_normal(
        mean=np.zeros(n_assets),
        cov=correlation,
        size=n
    )

    # Step 2: Generate a common chi-square scaling factor.
    chi_square_values = rng.chisquare(
        df=degrees_freedom,
        size=n
    )

    t_simulations = (
        normal_simulations
        / np.sqrt(
            chi_square_values[:, None] / degrees_freedom
        )
    )

    # Step 3: Transform t variables into uniform probabilities.
    uniform_simulations = stats.t.cdf(
        t_simulations,
        df=degrees_freedom
    )

    uniform_simulations = np.clip(
        uniform_simulations,
        1e-12,
        1 - 1e-12
    )

    # Step 4: Apply the fitted marginal inverse CDFs.
    simulated_returns = np.column_stack([
        marginals[j].ppf(uniform_simulations[:, j])
        for j in range(n_assets)
    ])

    return simulated_returns


# ==================================================================
# 13. Portfolio VaR and Expected Shortfall (Question 13.9)
# ==================================================================

def portfolio_var_es(
    pnl_by_asset,
    names,
    values,
    alpha=DEFAULT_ALPHA
):
    """
    Calculate VaR and ES for each asset and the total portfolio.

    Parameters
    ----------
    pnl_by_asset : ndarray
        Simulated dollar P&L, shape (simulations, assets).
    names : list
        Asset names.
    values : array-like
        Current dollar position values.
    alpha : float
        Lower-tail probability. Default is 0.05.

    Returns
    -------
    pandas.DataFrame
        Asset-level and total portfolio risk results.

    VaR and ES are reported as positive losses.
    Percentage measures are relative to position values.
    """
    pnl_by_asset = np.asarray(pnl_by_asset, dtype=float)
    values = np.asarray(values, dtype=float)

    if pnl_by_asset.ndim != 2:
        raise ValueError(
            "pnl_by_asset must be a 2-D array."
        )

    if pnl_by_asset.shape[1] != len(names):
        raise ValueError(
            "Number of asset names must match P&L columns."
        )

    if len(values) != len(names):
        raise ValueError(
            "Number of position values must match assets."
        )

    if not 0 < alpha < 1:
        raise ValueError(
            "alpha must be between 0 and 1."
        )

    if np.any(values <= 0):
        raise ValueError(
            "Position values must be positive."
        )

    if not np.all(np.isfinite(pnl_by_asset)):
        raise ValueError(
            "P&L contains missing or non-finite values."
        )

    # Individual asset P&L and total portfolio P&L.
    pnl_columns = [
        pnl_by_asset[:, j]
        for j in range(pnl_by_asset.shape[1])
    ]

    total_portfolio_pnl = pnl_by_asset.sum(axis=1)
    pnl_columns.append(total_portfolio_pnl)

    asset_names = list(names) + ["Total"]

    position_values = list(values) + [
        float(values.sum())
    ]

    confidence = int(round((1 - alpha) * 100))

    results = []

    for name, pnl, position_value in zip(
        asset_names,
        pnl_columns,
        position_values
    ):

        quantile = np.quantile(pnl, alpha)

        tail_observations = pnl[pnl <= quantile]

        var = -quantile
        es = -tail_observations.mean()

        results.append({
            "Stock": name,
            f"VaR{confidence}": var,
            f"ES{confidence}": es,
            f"VaR{confidence}_Pct": var / position_value,
            f"ES{confidence}_Pct": es / position_value
        })

    return pd.DataFrame(results)
