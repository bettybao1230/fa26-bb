import numpy as np
import pandas as pd

from scipy.linalg import cholesky, solve_triangular
from scipy.optimize import minimize_scalar
from scipy.special import gammaln
from scipy.stats import kendalltau, norm, t


# ============================================================
# 1. Observation Counts and Correlation
# ============================================================

def pairwise_observation_counts(df):
    """Count jointly observed values for each pair of columns."""
    observed = df.notna().astype(int)
    return observed.T @ observed


def complete_case_correlation(df):
    """Calculate correlations using complete observations only."""
    return df.dropna().corr()


def pairwise_correlation(df):
    """Calculate correlations using all available paired observations."""
    return df.corr()


def eigenvalues_of_correlation(corr):
    """Return the eigenvalues of a symmetric matrix."""
    matrix = np.asarray(corr, dtype=float)
    matrix = (matrix + matrix.T) / 2
    return np.linalg.eigvalsh(matrix)


def is_psd(matrix, tolerance=1e-10):
    """Check whether a symmetric matrix is positive semidefinite."""
    eigenvalues = eigenvalues_of_correlation(matrix)
    return eigenvalues.min() >= -tolerance


# ============================================================
# 2. Correlation Matrix Repair
# ============================================================

def rebonato_jackel(corr, tolerance=1e-12):
    """Repair a correlation matrix by clipping negative eigenvalues."""
    matrix = np.asarray(corr, dtype=float)
    matrix = (matrix + matrix.T) / 2

    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    eigenvalues = np.maximum(eigenvalues, 0.0)

    repaired = (eigenvectors * eigenvalues) @ eigenvectors.T

    # Normalize to unit diagonal.
    diagonal = np.sqrt(np.maximum(np.diag(repaired), tolerance))
    repaired /= np.outer(diagonal, diagonal)

    repaired = (repaired + repaired.T) / 2
    np.fill_diagonal(repaired, 1.0)

    return pd.DataFrame(
        repaired,
        index=corr.index,
        columns=corr.columns,
    )


def higham_nearest_correlation(corr, max_iterations=100, tolerance=1e-10):
    """Approximate the nearest correlation matrix using alternating projections."""
    matrix = np.asarray(corr, dtype=float)
    matrix = (matrix + matrix.T) / 2

    x = matrix.copy()
    correction = np.zeros_like(matrix)

    for _ in range(max_iterations):
        # Project onto the positive semidefinite cone.
        residual = x - correction
        residual = (residual + residual.T) / 2

        eigenvalues, eigenvectors = np.linalg.eigh(residual)
        positive_eigenvalues = np.maximum(eigenvalues, 0.0)

        x_psd = (eigenvectors * positive_eigenvalues) @ eigenvectors.T
        correction = x_psd - residual

        # Enforce the unit-diagonal constraint.
        x_new = x_psd.copy()
        np.fill_diagonal(x_new, 1.0)
        x_new = (x_new + x_new.T) / 2

        difference = np.linalg.norm(x_new - x, ord="fro")
        x = x_new

        if difference < tolerance:
            break

    return pd.DataFrame(
        x,
        index=corr.index,
        columns=corr.columns,
    )


# ============================================================
# 3. Covariance and Portfolio Variance
# ============================================================

def covariance_from_correlation(corr, std):
    """Construct a covariance matrix using Sigma = D R D."""
    correlation = np.asarray(corr, dtype=float)
    std = np.asarray(std, dtype=float)

    diagonal = np.diag(std)
    covariance = diagonal @ correlation @ diagonal

    return pd.DataFrame(
        covariance,
        index=corr.index,
        columns=corr.columns,
    )


def portfolio_variance(covariance, weights):
    """Calculate portfolio variance: w' Sigma w."""
    covariance = np.asarray(covariance, dtype=float)
    weights = np.asarray(weights, dtype=float)

    return float(weights.T @ covariance @ weights)


# ============================================================
# 4. Returns and Distribution Moments
# ============================================================

def arithmetic_returns(prices):
    """Calculate simple returns: r_t = P_t / P_(t-1) - 1."""
    prices = np.asarray(prices, dtype=float)

    if np.any(prices <= 0):
        raise ValueError("Prices must be positive.")

    return prices[1:] / prices[:-1] - 1.0


def four_moments(x):
    """Return mean, central-moment variance, skewness, and kurtosis."""
    x = np.asarray(x, dtype=float)
    mean = np.mean(x)

    centered = x - mean
    m2 = np.mean(centered**2)
    m3 = np.mean(centered**3)
    m4 = np.mean(centered**4)

    skewness = m3 / m2**1.5
    kurtosis = m4 / m2**2

    return mean, m2, skewness, kurtosis


# ============================================================
# 5. Value at Risk (VaR)
# ============================================================

def normal_var(returns, portfolio_value, alpha):
    """Calculate Normal VaR from the sample mean and volatility."""
    mean_return = np.mean(returns)
    volatility = np.std(returns, ddof=1)

    quantile = norm.ppf(alpha, loc=mean_return, scale=volatility)
    return max(-portfolio_value * quantile, 0.0)


def historical_var(returns, portfolio_value, alpha):
    """Calculate historical VaR from the empirical return quantile."""
    quantile = np.quantile(returns, alpha)
    return max(-portfolio_value * quantile, 0.0)


def fit_student_t(returns):
    """Fit a Student-t distribution and return (df, loc, scale)."""
    return t.fit(returns)


def student_t_var(returns, portfolio_value, alpha):
    """Calculate VaR using a fitted Student-t distribution."""
    df, loc, scale = fit_student_t(returns)
    quantile = t.ppf(alpha, df=df, loc=loc, scale=scale)

    return max(-portfolio_value * quantile, 0.0)


# ============================================================
# 6. Exponentially Weighted Moving Average (EWMA)
# ============================================================

def ewma_volatility(returns, lam):
    """Estimate volatility using exponentially weighted squared returns."""
    variance = np.var(returns, ddof=1)

    for value in returns:
        variance = lam * variance + (1 - lam) * value**2

    return np.sqrt(variance)


def ewma_normal_var(returns, portfolio_value, alpha, lam):
    """Calculate Normal VaR using EWMA volatility."""
    mean_return = np.mean(returns)
    volatility = ewma_volatility(returns, lam)

    quantile = norm.ppf(alpha)
    var = -portfolio_value * (mean_return + volatility * quantile)

    return max(var, 0.0)


def ewma_student_t_var(returns, portfolio_value, alpha, lam):
    """Calculate Student-t VaR using fitted degrees of freedom and EWMA volatility."""
    df, loc, _ = fit_student_t(returns)
    volatility = ewma_volatility(returns, lam)

    quantile = t.ppf(alpha, df=df)
    var = -portfolio_value * (loc + volatility * quantile)

    return max(var, 0.0)


# ============================================================
# 7. Portfolio P&L and Expected Shortfall (ES)
# ============================================================

def compute_pnl(returns_a, returns_b, investment_a, investment_b):
    """Calculate portfolio P&L from two return series and positions."""
    returns_a = np.asarray(returns_a, dtype=float)
    returns_b = np.asarray(returns_b, dtype=float)

    return investment_a * returns_a + investment_b * returns_b


def historical_es(pnl, alpha):
    """Calculate historical ES as the average loss in the worst alpha tail."""
    pnl = np.asarray(pnl, dtype=float)
    n_tail = max(1, int(np.ceil(alpha * len(pnl))))

    return max(-np.mean(np.sort(pnl)[:n_tail]), 0.0)


# ============================================================
# 8. Marginal Distribution Fitting and Model Selection
# ============================================================

def fit_distribution(x, distribution):
    """Fit a SciPy distribution to finite observations using maximum likelihood."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]

    if len(x) == 0:
        raise ValueError("x must contain finite observations.")

    return distribution.fit(x)


def calculate_log_likelihood(x, distribution, params):
    """Calculate the log-likelihood for specified distribution parameters."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]

    logpdf = distribution.logpdf(x, *params)

    if not np.all(np.isfinite(logpdf)):
        return -np.inf

    return float(np.sum(logpdf))


def calculate_aicc(log_likelihood, n, k):
    """Calculate AICc with its finite-sample correction."""
    if n <= k + 1:
        raise ValueError("AICc requires n > k + 1.")

    if not np.isfinite(log_likelihood):
        return np.inf

    aic = 2 * k - 2 * log_likelihood
    correction = 2 * k * (k + 1) / (n - k - 1)

    return float(aic + correction)


def information_criteria(loglik, n, k):
    """Calculate AICc and BIC from log-likelihood, sample size, and parameter count."""
    if n <= k + 1:
        raise ValueError("AICc requires n > k + 1.")

    aicc = (
        -2 * loglik
        + 2 * k
        + 2 * k * (k + 1) / (n - k - 1)
    )
    bic = -2 * loglik + k * np.log(n)

    return {"AICc": float(aicc), "BIC": float(bic)}


# ============================================================
# 9. Copula Estimation
# ============================================================

def transform_to_uniform(data, marginal_specs):
    """Transform observations to uniform values using fitted marginal CDFs."""
    uniforms = np.column_stack([
        distribution.cdf(
            data[column].to_numpy(dtype=float),
            *params,
        )
        for column, (distribution, params) in marginal_specs.items()
    ])

    return np.clip(uniforms, 1e-10, 1 - 1e-10)


def kendall_tau_to_correlation(U):
    """Estimate copula correlations using rho = sin(pi * tau / 2)."""
    U = np.asarray(U, dtype=float)
    n_assets = U.shape[1]
    correlation = np.eye(n_assets)

    for i in range(n_assets):
        for j in range(i + 1, n_assets):
            tau_hat = kendalltau(U[:, i], U[:, j]).statistic
            correlation[i, j] = correlation[j, i] = np.sin(
                np.pi * tau_hat / 2
            )

    if np.linalg.eigvalsh(correlation).min() <= 0:
        raise ValueError(
            "Estimated correlation matrix is not positive definite. "
            "Apply a nearest-correlation-matrix adjustment."
        )

    return correlation


def gaussian_copula_loglik(U, R):
    """Calculate the Gaussian copula log-likelihood."""
    U = np.asarray(U, dtype=float)
    Z = norm.ppf(U)

    L = cholesky(R, lower=True)
    Y = solve_triangular(L, Z.T, lower=True)

    logdet_R = 2 * np.log(np.diag(L)).sum()

    log_density = (
        -0.5 * logdet_R
        - 0.5 * np.sum(Y**2)
        + 0.5 * np.sum(Z**2)
    )

    return float(log_density)


def t_copula_loglik(U, R, nu):
    """Calculate the Student-t copula log-likelihood."""
    U = np.asarray(U, dtype=float)
    Q = t.ppf(U, df=nu)

    L = cholesky(R, lower=True)
    Y = solve_triangular(L, Q.T, lower=True).T

    quadratic = np.sum(Y**2, axis=1)
    n_assets = Q.shape[1]
    logdet_R = 2 * np.log(np.diag(L)).sum()

    log_joint = (
        gammaln((nu + n_assets) / 2)
        - gammaln(nu / 2)
        - (n_assets / 2) * np.log(nu * np.pi)
        - 0.5 * logdet_R
        - ((nu + n_assets) / 2) * np.log1p(quadratic / nu)
    )

    log_marginal = np.sum(t.logpdf(Q, df=nu), axis=1)

    return float(np.sum(log_joint - log_marginal))


def fit_t_copula_df(U, R, bounds=(2.01, 200)):
    """Estimate Student-t copula degrees of freedom while holding R fixed."""
    result = minimize_scalar(
        lambda log_nu: -t_copula_loglik(U, R, np.exp(log_nu)),
        bounds=(np.log(bounds[0]), np.log(bounds[1])),
        method="bounded",
    )

    if not result.success:
        raise RuntimeError("t-copula optimization failed.")

    nu_hat = float(np.exp(result.x))
    return nu_hat, t_copula_loglik(U, R, nu_hat)


# ============================================================
# 10. Copula Simulation
# ============================================================

def simulate_gaussian_copula(n, R, seed=123):
    """Simulate uniform observations from a Gaussian copula."""
    rng = np.random.default_rng(seed)
    R = np.asarray(R, dtype=float)

    Z = rng.multivariate_normal(
        mean=np.zeros(R.shape[0]),
        cov=R,
        size=n,
    )

    return np.clip(norm.cdf(Z), 1e-10, 1 - 1e-10)


def simulate_t_copula(n, R, nu, seed=123):
    """Simulate uniform observations from a Student-t copula."""
    rng = np.random.default_rng(seed)
    R = np.asarray(R, dtype=float)
    n_assets = R.shape[0]

    Z = rng.multivariate_normal(
        mean=np.zeros(n_assets),
        cov=R,
        size=n,
    )

    chi2 = rng.chisquare(df=nu, size=n)
    t_samples = Z / np.sqrt(chi2[:, None] / nu)

    return np.clip(t.cdf(t_samples, df=nu), 1e-10, 1 - 1e-10)


def transform_uniform_to_margins(U, marginal_specs):
    """Transform uniform samples into returns using fitted marginal PPFs."""
    U = np.asarray(U, dtype=float)

    return np.column_stack([
        distribution.ppf(U[:, j], *params)
        for j, (distribution, params) in enumerate(marginal_specs.values())
    ])


# ============================================================
# 11. Joint-Tail Analysis
# ============================================================

def joint_tail_days(data, tail, target_days, assets=None):
    """Estimate joint lower- and upper-tail event counts per target period."""
    data = np.asarray(data, dtype=float)

    if data.ndim != 2 or data.shape[0] == 0 or data.shape[1] < 2:
        raise ValueError("data must be a non-empty 2D array with at least two assets.")
    if not 0 < tail < 0.5:
        raise ValueError("tail must be between 0 and 0.5.")
    if target_days <= 0:
        raise ValueError("target_days must be positive.")
    if not np.all(np.isfinite(data)):
        raise ValueError("data must contain only finite values.")

    n_obs, n_assets = data.shape

    if assets is None:
        assets = [f"X{i + 1}" for i in range(n_assets)]
    if len(assets) != n_assets:
        raise ValueError("Number of asset names must match the data columns.")

    lower = np.quantile(data, tail, axis=0)
    upper = np.quantile(data, 1 - tail, axis=0)

    results = []

    for i in range(n_assets):
        for j in range(i + 1, n_assets):
            lower_count = np.sum(
                (data[:, i] <= lower[i])
                & (data[:, j] <= lower[j])
            )
            upper_count = np.sum(
                (data[:, i] >= upper[i])
                & (data[:, j] >= upper[j])
            )

            results.append({
                "Pair": f"{assets[i]}-{assets[j]}",
                "Joint lower-tail days": lower_count / n_obs * target_days,
                "Joint upper-tail days": upper_count / n_obs * target_days,
            })

    return pd.DataFrame(results)


def copula_daily_loglik_difference(U, R, nu):
    """Return daily log-likelihood differences: t copula minus Gaussian copula."""
    U = np.asarray(U, dtype=float)
    Z = norm.ppf(U)
    Q = t.ppf(U, df=nu)

    L = cholesky(R, lower=True)
    logdet_R = 2 * np.log(np.diag(L)).sum()
    n_assets = U.shape[1]

    # Gaussian copula log-likelihood for each observation.
    Y_gaussian = solve_triangular(L, Z.T, lower=True).T

    gaussian_daily = (
        -0.5 * logdet_R
        - 0.5 * np.sum(Y_gaussian**2, axis=1)
        + 0.5 * np.sum(Z**2, axis=1)
    )

    # Student-t copula log-likelihood for each observation.
    Y_t = solve_triangular(L, Q.T, lower=True).T
    quadratic = np.sum(Y_t**2, axis=1)

    log_joint = (
        gammaln((nu + n_assets) / 2)
        - gammaln(nu / 2)
        - (n_assets / 2) * np.log(nu * np.pi)
        - 0.5 * logdet_R
        - ((nu + n_assets) / 2) * np.log1p(quadratic / nu)
    )

    log_marginal = np.sum(t.logpdf(Q, df=nu), axis=1)
    t_daily = log_joint - log_marginal

    return t_daily - gaussian_daily


# ============================================================
# 12. Market Model and Simulation
# ============================================================

def fit_market_model(stock_returns, market_returns):
    """Fit r_i = alpha + beta * r_market + residual."""
    stock_returns = np.asarray(stock_returns, dtype=float)
    market_returns = np.asarray(market_returns, dtype=float)

    if stock_returns.ndim != 1 or market_returns.ndim != 1:
        raise ValueError("Returns must be one-dimensional.")
    if len(stock_returns) != len(market_returns):
        raise ValueError("Stock and market returns must have equal lengths.")
    if len(stock_returns) < 2:
        raise ValueError("At least two paired observations are required.")

    X = np.column_stack([
        np.ones(len(market_returns)),
        market_returns,
    ])

    alpha, beta = np.linalg.lstsq(X, stock_returns, rcond=None)[0]
    residuals = stock_returns - (alpha + beta * market_returns)

    return float(alpha), float(beta), residuals


def simulate_market_model(
    market_sd,
    beta_a,
    beta_b,
    residual_covariance,
    n_sim,
    seed,
):
    """Simulate two stock returns using market and correlated residual shocks."""
    rng = np.random.default_rng(seed)

    market = rng.normal(0.0, market_sd, size=n_sim)
    residuals = rng.multivariate_normal(
        mean=np.zeros(2),
        cov=np.asarray(residual_covariance, dtype=float),
        size=n_sim,
    )

    returns_a = beta_a * market + residuals[:, 0]
    returns_b = beta_b * market + residuals[:, 1]

    return returns_a, returns_b


def simulated_portfolio_var(
    returns_a,
    returns_b,
    position_a,
    position_b,
    confidence,
):
    """Calculate simulated portfolio VaR at the specified confidence level."""
    pnl = (
        position_a * np.asarray(returns_a)
        + position_b * np.asarray(returns_b)
    )

    return max(-float(np.quantile(pnl, 1 - confidence)), 0.0)


def delta_normal_var(covariance, weights, confidence):
    """Calculate delta-normal VaR: z * sqrt(w' Sigma w)."""
    variance = max(portfolio_variance(covariance, weights), 0.0)

    return float(norm.ppf(confidence) * np.sqrt(variance))