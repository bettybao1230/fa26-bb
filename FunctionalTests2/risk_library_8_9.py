"""
risklib.py

Reusable functions for:
1. Distribution fitting
2. Value at Risk (VaR)
3. Expected Shortfall (ES)
4. Gaussian copula simulation
5. Portfolio-level risk measurement

Loss convention
---------------
Returns and P&L are used as inputs. Negative values represent losses.

All VaR and ES results are reported as positive losses.

Absolute       = -quantile
Diff from Mean = sample_mean - quantile

For ES, the quantile is replaced by the average of observations
in the lower alpha tail.
"""
import numpy as np
import pandas as pd

from scipy import stats
from scipy.optimize import minimize


# ==================================================================
# 0. Default Settings
# ==================================================================
DEFAULT_ALPHA = 0.05


# ==================================================================
# 1. Distribution Fitting
# ==================================================================
def fit_normal(data):
    """
    Estimate the parameters of a normal distribution.

    Parameters
    ----------
    data : array-like
        Historical observations.

    Returns
    -------
    mean : float
        Sample mean.
    std_dev : float
        Sample standard deviation (ddof=1).
    """
    data = np.asarray(data, dtype=float)

    mean = data.mean()
    std_dev = data.std(ddof=1)

    return mean, std_dev


def fit_student_t(data):
    """
    Fit a location-scale Student-t distribution using maximum likelihood.

    Parameters
    ----------
    data : array-like
        Historical observations.

    Returns
    -------
    location : float
        Estimated location parameter.
    scale : float
        Estimated scale parameter.
    degrees_freedom : float
        Estimated degrees of freedom.
    """
    data = np.asarray(data, dtype=float)

    def negative_log_likelihood(parameters):
        location, scale, degrees_freedom = parameters

        # Require positive scale and degrees of freedom greater than 2
        if scale <= 0 or degrees_freedom <= 2:
            return 1e10

        log_likelihood = stats.t.logpdf(
            data,
            df=degrees_freedom,
            loc=location,
            scale=scale
        ).sum()

        return -log_likelihood

    # Initial parameter estimates
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
# 2. Value at Risk (VaR)
# ==================================================================
def var_normal(mean, std_dev, alpha=DEFAULT_ALPHA):
    """
    Calculate VaR assuming a normal distribution.

    Returns
    -------
    tuple
        (Absolute VaR, VaR Diff from Mean)
    """

    quantile = stats.norm.ppf(
        alpha,
        loc=mean,
        scale=std_dev
    )

    absolute_var = -quantile
    var_diff_from_mean = mean - quantile

    return absolute_var, var_diff_from_mean


def var_student_t(
    location,
    scale,
    degrees_freedom,
    alpha=DEFAULT_ALPHA
):
    """
    Calculate VaR assuming a Student-t distribution.

    Returns
    -------
    tuple
        (Absolute VaR, VaR Diff from Mean/Location)
    """
    quantile = stats.t.ppf(
        alpha,
        df=degrees_freedom,
        loc=location,
        scale=scale
    )

    absolute_var = -quantile
    var_diff_from_mean = location - quantile

    return absolute_var, var_diff_from_mean


def var_empirical(data, alpha=DEFAULT_ALPHA):
    """
    Calculate empirical VaR directly from observations.

    Returns
    -------
    tuple
        (Absolute VaR, VaR Diff from Mean)
    """
    data = np.asarray(data, dtype=float)

    sample_mean = data.mean()
    quantile = np.quantile(data, alpha)

    absolute_var = -quantile
    var_diff_from_mean = sample_mean - quantile

    return absolute_var, var_diff_from_mean


# ==================================================================
# 3. Expected Shortfall (ES)
# ==================================================================
def es_normal(mean, std_dev, alpha=DEFAULT_ALPHA):
    """
    Calculate Expected Shortfall assuming a normal distribution.

    Uses the closed-form normal distribution formula.

    Returns
    -------
    tuple
        (Absolute ES, ES Diff from Mean)
    """
    standardized_quantile = stats.norm.ppf(alpha)

    expected_tail_distance = (
        std_dev
        * stats.norm.pdf(standardized_quantile)
        / alpha
    )

    absolute_es = expected_tail_distance - mean
    es_diff_from_mean = expected_tail_distance

    return absolute_es, es_diff_from_mean


def es_student_t(
    location,
    scale,
    degrees_freedom,
    alpha=DEFAULT_ALPHA
):
    """
    Calculate Expected Shortfall assuming a Student-t distribution.

    Uses the closed-form Student-t lower-tail expectation formula.

    Returns
    -------
    tuple
        (Absolute ES, ES Diff from Mean/Location)
    """
    standardized_quantile = stats.t.ppf(
        alpha,
        df=degrees_freedom
    )

    standardized_tail_expectation = (
        stats.t.pdf(
            standardized_quantile,
            df=degrees_freedom
        )
        / alpha
        * (degrees_freedom + standardized_quantile**2)
        / (degrees_freedom - 1)
    )

    expected_tail_distance = (
        scale * standardized_tail_expectation
    )

    absolute_es = expected_tail_distance - location
    es_diff_from_mean = expected_tail_distance

    return absolute_es, es_diff_from_mean


def es_empirical(data, alpha=DEFAULT_ALPHA):
    """
    Calculate empirical Expected Shortfall from observations.

    ES is the average loss among observations at or below
    the lower alpha quantile.

    Returns
    -------
    tuple
        (Absolute ES, ES Diff from Mean)
    """
    data = np.asarray(data, dtype=float)

    sample_mean = data.mean()
    quantile = np.quantile(data, alpha)

    tail_observations = data[data <= quantile]
    tail_mean = tail_observations.mean()

    absolute_es = -tail_mean
    es_diff_from_mean = sample_mean - tail_mean

    return absolute_es, es_diff_from_mean


# ==================================================================
# 4. Simulation-Based VaR and ES
# ==================================================================
def simulate_student_t(
    location,
    scale,
    degrees_freedom,
    n=1_000_000,
    seed=42
):
    """
    Generate random observations from a fitted Student-t distribution.
    """
    rng = np.random.default_rng(seed)

    simulated_data = stats.t.rvs(
        df=degrees_freedom,
        loc=location,
        scale=scale,
        size=n,
        random_state=rng
    )

    return simulated_data


def var_es_from_simulation(data, alpha=DEFAULT_ALPHA):
    """
    Calculate both VaR and ES from simulated observations.

    Returns
    -------
    tuple
        (VaR result, ES result)

        Each result contains:
        (Absolute, Diff from Mean)
    """
    var_result = var_empirical(data, alpha)
    es_result = es_empirical(data, alpha)

    return var_result, es_result



# ==================================================================
# 5. Marginal Distribution Objects
# ==================================================================
class FittedMarginal:
    """
    Store a fitted marginal distribution for one stock.

    Supported distributions:
        - Normal
        - Student-t

    Main methods:
        fit() : Fit a distribution to historical returns.
        cdf() : Convert returns into cumulative probabilities.
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
            Historical returns for one stock.
        kind : str
            "Normal" or "T".

        Returns
        -------
        FittedMarginal
            Fitted marginal distribution.
        """
        kind = kind.strip().lower()

        if kind == "normal":

            mean, std_dev = fit_normal(data)

            return cls(
                stats.norm,
                {
                    "loc": mean,
                    "scale": std_dev
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
# 6. Gaussian Copula Simulation
# ==================================================================
def gaussian_copula_simulate(
    returns,
    kinds,
    n=100_000,
    seed=42
):
    """
    Generate correlated stock returns using a Gaussian copula.

    Parameters
    ----------
    returns : ndarray
        Historical returns with shape (observations, stocks).
    kinds : list
        Distribution type for each stock.
        Example: ["Normal", "T", "Normal"].
    n : int
        Number of simulations.
    seed : int
        Random seed.

    Returns
    -------
    sim_returns : ndarray
        Simulated returns with shape (n, stocks).
    marginals : list
        Fitted marginal distribution objects.

    Procedure
    ---------
    1. Fit a marginal distribution for each stock.
    2. Transform historical returns into uniform variables.
    3. Estimate the dependence correlation matrix.
    4. Generate correlated standard normal variables.
    5. Transform normal variables into uniform variables.
    6. Transform uniform variables into simulated stock returns.
    """
    returns = np.asarray(returns, dtype=float)

    # Validate input dimensions
    if returns.ndim != 2:
        raise ValueError("returns must be a 2-D array.")

    number_of_stocks = returns.shape[1]

    if number_of_stocks != len(kinds):
        raise ValueError(
            "Number of distributions must match number of stocks."
        )

    if number_of_stocks < 2:
        raise ValueError(
            "At least two stocks are required."
        )

    if not np.all(np.isfinite(returns)):
        raise ValueError(
            "Historical returns contain missing or non-finite values."
        )

    # Step 1: Fit marginal distributions
    marginals = [
        FittedMarginal.fit(
            returns[:, j],
            kinds[j]
        )
        for j in range(number_of_stocks)
    ]

    # Step 2: Transform historical returns into uniform variables
    uniforms = np.column_stack([
        marginals[j].cdf(returns[:, j])
        for j in range(number_of_stocks)
    ])

    # Step 3: Estimate Spearman rank correlation
    correlation = np.asarray(
        stats.spearmanr(uniforms).statistic
    )

    # Handle the two-stock case, where scipy may return a scalar
    if correlation.ndim == 0:
        correlation = np.array([
            [1.0, float(correlation)],
            [float(correlation), 1.0]
        ])

    # Step 4: Generate correlated standard normal variables
    rng = np.random.default_rng(seed)

    normal_simulations = rng.multivariate_normal(
        mean=np.zeros(number_of_stocks),
        cov=correlation,
        size=n
    )

    # Step 5: Transform normal variables into uniform variables
    uniform_simulations = stats.norm.cdf(
        normal_simulations
    )

    # Step 6: Transform uniforms into stock returns
    simulated_returns = np.column_stack([
        marginals[j].ppf(uniform_simulations[:, j])
        for j in range(number_of_stocks)
    ])

    return simulated_returns, marginals


# ==================================================================
# 7. Portfolio VaR and Expected Shortfall
# ==================================================================
def portfolio_var_es(
    pnl_by_asset,
    names,
    values,
    alpha=DEFAULT_ALPHA
):
    """
    Calculate VaR and ES for each stock and the total portfolio.

    Parameters
    ----------
    pnl_by_asset : ndarray
        Simulated dollar P&L, shape (simulations, stocks).
    names : list
        Stock names.
    values : array-like
        Current dollar value of each stock position.
    alpha : float
        Lower-tail probability. Default is 0.05.

    Returns
    -------
    pandas.DataFrame
        One row for each stock and one row for the total portfolio.

        Columns:
            Stock
            VaR95
            ES95
            VaR95_Pct
            ES95_Pct

    Notes
    -----
    VaR and ES are absolute losses measured from zero.
    Percentage measures are calculated relative to position values.
    """
    pnl_by_asset = np.asarray(pnl_by_asset, dtype=float)
    values = np.asarray(values, dtype=float)

    # Validate inputs
    if pnl_by_asset.ndim != 2:
        raise ValueError(
            "pnl_by_asset must be a 2-D array."
        )

    if pnl_by_asset.shape[1] != len(names):
        raise ValueError(
            "Number of stock names must match P&L columns."
        )

    if len(values) != len(names):
        raise ValueError(
            "Number of position values must match stocks."
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

    # Include individual stocks and total portfolio
    pnl_columns = list(pnl_by_asset.T)

    total_portfolio_pnl = pnl_by_asset.sum(axis=1)
    pnl_columns.append(total_portfolio_pnl)

    stock_names = list(names) + ["Total"]

    position_values = list(values) + [
        float(values.sum())
    ]

    # Determine confidence level for column names
    confidence = int(round((1 - alpha) * 100))

    results = []

    # Calculate risk measures for each stock and total portfolio
    for name, pnl, position_value in zip(
        stock_names,
        pnl_columns,
        position_values
    ):

        var = var_empirical(pnl, alpha)[0]
        es = es_empirical(pnl, alpha)[0]

        results.append({
            "Stock": name,
            f"VaR{confidence}": var,
            f"ES{confidence}": es,
            f"VaR{confidence}_Pct": var / position_value,
            f"ES{confidence}_Pct": es / position_value
        })

    return pd.DataFrame(results)
