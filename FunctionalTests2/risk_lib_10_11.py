import numpy as np
from scipy import stats
from scipy.optimize import minimize


# ------------------------------------------------------------------
# Portfolio optimisation (normal assumption: risk = portfolio std dev)
# ------------------------------------------------------------------
def portfolio_volatility(weights, cov):
    weights = np.asarray(weights, dtype=float)
    return np.sqrt(weights @ cov @ weights)


def risk_contributions(weights, cov):
    """Each asset's share of portfolio volatility: w_i * (cov @ w)_i / sigma_p.
    The contributions add up to the portfolio volatility."""
    weights = np.asarray(weights, dtype=float)
    return weights * (cov @ weights) / portfolio_volatility(weights, cov)


def risk_parity_weights(cov, risk_budget=None):
    """Long-only, fully invested weights whose risk contributions are in
    proportion to risk_budget (equal risk contributions if None).

    Example: risk_budget = [1, 1, 1, 1, 0.5] gives the last asset half the
    risk of each of the others.
    """
    cov = np.asarray(cov, dtype=float)
    n = cov.shape[0]
    budget = np.ones(n) if risk_budget is None else np.asarray(risk_budget, dtype=float)

    def objective(w):
        # Risk contribution per unit of budget should be equal across assets
        scaled = risk_contributions(w, cov) / budget
        return 1e5 * np.sum((scaled - scaled.mean()) ** 2)

    start = np.full(n, 1.0 / n)
    result = minimize(objective, start, method="SLSQP",
                      bounds=[(0.0, None)] * n,
                      constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0}],
                      options={"ftol": 1e-14, "maxiter": 1000})
    return result.x


def max_sharpe_weights(means, cov, rf, lower=0.0, upper=None):
    """Fully invested weights that maximise (w @ means - rf) / sigma_p,
    with every weight between lower and upper."""
    means = np.asarray(means, dtype=float)
    cov = np.asarray(cov, dtype=float)
    n = len(means)

    def negative_sharpe(w):
        return -(w @ means - rf) / portfolio_volatility(w, cov)

    start = np.full(n, 1.0 / n)
    result = minimize(negative_sharpe, start, method="SLSQP",
                      bounds=[(lower, upper)] * n,
                      constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0}],
                      options={"ftol": 1e-14, "maxiter": 1000})
    return result.x



# ------------------------------------------------------------------
# Ex-post return and volatility attribution
# ------------------------------------------------------------------
def drift_weights(start_weights, returns):
    """Weights at the start of each period, drifting with asset returns.

    returns : T x N array of asset returns
    Returns (T x N weights, length-T portfolio returns).
    """
    returns = np.asarray(returns, dtype=float)
    n_periods = returns.shape[0]
    weights = np.zeros_like(returns)
    port_returns = np.zeros(n_periods)

    w = np.asarray(start_weights, dtype=float).copy()
    for t in range(n_periods):
        weights[t] = w
        port_returns[t] = w @ returns[t]
        w = w * (1 + returns[t]) / (1 + port_returns[t])
    return weights, port_returns


def carino_k(port_returns):
    """Carino linking factors so that single-period contributions add up
    to the total multi-period portfolio return."""
    port_returns = np.asarray(port_returns, dtype=float)
    total = np.prod(1 + port_returns) - 1
    big_k = np.log1p(total) / total
    return np.log1p(port_returns) / (port_returns * big_k)


def vol_attribution(contributions, port_returns):
    """Split portfolio volatility: sd(R) * beta of each contribution series
    on the portfolio return. contributions is T x K."""
    contributions = np.asarray(contributions, dtype=float)
    port_returns = np.asarray(port_returns, dtype=float)
    var_p = np.var(port_returns, ddof=1)
    betas = np.array([np.cov(c, port_returns, ddof=1)[0, 1] / var_p
                      for c in contributions.T])
    return betas * np.std(port_returns, ddof=1)


def expost_attribution(start_weights, returns):
    """Ex-post attribution by asset.

    Returns a dict with arrays (one value per asset, then the portfolio):
      total_return, return_attribution, vol_attribution
    """
    returns = np.asarray(returns, dtype=float)
    weights, port_returns = drift_weights(start_weights, returns)
    contributions = weights * returns                      # T x N

    k = carino_k(port_returns)
    port_total = np.prod(1 + port_returns) - 1

    total_return = np.append(np.prod(1 + returns, axis=0) - 1, port_total)
    return_attr = np.append(k @ contributions, port_total)
    vol_attr = vol_attribution(contributions, port_returns)
    vol_attr = np.append(vol_attr, np.std(port_returns, ddof=1))
    return {"total_return": total_return,
            "return_attribution": return_attr,
            "vol_attribution": vol_attr}


def expost_factor_attribution(start_weights, stock_returns, factor_returns, betas):
    """Ex-post attribution to factors plus alpha.

    betas : N stocks x F factors
    Returns a dict with arrays (one value per factor, then Alpha, then the
    portfolio): total_return, return_attribution, vol_attribution
    """
    stock_returns = np.asarray(stock_returns, dtype=float)
    factor_returns = np.asarray(factor_returns, dtype=float)
    betas = np.asarray(betas, dtype=float)

    weights, port_returns = drift_weights(start_weights, stock_returns)

    # Portfolio factor exposure each period, and the return it explains
    exposures = weights @ betas                            # T x F
    factor_contrib = exposures * factor_returns            # T x F
    alpha = port_returns - factor_contrib.sum(axis=1)      # T
    contributions = np.column_stack([factor_contrib, alpha])

    k = carino_k(port_returns)
    port_total = np.prod(1 + port_returns) - 1

    total_return = np.append(np.prod(1 + factor_returns, axis=0) - 1,
                             [np.prod(1 + alpha) - 1, port_total])
    return_attr = np.append(k @ contributions, port_total)
    vol_attr = np.append(vol_attribution(contributions, port_returns),
                         np.std(port_returns, ddof=1))
    return {"total_return": total_return,
            "return_attribution": return_attr,
            "vol_attribution": vol_attr}

