"""Reusable library functions for return statistics, Sharpe, ES and optimization."""
import numpy as np
from scipy import stats
from scipy.optimize import minimize


def moments(x):
    """First four moments: mean, std (sample), skewness, excess kurtosis."""
    x = np.asarray(x, float)
    return {
        "mean": x.mean(),
        "std": x.std(ddof=1),
        "skew": stats.skew(x, bias=False),
        "exkurt": stats.kurtosis(x, fisher=True, bias=False),
    }


def periodic_rf(rf_annual, periods=12):
    """Convert an annual risk-free rate to a per-period rate."""
    return (1 + rf_annual) ** (1 / periods) - 1


def sharpe_annual(x, rf_annual=0.03, periods=12):
    """Annualized Sharpe ratio from periodic total returns."""
    ex = np.asarray(x, float) - periodic_rf(rf_annual, periods)
    return ex.mean() / ex.std(ddof=1) * np.sqrt(periods)


def leave_one_out_sharpe(x, rf_annual=0.03, periods=12):
    """Sharpe ratio recomputed with each single observation removed."""
    x = np.asarray(x, float)
    return np.array([sharpe_annual(np.delete(x, i), rf_annual, periods)
                     for i in range(len(x))])


def moment_share_of_obs(x, i, order):
    """Share of the order-th central moment contributed by observation i."""
    x = np.asarray(x, float)
    d = (x - x.mean()) ** order
    return d[i] / d.sum()


def es_historical(x, alpha=0.01):
    """Historical Expected Shortfall (positive = loss).
    Averages the worst alpha fraction, weighting the boundary observation
    fractionally (ES at 1% of 240 obs = worst 2.4 obs)."""
    x = np.sort(np.asarray(x, float))
    k = alpha * len(x)
    n_full = int(np.floor(k))
    tail = x[:n_full].sum()
    if n_full < len(x):
        tail += (k - n_full) * x[n_full]
    return -tail / k


def es_normal(mu, sigma, alpha=0.01):
    """ES of a normal with mean mu and sd sigma (positive = loss)."""
    z = stats.norm.ppf(alpha)
    return -(mu - sigma * stats.norm.pdf(z) / alpha)


def max_sharpe_long_only(R, rf_periodic):
    """Long-only, fully invested max-Sharpe weights from sample mean and cov."""
    R = np.asarray(R, float)
    mu = R.mean(axis=0) - rf_periodic
    S = np.cov(R, rowvar=False)
    n = len(mu)
    neg_sr = lambda w: -(w @ mu) / np.sqrt(w @ S @ w)
    res = minimize(neg_sr, np.ones(n) / n, method="SLSQP",
                   bounds=[(0, 1)] * n,
                   constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}])
    return res.x


# ---------------------------------------------------------------------
# Portfolio construction
# ---------------------------------------------------------------------
def max_sharpe(R, rf_periodic, bounds=None):
    """Max-Sharpe weights (fully invested) from sample mean and cov.
    bounds=None -> unconstrained (shorting allowed), closed-form tangency:
        w = Sigma^-1 (mu - rf) / 1' Sigma^-1 (mu - rf)
    bounds=list of (lo, hi) -> numerical optimization with those bounds."""
    R = np.asarray(R, float)
    mu = R.mean(axis=0) - rf_periodic
    S = np.cov(R, rowvar=False)
    if bounds is None:
        z = np.linalg.solve(S, mu)
        return z / z.sum()
    n = len(mu)
    neg_sr = lambda w: -(w @ mu) / np.sqrt(w @ S @ w)
    x0 = np.array([min(max(1 / n, lo), hi) for lo, hi in bounds])
    res = minimize(neg_sr, x0, method="SLSQP", bounds=bounds,
                   constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}],
                   options={"ftol": 1e-12, "maxiter": 1000})
    return res.x


def efficient_frontier(R, n_points=60, bounds=None):
    """Minimum-variance frontier. Returns arrays (target means, stds, weights).
    bounds default = long only [(0,1)]*n."""
    R = np.asarray(R, float)
    mu = R.mean(axis=0)
    S = np.cov(R, rowvar=False)
    n = len(mu)
    bounds = bounds or [(0, 1)] * n
    # lowest attainable mean = global min-variance portfolio's mean
    gmv = minimize(lambda w: w @ S @ w, np.ones(n) / n, method="SLSQP", bounds=bounds,
                   constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}])
    targets = np.linspace(gmv.x @ mu, mu.max(), n_points)
    means, stds, ws = [], [], []
    w0 = gmv.x
    for t in targets:
        cons = [{"type": "eq", "fun": lambda w: w.sum() - 1},
                {"type": "eq", "fun": lambda w, t=t: w @ mu - t}]
        res = minimize(lambda w: w @ S @ w, w0, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"ftol": 1e-14, "maxiter": 1000})
        if res.success:
            w0 = res.x
            means.append(res.x @ mu)
            stds.append(np.sqrt(res.x @ S @ res.x))
            ws.append(res.x)
    return np.array(means), np.array(stds), np.array(ws)


# ---------------------------------------------------------------------
# VaR (positive = loss)
# ---------------------------------------------------------------------
def var_historical(x, alpha=0.05):
    """Historical VaR: negative of the alpha empirical quantile."""
    return -np.quantile(np.asarray(x, float), alpha)


def var_normal(mu, sigma, alpha=0.05):
    """VaR of a normal with mean mu and sd sigma."""
    return -(mu + sigma * stats.norm.ppf(alpha))


def risk_table(x, alphas=(0.05, 0.01)):
    """Historical and normal VaR / ES at each alpha for a return series."""
    x = np.asarray(x, float)
    mu, sd = x.mean(), x.std(ddof=1)
    out = {}
    for a in alphas:
        p = f"{int(round(a * 100))}%"
        out[f"VaR{p} hist"] = var_historical(x, a)
        out[f"VaR{p} norm"] = var_normal(mu, sd, a)
        out[f"ES{p} hist"] = es_historical(x, a)
        out[f"ES{p} norm"] = es_normal(mu, sd, a)
    return out


# ---------------------------------------------------------------------
# Regression
# ---------------------------------------------------------------------
def ols(y, x):
    """OLS of y on [1, x]. Returns dict with alpha, beta, their SEs and
    t-stats, R^2 and n."""
    y = np.asarray(y, float)
    X = np.column_stack([np.ones(len(y)), np.asarray(x, float)])
    n, k = X.shape
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    e = y - X @ b
    s2 = e @ e / (n - k)
    se = np.sqrt(np.diag(s2 * np.linalg.inv(X.T @ X)))
    r2 = 1 - (e @ e) / ((y - y.mean()) @ (y - y.mean()))
    return {"alpha": b[0], "beta": b[1], "se_alpha": se[0], "se_beta": se[1],
            "t_alpha": b[0] / se[0], "t_beta": b[1] / se[1], "r2": r2, "n": n}


# ---------------------------------------------------------------------
# Estimation error
# ---------------------------------------------------------------------
def se_mean(x):
    """Standard error of the sample mean: s / sqrt(n)."""
    x = np.asarray(x, float)
    return x.std(ddof=1) / np.sqrt(len(x))


def bootstrap_max_sharpe(R, rf_periodic, bounds=None, n_boot=1000, seed=0):
    """Resample months (rows) with replacement and re-solve max_sharpe.
    Returns (weights array n_boot x n_assets, index array of resampled rows)."""
    R = np.asarray(R, float)
    rng = np.random.default_rng(seed)
    T = len(R)
    W, idx = [], []
    for _ in range(n_boot):
        ix = rng.integers(0, T, T)
        W.append(max_sharpe(R[ix], rf_periodic, bounds))
        idx.append(ix)
    return np.array(W), np.array(idx)


def ols_multi(y, X):
    """OLS of y on [1, X] with several regressors.
    Returns coefficients, standard errors, t-stats (intercept first) and R^2."""
    y = np.asarray(y, float)
    X = np.column_stack([np.ones(len(y)), np.asarray(X, float)])
    n, k = X.shape
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    e = y - X @ b
    s2 = e @ e / (n - k)
    se = np.sqrt(np.diag(s2 * np.linalg.inv(X.T @ X)))
    r2 = 1 - (e @ e) / ((y - y.mean()) @ (y - y.mean()))
    return {"coef": b, "se": se, "t": b / se, "r2": r2, "n": n}


def return_to_es(x, rf_periodic, alpha=0.01):
    """Mean monthly excess return divided by monthly historical ES."""
    x = np.asarray(x, float)
    return (x.mean() - rf_periodic) / es_historical(x, alpha)