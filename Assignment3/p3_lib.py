"""Library: Black-Scholes pricing and Greeks for European options (no dividends)."""
import numpy as np
from scipy.stats import norm

def _d1d2(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return d1, d1 - sigma * np.sqrt(T)

def bs_price(S, K, T, r, sigma, kind):
    """Black-Scholes price. kind = 'call' or 'put'. T in years."""
    if T <= 0:
        return np.maximum(S - K, 0) if kind == "call" else np.maximum(K - S, 0)
    d1, d2 = _d1d2(S, K, T, r, sigma)
    if kind == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def bs_greeks(S, K, T, r, sigma, kind):
    """Delta, gamma, and theta (per year, i.e. dV/dt in calendar-year units)."""
    d1, d2 = _d1d2(S, K, T, r, sigma)
    pdf = norm.pdf(d1)
    gamma = pdf / (S * sigma * np.sqrt(T))
    common = -S * pdf * sigma / (2 * np.sqrt(T))
    if kind == "call":
        delta = norm.cdf(d1)
        theta = common - r * K * np.exp(-r * T) * norm.cdf(d2)
    else:
        delta = norm.cdf(d1) - 1
        theta = common + r * K * np.exp(-r * T) * norm.cdf(-d2)
    return {"delta": delta, "gamma": gamma, "theta": theta}

def position_value(row, S, r, days_per_year, shift_days=0):
    """Value of one position (Holding x unit value) at spot S, `shift_days` trading days later."""
    if row["Type"] == "Stock":
        return row["Holding"] * S
    T = (row["DaysToExpiry"] - shift_days) / days_per_year
    return row["Holding"] * bs_price(S, row["Strike"], T, r, row["ImpliedVol"], row["Type"].lower())

def position_greeks(row, S, r, days_per_year):
    """Holding-weighted delta, gamma, theta (theta per year) of one position."""
    if row["Type"] == "Stock":
        return {"delta": row["Holding"], "gamma": 0.0, "theta": 0.0}
    T = row["DaysToExpiry"] / days_per_year
    g = bs_greeks(S, row["Strike"], T, r, row["ImpliedVol"], row["Type"].lower())
    return {k: row["Holding"] * v for k, v in g.items()}

def delta_normal_var(delta, S, sigma_daily, horizon, alpha=0.05):
    """Delta-normal VaR (positive number = loss), zero mean."""
    return abs(delta) * S * sigma_daily * np.sqrt(horizon) * norm.ppf(1 - alpha)


# ---------------------------------------------------------------------------
# Return statistics, distribution fitting, VaR / ES
# ---------------------------------------------------------------------------
from scipy import stats, optimize

def four_moments(x):
    """Mean, variance (sample), skewness, excess kurtosis (bias-corrected)."""
    x = np.asarray(x)
    return {"mean": x.mean(), "variance": x.var(ddof=1),
            "skew": stats.skew(x, bias=False), "excess_kurtosis": stats.kurtosis(x, bias=False)}

def aicc(loglik, k, n):
    return 2 * k - 2 * loglik + 2 * k * (k + 1) / (n - k - 1)

def fit_normal(x):
    """MLE normal fit. Returns dict with params, loglik, AICc."""
    x = np.asarray(x); n = len(x)
    mu, sd = x.mean(), x.std(ddof=0)          # MLE
    ll = stats.norm.logpdf(x, mu, sd).sum()
    return {"mu": mu, "sigma": sd, "loglik": ll, "aicc": aicc(ll, 2, n)}

def fit_t(x):
    """MLE location-scale Student t fit (nu, mu, s). Returns dict with params, loglik, AICc."""
    x = np.asarray(x); n = len(x)
    def nll(p):
        nu, mu, s = p
        if nu <= 2.0001 or s <= 0:
            return np.inf
        return -stats.t.logpdf(x, nu, mu, s).sum()
    x0 = [6.0, x.mean(), x.std() * np.sqrt(4 / 6)]
    r = optimize.minimize(nll, x0, method="Nelder-Mead", options={"xatol": 1e-10, "fatol": 1e-10, "maxiter": 20000})
    nu, mu, s = r.x
    ll = -r.fun
    return {"nu": nu, "mu": mu, "sigma": s, "loglik": ll, "aicc": aicc(ll, 3, n)}

def var_es(pnl, alpha):
    """Empirical VaR and ES (reported as positive losses) from a P&L sample."""
    pnl = np.sort(np.asarray(pnl))
    q = np.quantile(pnl, alpha)
    return -q, -pnl[pnl <= q].mean()

def delta_normal_var_es(delta, S, sigma_daily, horizon, alpha):
    """Delta-normal VaR and ES, zero mean, positive = loss."""
    sd = abs(delta) * S * sigma_daily * np.sqrt(horizon)
    z = stats.norm.ppf(alpha)
    return -z * sd, sd * stats.norm.pdf(z) / alpha

def book_value(book, S, r, days_per_year, shift_days=0):
    """Full revaluation of a book (DataFrame of positions) for an array of spot prices S."""
    S = np.asarray(S, dtype=float)
    return sum(position_value(row, S, r, days_per_year, shift_days) for _, row in book.iterrows())


def taylor_pnl(dS, delta, gamma=0.0, theta=0.0, dt=0.0):
    """Second-order Taylor P&L: delta*dS + 0.5*gamma*dS^2 + theta*dt (theta per year, dt in years)."""
    dS = np.asarray(dS)
    return delta * dS + 0.5 * gamma * dS**2 + theta * dt