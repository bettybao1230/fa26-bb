import numpy as np
import scipy.stats as stats

# ------------------------------------------------------------------
# Option pricing: Generalized Black-Scholes-Merton (European)
#   b = cost of carry (b = r - q for a continuous dividend yield q)
# ------------------------------------------------------------------
def gbsm(is_call, S, K, T, r, b, vol):
    """European option price."""
    d1 = (np.log(S / K) + (b + 0.5 * vol ** 2) * T) / (vol * np.sqrt(T))
    d2 = d1 - vol * np.sqrt(T)
    carry = np.exp((b - r) * T)
    if is_call:
        return S * carry * stats.norm.cdf(d1) - K * np.exp(-r * T) * stats.norm.cdf(d2)
    return K * np.exp(-r * T) * stats.norm.cdf(-d2) - S * carry * stats.norm.cdf(-d1)


def gbsm_greeks(is_call, S, K, T, r, b, vol):
    """Closed-form Greeks for GBSM. Returns a dict with
    Delta, Gamma, Vega (per 1.00 of vol), Rho (per 1.00 of r, dividend
    yield held fixed) and Theta (change in value as time passes, per year)."""
    sqrt_t = np.sqrt(T)
    d1 = (np.log(S / K) + (b + 0.5 * vol ** 2) * T) / (vol * sqrt_t)
    d2 = d1 - vol * sqrt_t
    carry = np.exp((b - r) * T)
    disc = np.exp(-r * T)
    pdf_d1 = stats.norm.pdf(d1)

    gamma = carry * pdf_d1 / (S * vol * sqrt_t)
    vega = S * carry * pdf_d1 * sqrt_t
    time_decay = -S * carry * pdf_d1 * vol / (2 * sqrt_t)

    if is_call:
        delta = carry * stats.norm.cdf(d1)
        rho = T * K * disc * stats.norm.cdf(d2)
        theta = (time_decay - (b - r) * S * carry * stats.norm.cdf(d1)
                 - r * K * disc * stats.norm.cdf(d2))
    else:
        delta = carry * (stats.norm.cdf(d1) - 1)
        rho = -T * K * disc * stats.norm.cdf(-d2)
        theta = (time_decay + (b - r) * S * carry * stats.norm.cdf(-d1)
                 + r * K * disc * stats.norm.cdf(-d2))

    return {"Delta": delta, "Gamma": gamma, "Vega": vega, "Rho": rho, "Theta": theta}


# ------------------------------------------------------------------
# Option pricing: American options on a CRR binomial tree
# ------------------------------------------------------------------
def american_binomial(is_call, S, K, T, r, b, vol, n_steps=500):
    """American option price on a CRR binomial tree with cost of carry b.
    S may be a single price or an array of prices (priced all at once)."""
    S = np.atleast_1d(np.asarray(S, dtype=float))
    sign = 1.0 if is_call else -1.0

    dt = T / n_steps
    u = np.exp(vol * np.sqrt(dt))
    d = 1.0 / u
    p_up = (np.exp(b * dt) - d) / (u - d)
    discount = np.exp(-r * dt)

    # Option values at maturity (one row per starting price)
    j = np.arange(n_steps + 1)
    prices = S[:, None] * u ** j * d ** (n_steps - j)
    values = np.maximum(sign * (prices - K), 0.0)

    # Step back through the tree, allowing early exercise at every node
    for step in range(n_steps - 1, -1, -1):
        j = np.arange(step + 1)
        prices = S[:, None] * u ** j * d ** (step - j)
        hold = discount * (p_up * values[:, 1:] + (1 - p_up) * values[:, :-1])
        values = np.maximum(sign * (prices - K), hold)

    return values[:, 0] if values.shape[0] > 1 else values[0, 0]


def american_binomial_greeks(is_call, S, K, T, r, b, vol, n_steps=500,
                             bump=1e-6, gamma_bump=1.5):
    """Finite-difference Greeks for the American binomial price.

    Delta, Vega, Rho and Theta use central differences with a small bump.
    Gamma uses a larger price bump, because a tree's second derivative is
    noisy at small bumps.
    Rho changes r with the cost of carry b held fixed.
    Theta is the derivative with respect to time to maturity (dV/dT).
    """
    def price(S=S, T=T, r=r, vol=vol):
        return american_binomial(is_call, S, K, T, r, b, vol, n_steps)

    base = price()
    h, g = bump, gamma_bump
    return {
        "Value": base,
        "Delta": (price(S=S + h) - price(S=S - h)) / (2 * h),
        "Gamma": (price(S=S + g) - 2 * base + price(S=S - g)) / g ** 2,
        "Vega":  (price(vol=vol + h) - price(vol=vol - h)) / (2 * h),
        "Rho":   (price(r=r + h) - price(r=r - h)) / (2 * h),
        "Theta": (price(T=T + h) - price(T=T - h)) / (2 * h),
    }


def american_binomial_discrete_div(is_call, S, K, T, r, div_amounts, div_steps,
                                   vol, n_steps=500):
    """American option with discrete cash dividends on a binomial tree.

    div_amounts : cash dividend amounts
    div_steps   : tree step at which each dividend is paid (ascending)

    The tree runs up to the first dividend. At each node there, the holder
    either exercises, or keeps an option on the price after the dividend,
    valued with a new tree for the time left (with the remaining dividends).
    S may be a single price or an array of prices.
    """
    S = np.atleast_1d(np.asarray(S, dtype=float))
    div_amounts = list(div_amounts)
    div_steps = list(div_steps)

    # No dividends left before maturity: plain American tree (b = r)
    if not div_steps or div_steps[0] > n_steps:
        result = american_binomial(is_call, S, K, T, r, r, vol, n_steps)
        return np.atleast_1d(result) if len(S) > 1 else float(np.atleast_1d(result)[0])

    sign = 1.0 if is_call else -1.0
    dt = T / n_steps
    u = np.exp(vol * np.sqrt(dt))
    d = 1.0 / u
    p_up = (np.exp(r * dt) - d) / (u - d)
    discount = np.exp(-r * dt)

    first = div_steps[0]

    # Values at the first dividend date
    j = np.arange(first + 1)
    prices = S[:, None] * u ** j * d ** (first - j)
    after_div = (prices - div_amounts[0]).ravel()
    keep = american_binomial_discrete_div(
        is_call, after_div, K, T - first * dt, r,
        div_amounts[1:], [s - first for s in div_steps[1:]],
        vol, n_steps - first)
    keep = np.asarray(keep).reshape(prices.shape)
    values = np.maximum(sign * (prices - K), keep)

    # Step back from the dividend date to today
    for step in range(first - 1, -1, -1):
        j = np.arange(step + 1)
        prices = S[:, None] * u ** j * d ** (step - j)
        hold = discount * (p_up * values[:, 1:] + (1 - p_up) * values[:, :-1])
        values = np.maximum(sign * (prices - K), hold)

    return values[:, 0] if values.shape[0] > 1 else values[0, 0]
