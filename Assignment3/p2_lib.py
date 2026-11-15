"""
p2_lib.py
Reusable CRR binomial-tree pricer for European / American options
on a stock that pays known DISCRETE CASH dividends.

Dividend treatment: escrowed-dividend model (Hull).
  S*(0) = S0 - PV(all dividends paid before T)
  The tree is built on S* (recombining, volatility sigma).
  At a node at time t:  S(t) = S*(t) + PV_t(dividends paid in (t, T]).
  Exercise value and payoffs use S(t).

Convention for a dividend exactly at step n (t_n = t_div):
  the node at step n is the instant JUST BEFORE the dividend (cum-dividend),
  so the dividend at t_n is still included in S(t_n). This is the moment at
  which early exercise of a call can be optimal.
"""

import math
import numpy as np


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def pv_dividends(divs, r, t_from=0.0, T=None, include_at_t=True):
    """PV at time t_from of dividends paid after t_from (and on/before T).
    divs: list of (time, amount). include_at_t=True counts a dividend paid
    exactly at t_from (cum-dividend node)."""
    pv = 0.0
    for td, D in divs:
        if T is not None and td > T + 1e-12:
            continue
        after = td >= t_from - 1e-12 if include_at_t else td > t_from + 1e-12
        if after:
            pv += D * math.exp(-r * (td - t_from))
    return pv


def early_exercise_check(divs, X, r, T):
    """Week 7 condition for an American call: early exercise just before
    dividend n can be optimal only if  D_n > X * (1 - exp(-r (T - t_n))).
    (For the last dividend; for earlier ones the comparison is against the
    interest on X until the NEXT dividend: D_n > X(1 - e^{-r(t_{n+1}-t_n)}).
    Both versions are returned.)"""
    out = []
    times = [t for t, _ in divs]
    for i, (tn, Dn) in enumerate(divs):
        thr_T = X * (1 - math.exp(-r * (T - tn)))
        t_next = times[i + 1] if i + 1 < len(divs) else T
        thr_next = X * (1 - math.exp(-r * (t_next - tn)))
        out.append(dict(t=tn, D=Dn,
                        threshold_to_T=thr_T, possible_vs_T=Dn > thr_T,
                        threshold_to_next=thr_next, possible_vs_next=Dn > thr_next))
    return out


# ------------------------------------------------------------------
# Core pricer
# ------------------------------------------------------------------
def binomial_price(S0, X, T, r, sigma, N, divs=(), kind="call",
                   american=False, return_tree=False):
    """CRR binomial tree with escrowed discrete cash dividends.

    Parameters
    ----------
    S0, X, T, r, sigma : spot, strike, maturity (yrs), cc risk-free, vol
    N      : number of steps
    divs   : iterable of (t_div, amount)
    kind   : 'call' or 'put'
    american : True -> allow early exercise at every node
    return_tree : also return dict with stock/option trees and
                  early-exercise flags

    Returns price (and optionally the trees).
    """
    divs = [(t, D) for t, D in divs if 0 < t <= T]
    dt = T / N
    u = math.exp(sigma * math.sqrt(dt))
    d = 1 / u
    a = math.exp(r * dt)
    p = (a - d) / (u - d)
    disc = math.exp(-r * dt)
    if not (0 < p < 1):
        raise ValueError("Risk-neutral probability outside (0,1); increase N.")

    S_star0 = S0 - pv_dividends(divs, r, 0.0, T)
    sign = 1.0 if kind == "call" else -1.0

    def stock(n):
        """Full (cum-dividend) stock prices at step n, j = 0..n up-moves."""
        t = n * dt
        j = np.arange(n + 1)
        s_star = S_star0 * u ** j * d ** (n - j)
        return s_star + pv_dividends(divs, r, t, T, include_at_t=True)

    # terminal
    ST = stock(N)
    V = np.maximum(sign * (ST - X), 0.0)

    stock_tree, opt_tree, ex_tree = [None] * (N + 1), [None] * (N + 1), [None] * (N + 1)
    if return_tree:
        stock_tree[N], opt_tree[N] = ST, V.copy()
        ex_tree[N] = V > 0

    for n in range(N - 1, -1, -1):
        cont = disc * (p * V[1:] + (1 - p) * V[:-1])
        if american:
            Sn = stock(n)
            exer = np.maximum(sign * (Sn - X), 0.0)
            ex_flag = exer > cont + 1e-12
            V = np.where(ex_flag, exer, cont)
        else:
            V = cont
            ex_flag = np.zeros(n + 1, dtype=bool)
        if return_tree:
            stock_tree[n], opt_tree[n], ex_tree[n] = stock(n), V.copy(), ex_flag

    price = float(V[0])
    if return_tree:
        return price, dict(dt=dt, u=u, d=d, p=p, stock=stock_tree,
                           option=opt_tree, exercise=ex_tree)
    return price


def early_exercise_steps(tree):
    """List of (step, time, number_of_nodes_exercised) where early exercise
    is optimal somewhere in the tree."""
    res = []
    for n, flags in enumerate(tree["exercise"][:-1]):
        k = int(np.sum(flags))
        if k:
            res.append((n, n * tree["dt"], k))
    return res


def price_all_four(S0, X, T, r, sigma, N, divs=()):
    """Return dict with EC, AC, EP, AP and the two early-exercise premiums."""
    ec = binomial_price(S0, X, T, r, sigma, N, divs, "call", False)
    ac = binomial_price(S0, X, T, r, sigma, N, divs, "call", True)
    ep = binomial_price(S0, X, T, r, sigma, N, divs, "put", False)
    ap = binomial_price(S0, X, T, r, sigma, N, divs, "put", True)
    return dict(EC=ec, AC=ac, EP=ep, AP=ap,
                call_premium=ac - ec, put_premium=ap - ep)


def black_scholes(S0, X, T, r, sigma, kind="call", divs=()):
    """European BS with escrowed dividends (benchmark for convergence)."""
    S = S0 - pv_dividends(divs, r, 0.0, T)
    d1 = (math.log(S / X) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    N_ = lambda x: 0.5 * (1 + math.erf(x / math.sqrt(2)))
    if kind == "call":
        return S * N_(d1) - X * math.exp(-r * T) * N_(d2)
    return X * math.exp(-r * T) * N_(-d2) - S * N_(-d1)


# ------------------------------------------------------------------
# Non-recombining tree for discrete cash dividends (Hull, ch. 21)
# ------------------------------------------------------------------
def nonrecomb_price(S0, X, T, r, sigma, N, divs=(), kind="call", american=False):
    """CRR tree on the actual stock price. At each dividend step the price
    drops by D, so every node there starts its own new sub-tree (the tree
    stops recombining). Dividend dates must fall exactly on tree steps.

    At a dividend step the holder decides JUST BEFORE the dividend:
        value = max(exercise at cum-div price, value of holding ex-div)
    and the ex-div node itself also allows exercise (relevant for puts).
    """
    divs = sorted((t, D) for t, D in divs if 0 < t < T)
    dt = T / N
    u = math.exp(sigma * math.sqrt(dt)); d = 1 / u
    p = (math.exp(r * dt) - d) / (u - d); disc = math.exp(-r * dt)
    sign = 1.0 if kind == "call" else -1.0
    payoff = lambda S: np.maximum(sign * (S - X), 0.0)

    ks = []
    for t, _ in divs:
        k = t / dt
        if abs(k - round(k)) > 1e-9:
            raise ValueError(f"Dividend at t={t} is not on a tree step (N={N}).")
        ks.append(int(round(k)))
    bounds = [0] + ks + [N]

    CHUNK = 2000   # sub-trees processed at once (keeps memory small)

    def seg_value(i, S_start, chunk=2000):
        """Option values at the start nodes of segment i (one per sub-tree).
        Sub-trees are processed in chunks to keep memory small."""
        if len(S_start) > chunk:
            return np.concatenate([seg_value(i, S_start[c:c + chunk], chunk)
                                   for c in range(0, len(S_start), chunk)])
        m = bounds[i + 1] - bounds[i]
        j = np.arange(m + 1)
        S_end = S_start[:, None] * (u ** j * d ** (m - j))
        if i == len(bounds) - 2:                       # last segment -> maturity
            V = payoff(S_end)
        else:                                          # dividend at segment end
            D = divs[i][1]
            hold = seg_value(i + 1, np.maximum(S_end - D, 0.0).ravel(), chunk)
            hold = hold.reshape(S_end.shape)
            V = np.maximum(payoff(S_end), hold) if american else hold
        del S_end
        for n in range(m - 1, -1, -1):
            V = disc * (p * V[:, 1:] + (1 - p) * V[:, :-1])
            if american:
                jj = j[:n + 1]
                np.maximum(V, payoff(S_start[:, None] * (u ** jj * d ** (n - jj))), out=V)
        return V[:, 0]

    return float(seg_value(0, np.array([float(S0)]))[0])


def gbsm(S, X, T, r, b, sigma, kind="call"):
    """Generalized Black-Scholes-Merton (cost of carry b)."""
    Nf = lambda x: 0.5 * (1 + math.erf(x / math.sqrt(2)))
    d1 = (math.log(S / X) + (b + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if kind == "call":
        return S * math.exp((b - r) * T) * Nf(d1) - X * math.exp(-r * T) * Nf(d2)
    return X * math.exp(-r * T) * Nf(-d2) - S * math.exp((b - r) * T) * Nf(-d1)


def fd_delta_gamma(pricer, S0, h):
    """Central finite differences in S0. pricer: function of S0."""
    up, mid, dn = pricer(S0 + h), pricer(S0), pricer(S0 - h)
    return (up - dn) / (2 * h), (up - 2 * mid + dn) / h ** 2