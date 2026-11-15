"""
p1_lib.py

Reusable functions for option-chain analysis.

Sections:
1. Put-call parity
2. Butterfly convexity check
3. Generalized Black-Scholes-Merton (GBSM)
4. Implied volatility

This file only defines functions. The market inputs and the analysis for
a particular assignment belong in the question-specific script.
"""

import numpy as np
import pandas as pd

from math import erf, exp, log, pi, sqrt


# ===========================================================================
# 1. PUT-CALL PARITY
# ===========================================================================

def check_put_call_parity(df, S, r, q, T):
    """
    Check put-call parity using option mid prices.

    C - P = S * exp(-q*T) - K * exp(-r*T)

    A strike is flagged when the absolute parity gap is larger than the
    combined call and put bid-ask spreads.

    Expected columns in df:
        Type, Strike, Bid, Ask
    """

    # Work on a copy so the original DataFrame is not changed.
    data = df.copy()

    # Standardize option type and calculate mid prices and bid-ask spreads.
    data["Type"] = data["Type"].str.strip().str.capitalize()
    data["Mid"] = (data["Bid"] + data["Ask"]) / 2.0
    data["Spread"] = data["Ask"] - data["Bid"]

    # Separate calls and puts, using strike as the matching key.
    calls = data[data["Type"] == "Call"].set_index("Strike")
    puts = data[data["Type"] == "Put"].set_index("Strike")

    # Keep only strikes that have both a call and a put quote.
    paired = calls[["Mid", "Spread"]].join(
        puts[["Mid", "Spread"]],
        how="inner",
        lsuffix="_C",
        rsuffix="_P"
    ).sort_index()

    K = paired.index

    # Theoretical and market call-minus-put differences.
    paired["Theoretical"] = (
        S * np.exp(-q * T) - K * np.exp(-r * T)
    )
    paired["Market"] = paired["Mid_C"] - paired["Mid_P"]

    # Difference between the market value and the parity value.
    paired["Gap"] = paired["Market"] - paired["Theoretical"]

    # Combined bid-ask spread of the call and put.
    paired["Band"] = paired["Spread_C"] + paired["Spread_P"]

    # Flag a strike if the parity gap is larger than the spread band.
    paired["Flag"] = paired["Gap"].abs() > paired["Band"]

    return paired


# ===========================================================================
# 2. BUTTERFLY CONVEXITY CHECK
# ===========================================================================

def butterfly_cost(prices):
    """
    Calculate the butterfly cost at each interior strike.

    For equally spaced strikes:
        Butterfly cost = V(K-h) - 2*V(K) + V(K+h)

    A negative value may indicate a violation of option-price convexity.

    The input prices must be ordered by increasing, equally spaced strikes.
    The first and last entries are NaN because they do not have two neighbors.
    """

    prices = np.asarray(prices, dtype=float)

    # Start with NaN values. Only interior strikes have a butterfly cost.
    result = np.full(len(prices), np.nan)

    # Apply the second-difference formula to each interior strike.
    result[1:-1] = (
        prices[:-2] - 2.0 * prices[1:-1] + prices[2:]
    )

    return result


# ===========================================================================
# 3. GENERALIZED BLACK-SCHOLES-MERTON (GBSM)
# ===========================================================================

def norm_cdf(x):
    """Standard normal cumulative distribution function."""
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def norm_pdf(x):
    """Standard normal probability density function."""
    return exp(-0.5 * x * x) / sqrt(2.0 * pi)


def d1_d2(S, K, T, r, b, sigma):
    """
    Calculate d1 and d2 in the GBSM model.

    b is the cost of carry. For a dividend-paying stock:
        b = r - q
    """

    if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
        raise ValueError("S, K, T, and sigma must be positive.")

    vol_sqrt_T = sigma * sqrt(T)

    d1 = (
        log(S / K) + (b + 0.5 * sigma ** 2) * T
    ) / vol_sqrt_T

    d2 = d1 - vol_sqrt_T

    return d1, d2


def gbsm_price(is_call, S, K, T, r, b, sigma):
    """
    Calculate a European call or put price using GBSM.

    is_call:
        True  -> call
        False -> put
    """

    d1, d2 = d1_d2(S, K, T, r, b, sigma)

    stock_discount = exp((b - r) * T)
    strike_discount = exp(-r * T)

    if is_call:
        price = (
            S * stock_discount * norm_cdf(d1)
            - K * strike_discount * norm_cdf(d2)
        )
    else:
        price = (
            K * strike_discount * norm_cdf(-d2)
            - S * stock_discount * norm_cdf(-d1)
        )

    return price


def gbsm_vega(S, K, T, r, b, sigma):
    """
    Calculate GBSM vega.

    Vega is the option-price change for a 1.00 change in volatility.
    Divide by 100 to get the price change for one volatility percentage point.
    """

    d1, _ = d1_d2(S, K, T, r, b, sigma)

    vega = (
        S * exp((b - r) * T) * norm_pdf(d1) * sqrt(T)
    )

    return vega


# ===========================================================================
# 4. NO-ARBITRAGE PRICE BOUNDS
# ===========================================================================

def price_bounds(is_call, S, K, T, r, b):
    """
    Return the lower and upper no-arbitrage price bounds for a European option.
    """

    stock_pv = S * exp((b - r) * T)
    strike_pv = K * exp(-r * T)

    if is_call:
        lower_bound = max(stock_pv - strike_pv, 0.0)
        upper_bound = stock_pv
    else:
        lower_bound = max(strike_pv - stock_pv, 0.0)
        upper_bound = strike_pv

    return lower_bound, upper_bound


# ===========================================================================
# 5. IMPLIED VOLATILITY
# ===========================================================================

def implied_vol(
    is_call, S, K, T, r, b, price,
    low_vol=1e-6, high_vol=5.0, tolerance=1e-10
):
    """
    Find implied volatility by inverting the GBSM price with bisection.

    Returns:
        (sigma, "")    if an implied volatility is found
        (None, reason) if the quote is outside the model's price bounds
                       or outside the volatility search range
    """

    if T <= 0:
        return None, "time to maturity must be positive"

    # Check whether the market price is inside the theoretical price bounds.
    lower_bound, upper_bound = price_bounds(
        is_call, S, K, T, r, b
    )

    if price <= lower_bound:
        return None, (
            f"price {price:.4f} <= lower bound {lower_bound:.4f}"
        )

    if price >= upper_bound:
        return None, (
            f"price {price:.4f} >= upper bound {upper_bound:.4f}"
        )

    # Calculate model prices at the two ends of the volatility search range.
    price_at_low_vol = gbsm_price(
        is_call, S, K, T, r, b, low_vol
    )
    price_at_high_vol = gbsm_price(
        is_call, S, K, T, r, b, high_vol
    )

    if price < price_at_low_vol:
        return None, "price is below the model price at the minimum volatility"

    if price > price_at_high_vol:
        return None, (
            f"implied volatility is above the search limit of {high_vol:.2f}"
        )

    # Bisection: option price increases as volatility increases.
    for _ in range(200):
        middle_vol = 0.5 * (low_vol + high_vol)

        middle_price = gbsm_price(
            is_call, S, K, T, r, b, middle_vol
        )

        if middle_price < price:
            low_vol = middle_vol
        else:
            high_vol = middle_vol

        if high_vol - low_vol < tolerance:
            break

    implied_sigma = 0.5 * (low_vol + high_vol)

    return implied_sigma, ""
