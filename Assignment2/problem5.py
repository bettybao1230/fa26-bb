import numpy as np
import pandas as pd

from risklib import (
    arithmetic_returns,
    fit_market_model,
    simulate_market_model,
    simulated_portfolio_var,
    delta_normal_var,
)


# ============================================================
# Settings and Data
# ============================================================

N_SIM = 100_000
POSITION = 1_000_000
CONFIDENCE = 0.95
SEED = 585

prices = pd.read_csv("problem5.csv")
prices = prices[["MKT", "A", "B"]].apply(
    pd.to_numeric, errors="coerce"
)

# Calculate simple returns and align observations across assets.
returns = prices.apply(arithmetic_returns).dropna()

market = returns["MKT"].to_numpy()
stock_a = returns["A"].to_numpy()
stock_b = returns["B"].to_numpy()


# ============================================================
# Part (b): Fit Market Models
# ============================================================

alpha_a, beta_a, eps_a = fit_market_model(stock_a, market)
alpha_b, beta_b, eps_b = fit_market_model(stock_b, market)

residuals = np.column_stack([eps_a, eps_b])
residual_cov = np.cov(residuals, rowvar=False, ddof=1)
residual_corr = np.corrcoef(eps_a, eps_b)[0, 1]

print("\nPART (b): REGRESSION RESULTS")
print(f"Alpha A: {alpha_a:.8f}")
print(f"Beta A: {beta_a:.6f}")
print(f"Residual SD A: {np.std(eps_a, ddof=1):.8f}")

print(f"\nAlpha B: {alpha_b:.8f}")
print(f"Beta B: {beta_b:.6f}")
print(f"Residual SD B: {np.std(eps_b, ddof=1):.8f}")

print(f"\nResidual correlation: {residual_corr:.6f}")
print("\nResidual covariance matrix:")
print(residual_cov)


# ============================================================
# Part (c): Simulate Portfolio VaR
# ============================================================

market_sd = np.std(market, ddof=1)

# Scenario 1: Preserve the estimated residual covariance.
returns_a_full, returns_b_full = simulate_market_model(
    market_sd,
    beta_a,
    beta_b,
    residual_cov,
    n_sim=N_SIM,
    seed=SEED,
)

# Scenario 2: Assume the two residual series are uncorrelated.
residual_cov_zero = residual_cov.copy()
residual_cov_zero[0, 1] = 0.0
residual_cov_zero[1, 0] = 0.0

returns_a_zero, returns_b_zero = simulate_market_model(
    market_sd,
    beta_a,
    beta_b,
    residual_cov_zero,
    n_sim=N_SIM,
    seed=SEED,
)

portfolios = {
    "P1 (Long A, Long B)": (POSITION, POSITION),
    "P2 (Long A, Short B)": (POSITION, -POSITION),
}

results_c = []

for portfolio, (weight_a, weight_b) in portfolios.items():
    var_full = simulated_portfolio_var(
        returns_a_full,
        returns_b_full,
        weight_a,
        weight_b,
        confidence=CONFIDENCE,
    )

    var_zero = simulated_portfolio_var(
        returns_a_zero,
        returns_b_zero,
        weight_a,
        weight_b,
        confidence=CONFIDENCE,
    )

    results_c.append({
        "Portfolio": portfolio,
        "Full covariance VaR ($)": var_full,
        "Zero covariance VaR ($)": var_zero,
    })

print("\nPART (c): SIMULATED 95% VaR")
print(
    pd.DataFrame(results_c).to_string(
        index=False,
        formatters={
            "Full covariance VaR ($)": "${:,.2f}".format,
            "Zero covariance VaR ($)": "${:,.2f}".format,
        },
    )
)


# ============================================================
# Part (d): Delta-Normal VaR
# ============================================================

sample_cov = returns[["A", "B"]].cov().to_numpy()

print("\nSample covariance matrix of A and B:")
print(sample_cov)

print("\nPART (d): DELTA-NORMAL 95% VaR")

for portfolio, (weight_a, weight_b) in portfolios.items():
    weights = np.array([weight_a, weight_b])

    var = delta_normal_var(
        sample_cov,
        weights,
        confidence=CONFIDENCE,
    )

    print(f"{portfolio}: ${var:,.2f}")