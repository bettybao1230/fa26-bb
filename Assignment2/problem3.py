import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from risklib import (
    four_moments,
    compute_pnl,
    historical_var,
    historical_es,
    normal_var,
)


# ============================================================
# Load Data and Define Investment Amounts
# ============================================================
df = pd.read_csv("problem3.csv")

A = df["A"]
B = df["B"]

investment_A = 1_000_000
investment_B = 1_000_000
investment_concentrated = 2_000_000
investment_diversified = investment_A + investment_B


# ============================================================
# Plot P&L Distributions
# ============================================================
# Choice 1: Invest $2M entirely in A.
pnl_concentrated = compute_pnl(
    A, B, investment_concentrated, 0
)

# Choice 2: Invest $1M in each bond.
pnl_diversified = compute_pnl(
    A, B, investment_A, investment_B
)

plt.figure(figsize=(10, 6))

plt.hist(
    pnl_concentrated,
    bins=50,
    alpha=0.6,
    label="Concentrated: $2M in A",
)

plt.hist(
    pnl_diversified,
    bins=50,
    alpha=0.6,
    label="Diversified: $1M in A + $1M in B",
)

plt.xlabel("P&L ($)")
plt.ylabel("Frequency")
plt.title("Portfolio P&L Distributions")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# ============================================================
# Part (a): Calculate the First Four Moments
# ============================================================
# Report the mean, variance, skewness, and kurtosis of each bond.
for name, returns in [("Bond A", A), ("Bond B", B)]:
    mean, variance, skewness, kurtosis = four_moments(returns)

    print(f"\n{name}")
    print(f"Mean:     {mean:.6f}")
    print(f"Variance: {variance:.6f}")
    print(f"Skewness: {skewness:.6f}")
    print(f"Kurtosis: {kurtosis:.6f}")


# ============================================================
# Part (b): Count Losses Greater Than 20%
# ============================================================
# A return below -20% represents a loss greater than 20%.
A_large_loss = A < -0.20
B_large_loss = B < -0.20

# Count scenarios in which either bond experiences a large loss.
either_large_loss = A_large_loss | B_large_loss

print("\nPart (b): Scenarios with Losses Greater Than 20%")
print(f"Bond A:       {A_large_loss.sum()}")
print(f"Bond B:       {B_large_loss.sum()}")
print(f"At least one: {either_large_loss.sum()}")


# ============================================================
# Define Portfolio Returns and Values
# ============================================================
# Returns are expressed relative to each portfolio's total value.
returns_1m_A = A
returns_1m_B = B
returns_2m_A = A
returns_1m_each = (A + B) / 2

value_1m_A = investment_A
value_1m_B = investment_B
value_2m_A = investment_concentrated
value_1m_each = investment_diversified

positions = {
    "$1M in A": (returns_1m_A, value_1m_A),
    "$1M in B": (returns_1m_B, value_1m_B),
    "$2M in A": (returns_2m_A, value_2m_A),
    "$1M in A + $1M in B": (
        returns_1m_each,
        value_1m_each,
    ),
}


# ============================================================
# Helper Function: Print Historical VaR and ES
# ============================================================
def print_historical_risk(positions, alpha, part_label):
    """Print historical VaR and ES for each portfolio."""
    print(f"\n{part_label}: Historical VaR and ES")
    print(f"Alpha = {alpha:.0%}\n")

    for name, (returns, portfolio_value) in positions.items():
        # Calculate VaR using the risklib function.
        var = historical_var(
            returns,
            portfolio_value,
            alpha,
        )

        # Convert portfolio returns into dollar P&L and use
        # the risklib function to calculate expected shortfall.
        pnl = portfolio_value * returns
        es = historical_es(pnl, alpha)

        print(f"{name}:")
        print(f"  VaR = ${var:,.2f}")
        print(f"  ES  = ${es:,.2f}")
        print()


# ============================================================
# Part (c): Historical VaR and ES at 5%
# ============================================================
print_historical_risk(
    positions,
    alpha=0.05,
    part_label="Part (c)",
)


# ============================================================
# Part (d): Normal VaR at 5%
# ============================================================
# Compare parametric Normal VaR across the same four portfolios.
alpha = 0.05

print("Part (d): Normal VaR")
print(f"Alpha = {alpha:.0%}\n")

for name, (returns, portfolio_value) in positions.items():
    var = normal_var(
        returns,
        portfolio_value,
        alpha,
    )

    print(f"{name}:")
    print(f"  Normal VaR = ${var:,.2f}")
    print()


# ============================================================
# Part (e): Historical VaR and ES at 1%
# ============================================================
print_historical_risk(
    positions,
    alpha=0.01,
    part_label="Part (e)",
)

