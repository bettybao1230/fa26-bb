import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from risklib import arithmetic_returns

# ============================================================
# Load data
# ============================================================

df = pd.read_csv("problem2.csv")
prices = df["Price"].dropna().to_numpy()

# 501 prices -> 500 arithmetic returns
returns = arithmetic_returns(prices)

# Compute arithmetic returns and remove sample mean
mean_return = np.mean(returns)
demeaned_returns = returns - mean_return

print("Number of prices:", len(prices))
print("Number of returns:", len(returns))
print("Sample mean return:", mean_return)
print("Demand return:", demeaned_returns)

# ============================================================
# Plot demeaned returns
# ============================================================

plt.figure(figsize=(10, 5))

plt.plot(
    np.arange(1, len(demeaned_returns) + 1),
    demeaned_returns
)

plt.axhline(0, linewidth=1)

plt.xlabel("Day")
plt.ylabel("Demeaned Arithmetic Return")
plt.title("Demeaned Daily Arithmetic Returns")
plt.tight_layout()
plt.show()


from risklib import four_moments

# First four moments
mean, variance, skewness, kurtosis = four_moments(returns)
print()
print("First four moments of the full sample:")
print(f"Mean       = {mean:.6f}")
print(f"Variance   = {variance:.6f}")
print(f"Skewness   = {skewness:.6f}")
print(f"Kurtosis   = {kurtosis:.6f}")



from risklib import (
    normal_var,
    historical_var,
    student_t_var,
    ewma_normal_var,
    fit_student_t
)

# Inputs
portfolio_value = 1000000
alpha = 0.05

# Compute the five one-day VaR estimates
VaR_normal = normal_var(
    demeaned_returns,
    portfolio_value,
    alpha
)

VaR_ewma_97 = ewma_normal_var(
    demeaned_returns,
    portfolio_value,
    alpha,
    0.97
)

VaR_ewma_94 = ewma_normal_var(
    demeaned_returns,
    portfolio_value,
    alpha,
    0.94
)

VaR_t = student_t_var(
    demeaned_returns,
    portfolio_value,
    alpha
)

VaR_historical = historical_var(
    demeaned_returns,
    portfolio_value,
    alpha
)

# Fit Student-t distribution
df_t, loc_t, scale_t = fit_student_t(demeaned_returns)

# Print results
print("Five one-day VaR estimates:")
print(f"1. Normal, equally weighted: ${VaR_normal:,.2f}")
print(f"2. Normal, EWMA lambda=0.97: ${VaR_ewma_97:,.2f}")
print(f"3. Normal, EWMA lambda=0.94: ${VaR_ewma_94:,.2f}")
print(f"4. Student's t MLE:          ${VaR_t:,.2f}")
print(f"5. Historical simulation:    ${VaR_historical:,.2f}")

print("Fitted Student's t parameters:")
print(f"Degrees of freedom = {df_t:.6f}")
print(f"Location           = {loc_t:.6f}")
print(f"Scale              = {scale_t:.6f}")


regime_1 = demeaned_returns[:450]
regime_2 = demeaned_returns[450:]
sd_1 = np.std(regime_1, ddof=1)
sd_2 = np.std(regime_2, ddof=1)
print(f"Regime 1 SD = {sd_1:.6f}")
print(f"Regime 2 SD = {sd_2:.6f}")

from risklib import ewma_volatility
print(f"Volatility of lambda 0.94 = {ewma_volatility(demeaned_returns, 0.94)}")
print(f"Volatility of lambda 0.97 = {ewma_volatility(demeaned_returns, 0.97)}")

