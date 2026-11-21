"""
Question 10: Risk Parity and Maximum Sharpe Ratio Portfolios
Assumption: Normal distribution

All portfolio optimization functions are imported from risklib.
"""

import numpy as np
import pandas as pd
import risk_lib_10_11 as rl


# ============================================================
# 1. Load Input Data
# ============================================================
covariance_matrix = pd.read_csv("test5_2.csv").to_numpy()

expected_returns = (
    pd.read_csv("test10_3_means.csv")["Mean"].to_numpy()
)

risk_free_rate = 0.04


# ============================================================
# 2. Calculate Portfolio Weights
# ============================================================

# 10.1 Risk parity: equal risk contribution from each asset
weights_10_1 = rl.risk_parity_weights(covariance_matrix)


# 10.2 Risk parity: X5 has half the risk budget of other assets
risk_budget = [1, 1, 1, 1, 0.5]

weights_10_2 = rl.risk_parity_weights(
    covariance_matrix,
    risk_budget=risk_budget
)


# 10.3 Maximum Sharpe ratio: weights must be non-negative
weights_10_3 = rl.max_sharpe_weights(
    expected_returns,
    covariance_matrix,
    risk_free_rate,
    lower=0.0
)


# 10.4 Maximum Sharpe ratio: weights must be between 0.1 and 0.5
weights_10_4 = rl.max_sharpe_weights(
    expected_returns,
    covariance_matrix,
    risk_free_rate,
    lower=0.1,
    upper=0.5
)




# ============================================================
# 3. Print Portfolio Weights
# ============================================================

results = {
    "10.1": weights_10_1,
    "10.2": weights_10_2,
    "10.3": weights_10_3,
    "10.4": weights_10_4
}

for test_id, weights in results.items():

    print(f"\nQuestion {test_id}")
    print(pd.DataFrame({"W": weights}).to_string(index=False))
