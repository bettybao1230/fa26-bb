import numpy as np
import pandas as pd
from risk_library_8_9 import gaussian_copula_simulate, portfolio_var_es

# ------------------------------------------------------------------
# Parameters
# ------------------------------------------------------------------
ALPHA = 0.05
N_SIM = 1_000_000

# ------------------------------------------------------------------
# Load data
# ------------------------------------------------------------------
port = pd.read_csv("test9_1_portfolio.csv")
rets = pd.read_csv("test9_1_returns.csv")

names = port["Stock"].tolist()
holdings = port["Holding"].to_numpy(dtype=float)
prices = port["Starting Price"].to_numpy(dtype=float)
kinds = port["Distribution"].tolist()

values = holdings * prices

# ------------------------------------------------------------------
# Gaussian copula simulation
# ------------------------------------------------------------------
# 1. Simulate correlated stock returns
sim_returns, marginals = gaussian_copula_simulate(
    rets[names].to_numpy(),
    kinds,
    n=N_SIM
)

# 2. Convert simulated returns into dollar P&L
pnl = sim_returns * values

# 3. Calculate stock-level and portfolio-level VaR/ES
result = portfolio_var_es(
    pnl,
    names,
    values,
    ALPHA
)

print(result.to_string(index=False))