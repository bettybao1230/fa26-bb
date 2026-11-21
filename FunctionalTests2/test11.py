"""
Question 11: Ex-Post Return and Volatility Attribution
Using functions from risklib.
"""

import pandas as pd
import risk_lib_10_11 as rl


# ============================================================
# 1. Ex-Post Attribution by Asset (Question 11.1)
# ============================================================
returns_11_1 = pd.read_csv("test11_1_returns.csv")
weights_11_1 = pd.read_csv("test11_1_weights.csv")["W"].to_numpy()

attr_11_1 = rl.expost_attribution(
    weights_11_1,
    returns_11_1.to_numpy()
)

columns_11_1 = list(returns_11_1.columns) + ["Portfolio"]


# ============================================================
# 2. Ex-Post Attribution by Factor (Question 11.2)
# ============================================================
stock_returns = pd.read_csv("test11_2_stock_returns.csv")
factor_returns = pd.read_csv("test11_2_factor_returns.csv")

betas = pd.read_csv("test11_2_beta.csv").set_index("Stock")
weights_11_2 = pd.read_csv("test11_2_weights.csv")["W"].to_numpy()

beta_matrix = betas.loc[
    stock_returns.columns,
    factor_returns.columns
].to_numpy()

attr_11_2 = rl.expost_factor_attribution(
    weights_11_2,
    stock_returns.to_numpy(),
    factor_returns.to_numpy(),
    beta_matrix
)

columns_11_2 = list(factor_returns.columns) + ["Alpha", "Portfolio"]


# ============================================================
# 3. Print Attribution Results
# ============================================================
row_names = [
    "TotalReturn",
    "Return Attribution",
    "Vol Attribution"
]

result_keys = [
    "total_return",
    "return_attribution",
    "vol_attribution"
]

results = {
    "11.1 Ex-Post Attribution by Asset": (attr_11_1, columns_11_1),
    "11.2 Ex-Post Attribution by Factor": (attr_11_2, columns_11_2)
}

for title, (attribution, columns) in results.items():

    output = pd.DataFrame(
        [attribution[key] for key in result_keys],
        columns=columns
    )

    output.insert(0, "Value", row_names)

    print(f"\n{title}")
    print(output.to_string(index=False, float_format=lambda x: repr(float(x))))
