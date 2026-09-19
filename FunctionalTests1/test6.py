import numpy as np
import pandas as pd

# Test 6.1: calculate arithmetic returns

# Load the price data
prices = pd.read_csv("test6.csv")

# Keep Date separate, coerce only the price columns to numeric
dates = prices["Date"]
price_cols = prices.drop(columns=["Date"]).apply(pd.to_numeric, errors="coerce")

# Calculate arithmetic returns: r_t = P_t / P_{t-1} - 1
returns = price_cols.pct_change()

# Add Date back
returns.insert(0, "Date", dates)

# Drop only the first row (no previous price to compare against)
returns = returns.iloc[1:].reset_index(drop=True)

# Save and preview
print(returns)



# Test 6.2: calculate log returns
# Load the price data
prices = pd.read_csv("test6.csv")

# Keep Date separate, coerce only the price columns to numeric
dates = prices["Date"]
price_cols = prices.drop(columns=["Date"]).apply(pd.to_numeric, errors="coerce")

# Calculate log returns: r_t = ln(P_t / P_{t-1})
returns = np.log(price_cols / price_cols.shift(1))

# Add Date back
returns.insert(0, "Date", dates)

# Drop only the first row (no previous price to compare against)
returns = returns.iloc[1:].reset_index(drop=True)

# Save and preview
print(returns)

