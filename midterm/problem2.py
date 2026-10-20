import numpy as np
import pandas as pd
from risklib import normal_var_es

df = pd.read_csv("problem2.csv", parse_dates=["Date"])

# 1. Arithmetic returns: r_t = P_t / P_{t-1} - 1  (500 returns from 501 prices)
rets = df[["SPY", "XYZ"]].pct_change().dropna()
r_m = rets["SPY"].to_numpy()
r_x = rets["XYZ"].to_numpy()
n = len(r_m)

# 2. OLS fit of the market model
X = np.column_stack([np.ones(n), r_m])
(alpha, beta), *_ = np.linalg.lstsq(X, r_x, rcond=None)
resid = r_x - (alpha + beta * r_m)

# 3. Normal parameters
mu_m = r_m.mean()
sd_m = r_m.std(ddof=1)
sd_eps = np.sqrt(resid @ resid / (n - 2))   # 2 parameters estimated

# 4. Implied distribution of r_XYZ (sum of independent normals is normal)
mu_x = alpha + beta * mu_m
sd_x = np.sqrt(beta**2 * sd_m**2 + sd_eps**2)

print(f"n returns      = {n}")
print(f"alpha          = {alpha:.6e}")
print(f"beta           = {beta:.6f}")
print(f"R^2            = {1 - resid.var() / r_x.var():.4f}")
print(f"SPY mean, sd   = {mu_m:.6e}, {sd_m:.6e}")
print(f"residual sd    = {sd_eps:.6e}")
print(f"XYZ mean, sd   = {mu_x:.6e}, {sd_x:.6e}\n")

# 5. VaR and ES (as a percent of current position value)
shares = 100
position_value = shares * df["XYZ"].iloc[-1]
print(f"Position value = 100 x {df['XYZ'].iloc[-1]:.4f} = ${position_value:,.2f}\n")

rows = []
for conf in (0.95, 0.99):
    var, es = normal_var_es(mu_x, sd_x, 1 - conf)
    rows.append({
        "Confidence": f"{conf:.0%}",
        "VaR %": var * 100,
        "ES %": es * 100,
        "VaR $": var * position_value,
        "ES $": es * position_value,
    })
print(pd.DataFrame(rows).round(4).to_string(index=False))





