"""
Problem 4 - A Sharpe Ratio That Sells Insurance.

Usage:
    python problem4.py problem4.csv
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from p4_lib import (
    moments,
    periodic_rf,
    sharpe_annual,
    leave_one_out_sharpe,
    moment_share_of_obs,
    es_historical,
    es_normal,
    max_sharpe_long_only,
    max_sharpe,
    efficient_frontier,
    risk_table,
    ols,
    ols_multi,
    se_mean,
    bootstrap_max_sharpe,
    return_to_es,
)


# ============================================================
# Settings
# ============================================================

RF = 0.03                         # Annual risk-free rate
ALPHA = 0.01                      # Confidence level for 1% ES
N_BOOT = 1000                     # Number of bootstrap resamples


# ============================================================
# Load and Prepare Data
# ============================================================

# Read the input file from the command line.
# Use problem4.csv as the default if no filename is provided.
path = sys.argv[1] if len(sys.argv) > 1 else "problem4.csv"

df = pd.read_csv(
    path,
    parse_dates=["Month"],
    index_col="Month"
)

assets = df.columns.tolist()
R = df.values
n = len(assets)

# Convert the annual risk-free rate to a monthly rate.
rf_m = periodic_rf(RF)


# ============================================================
# (a) First Four Moments and Annualized Sharpe Ratio
# ============================================================

print("(a) Moments and Annualized Sharpe Ratios")

rows = {}

for asset in assets:

    # Calculate the first four monthly return moments.
    result = moments(df[asset])

    # Add the annualized Sharpe ratio.
    result["Sharpe_ann"] = sharpe_annual(df[asset], RF)

    rows[asset] = result

# Display the results in one table.
tab = pd.DataFrame(rows).T

print(tab.round(4).to_string())


# ------------------------------------------------------------
# (a) Influence of Individual Months
# ------------------------------------------------------------

print("\n(a) Single-Month Influence")

for asset in assets:

    returns = df[asset].values

    # Identify the worst-performing month.
    worst_index = returns.argmin()

    # Calculate the Sharpe ratio after removing each month,
    # one at a time.
    loo = leave_one_out_sharpe(returns, RF)

    # Recalculate excess kurtosis after removing the worst month.
    kurtosis_without_worst = moments(
        np.delete(returns, worst_index)
    )["exkurt"]

    print(
        f"{asset:5s} "
        f"Worst: {df.index[worst_index].date()} = "
        f"{returns[worst_index]:+.4f} | "
        f"Excess kurtosis: {tab.loc[asset, 'exkurt']:.2f} "
        f"-> {kurtosis_without_worst:.2f} | "
        f"Sharpe range: {loo.min():.3f}-{loo.max():.3f}"
    )


# ------------------------------------------------------------
# (a) Additional Analysis of PUTW
# ------------------------------------------------------------

putw = df["PUTW"].values

# Identify the worst PUTW return.
worst_index = putw.argmin()

print("\nPUTW Detail")

# Display the five most frequent monthly return values.
print(pd.Series(putw).round(4).value_counts().head(5).to_string())

# Count the number of negative-return months.
print(f"Months PUTW < 0: {(putw < 0).sum()} of {len(putw)}")

# Measure how much the worst month contributes to the
# third and fourth central moments.
print(
    f"Worst month share of 3rd central moment: "
    f"{moment_share_of_obs(putw, worst_index, 3):.1%}"
)

print(
    f"Worst month share of 4th central moment: "
    f"{moment_share_of_obs(putw, worst_index, 4):.1%}"
)

# Check how the Sharpe ratio changes after removing the worst month.
print(
    f"Sharpe with worst month removed: "
    f"{sharpe_annual(np.delete(putw, worst_index), RF):.3f}"
)


# ============================================================
# (b) Long-Only Maximum-Sharpe Portfolio and 1% ES
# ============================================================

# Find the portfolio weights that maximize the Sharpe ratio
# subject to long-only constraints.
w = max_sharpe_long_only(R, rf_m)

# Calculate the portfolio's monthly returns.
port = R @ w

print("\n(b) Long-Only Maximum-Sharpe Weights")

for asset, weight in zip(assets, w):
    print(f"  {asset:5s} {weight:7.2%}")


# Calculate portfolio moments and annualized Sharpe ratio.
pm = moments(port)

print(
    f"Portfolio Sharpe (ann): {sharpe_annual(port, RF):.3f}, "
    f"Skew: {pm['skew']:.2f}, "
    f"Excess kurtosis: {pm['exkurt']:.2f}"
)

# Compare historical and normal-distribution 1% Expected Shortfall.
es_h = es_historical(port, ALPHA)
es_n = es_normal(pm["mean"], pm["std"], ALPHA)

print(
    f"ES 1% historical: {es_h:.4f} | "
    f"ES 1% normal: {es_n:.4f} | "
    f"Ratio: {es_h / es_n:.2f}"
)

# Display the three worst portfolio months.
print("Worst 3 Portfolio Months:")

for j in np.argsort(port)[:3]:
    print(f"  {df.index[j].date()}  {port[j]:+.4f}")


# ============================================================
# (c) Maximum-Sharpe Portfolios, Efficient Frontier, and CML
# ============================================================

# Set portfolio weight constraints.
# The third portfolio additionally limits PUTW to 20%.
cap_bounds = [(0, 1)] * n
cap_bounds[assets.index("PUTW")] = (0, 0.20)

# Calculate maximum-Sharpe weights under three different constraints.
W = pd.DataFrame({
    "Unconstrained": max_sharpe(R, rf_m),
    "Long only": max_sharpe(R, rf_m, bounds=[(0, 1)] * n),
    "Long only, PUTW<=20%": max_sharpe(
        R, rf_m, bounds=cap_bounds
    ),
}, index=assets)

print("\n(c) Maximum-Sharpe Portfolio Weights")
print((W * 100).round(2).to_string())


# Calculate monthly returns for each optimized portfolio.
ports = {
    name: R @ W[name].values
    for name in W.columns
}


# ------------------------------------------------------------
# Efficient Frontier and Capital Market Line
# ------------------------------------------------------------

# Calculate the long-only efficient frontier.
f_mu, f_sd, _ = efficient_frontier(R, n_points=80)

# Annualize individual asset returns and standard deviations.
mu_a = R.mean(axis=0) * 12
sd_a = R.std(axis=0, ddof=1) * np.sqrt(12)

# Calculate annualized return and risk of the long-only
# maximum-Sharpe (tangency) portfolio.
tan = ports["Long only"]

tan_mu = tan.mean() * 12
tan_sd = tan.std(ddof=1) * np.sqrt(12)

# Calculate the portfolio with the additional PUTW constraint.
cap = ports["Long only, PUTW<=20%"]

# Annualized risk-free rate.
rf_a = rf_m * 12

# Construct the Capital Market Line (CML).
sd_line = np.linspace(
    0,
    max(sd_a.max(), f_sd.max() * np.sqrt(12)) * 1.05,
    100
)

cml = rf_a + (tan_mu - rf_a) / tan_sd * sd_line


# ------------------------------------------------------------
# Plot Efficient Frontier and CML
# ------------------------------------------------------------

ink = "#1f1f1e"
muted = "#6b6a63"
grid = "#e6e5df"

fig, ax = plt.subplots(figsize=(8, 5.5))

# Plot the annualized long-only efficient frontier.
ax.plot(
    f_sd * np.sqrt(12),
    f_mu * 12,
    color="#2a78d6",
    lw=2,
    label="Long-only efficient frontier"
)

# Plot the Capital Market Line.
ax.plot(
    sd_line,
    cml,
    color="#eb6834",
    lw=2,
    ls="--",
    label="Capital market line"
)

# Plot individual assets.
ax.scatter(
    sd_a,
    mu_a,
    s=60,
    color=ink,
    zorder=3,
    label="Assets"
)

# Label each asset.
for asset, x, y in zip(assets, sd_a, mu_a):
    ax.annotate(
        asset,
        (x, y),
        xytext=(6, -12 if asset == "PUTW" else 6),
        textcoords="offset points",
        color=ink,
        fontsize=10
    )

# Highlight the long-only tangency portfolio.
ax.scatter(
    [tan_sd],
    [tan_mu],
    s=320,
    marker="*",
    color="#eb6834",
    edgecolor="white",
    linewidth=1.5,
    zorder=4,
    label="Long-only tangency"
)

# Highlight the tangency portfolio with PUTW capped at 20%.
ax.scatter(
    [cap.std(ddof=1) * np.sqrt(12)],
    [cap.mean() * 12],
    s=90,
    marker="D",
    color="#1baf7a",
    edgecolor="white",
    linewidth=1.5,
    zorder=4,
    label="Tangency, PUTW <= 20%"
)

# Mark the risk-free asset.
ax.scatter([0], [rf_a], s=40, color=muted, zorder=3)

ax.annotate(
    "rf",
    (0, rf_a),
    xytext=(6, -12),
    textcoords="offset points",
    color=muted
)

# Add labels, title, and percentage formatting.
ax.set_xlabel("Annualized standard deviation")
ax.set_ylabel("Annualized mean return")
ax.set_title(
    "Long-only efficient frontier and capital market line",
    loc="left"
)

ax.xaxis.set_major_formatter(
    plt.FuncFormatter(lambda v, _: f"{v:.0%}")
)

ax.yaxis.set_major_formatter(
    plt.FuncFormatter(lambda v, _: f"{v:.0%}")
)

ax.set_xlim(left=0)
ax.grid(color=grid, lw=0.8)

# Remove the top and right borders.
for side in ("top", "right"):
    ax.spines[side].set_visible(False)

for side in ("left", "bottom"):
    ax.spines[side].set_color(muted)

ax.tick_params(colors=muted)
ax.legend(frameon=False, loc="lower right")

fig.tight_layout()
plt.show()


# ============================================================
# (d) Sharpe Ratio, Historical and Normal VaR / ES
# ============================================================

# Compare the three optimized portfolios using:
# 1. Annualized Sharpe ratio
# 2. Historical VaR and ES
# 3. Normal-distribution VaR and ES

rows = {}

for name, returns in ports.items():

    result = {
        "Sharpe_ann": sharpe_annual(returns, RF)
    }

    result.update(
        risk_table(returns, alphas=(0.05, 0.01))
    )

    rows[name] = result

risk = pd.DataFrame(rows).T

print("\n(d) Sharpe and Monthly VaR / ES (Positive = Loss)")
print(risk.round(4).T.to_string())


# ============================================================
# (e) CAPM Regression: PUTW Excess Returns on EQ Excess Returns
# ============================================================

# Convert both asset returns into excess returns.
y = df["PUTW"].values - rf_m
x = df["EQ"].values - rf_m

# Estimate the CAPM using the full sample.
full = ols(y, x)

print("\n(e) CAPM: PUTW Excess on EQ Excess (All Months)")

print(
    f"  Alpha: {full['alpha']:.5f} per month "
    f"({full['alpha'] * 12:.2%} per year), "
    f"t = {full['t_alpha']:.2f}"
)

print(
    f"  Beta: {full['beta']:.4f}, "
    f"t = {full['t_beta']:.2f}"
)

print(f"  R-squared: {full['r2']:.4f} | n = {full['n']}")


# ------------------------------------------------------------
# (e) CAPM During Significant EQ Down Months
# ------------------------------------------------------------

# Identify months when EQ excess return is below -5%.
down = x < -0.05

# Estimate CAPM using only these down months.
dn = ols(y[down], x[down])

print(f"\n(e) EQ Excess Returns < -5% (n = {dn['n']})")

print(
    f"  Alpha: {dn['alpha']:.5f}, "
    f"t = {dn['t_alpha']:.2f}"
)

print(
    f"  Beta: {dn['beta']:.4f}, "
    f"t = {dn['t_beta']:.2f}"
)

print(f"  R-squared: {dn['r2']:.4f}")


# Estimate CAPM for all other months.
up = ols(y[~down], x[~down])

print(
    f"\n(e) Other Months (n = {up['n']}): "
    f"Beta = {up['beta']:.4f}, "
    f"R-squared = {up['r2']:.4f}"
)


# ============================================================
# (f) Return per Unit of ES and ES Subadditivity
# ============================================================

# Compare each asset's and portfolio's mean excess return
# relative to its historical 1% Expected Shortfall.

print("\n(f) Mean Excess Return / Historical ES 1% (Monthly)")

for asset in assets:
    print(
        f"  {asset:22s} "
        f"{return_to_es(df[asset].values, rf_m):.3f}"
    )

for name, returns in ports.items():
    print(
        f"  {name:22s} "
        f"{return_to_es(returns, rf_m):.3f}"
    )


# ------------------------------------------------------------
# (f) Check ES Subadditivity
# ------------------------------------------------------------

# Calculate historical 1% ES and VaR for individual assets.
es_assets = np.array([
    es_historical(df[asset].values, 0.01)
    for asset in assets
])

var_assets = np.array([
    -np.quantile(df[asset].values, 0.01)
    for asset in assets
])

print(
    "\n(f) Subadditivity Check: "
    "Portfolio ES vs. Weighted Sum of Individual ES"
)

for name in W.columns:

    weights = W[name].values

    # Skip portfolios containing short positions.
    if (weights < 0).any():
        continue

    portfolio_returns = ports[name]

    # Compare portfolio ES with the weighted sum of asset ES.
    portfolio_es = es_historical(portfolio_returns, 0.01)
    weighted_es = weights @ es_assets

    # Also compare portfolio VaR with the weighted sum of asset VaR.
    portfolio_var = -np.quantile(portfolio_returns, 0.01)
    weighted_var = weights @ var_assets

    print(
        f"  {name:22s} "
        f"ES: {portfolio_es:.4f} <= {weighted_es:.4f} | "
        f"VaR: {portfolio_var:.4f} vs. {weighted_var:.4f}"
    )


# ============================================================
# (g) CAPM with an Option-Shaped Downside Term
# ============================================================

# Create a nonlinear term that becomes negative when EQ return
# falls below -5%.
#
# Model:
# PUTW_ex = alpha + beta * EQ_ex + c * min(EQ + 5%, 0)

kink = np.minimum(df["EQ"].values + 0.05, 0)

# Estimate the multiple regression with two explanatory variables:
# EQ excess return and the downside option-shaped term.
g = ols_multi(
    y,
    np.column_stack([x, kink])
)

print("\n(g) PUTW Excess on EQ Excess and Downside Term")

# Display alpha, market beta, and the downside-term coefficient.
for label, coef, t_stat in zip(
    ["Alpha", "Beta EQ", "Put term"],
    g["coef"],
    g["t"]
):
    print(
        f"  {label:9s} {coef:+.5f} "
        f"t = {t_stat:.2f}"
    )

print(f"  R-squared: {g['r2']:.4f}")

# The downside slope is the sum of the regular EQ beta
# and the coefficient on the downside term.
downside_slope = g["coef"][1] + g["coef"][2]

print(f"  Downside slope: {downside_slope:.4f}")


# ============================================================
# (h) Standard Errors of Means and Bootstrap of Tangency Weights
# ============================================================

# Calculate the mean return, standard error, 95% confidence interval,
# and t-statistic for the monthly excess mean of each asset.

print("\n(h) Mean Monthly Return, Standard Error, and t-Statistic")

for asset in assets:

    returns = df[asset].values

    se = se_mean(returns)
    mean_return = returns.mean()

    # Calculate the 95% confidence interval for the sample mean.
    lower = mean_return - 1.96 * se
    upper = mean_return + 1.96 * se

    # Test whether the average excess return differs from zero.
    t_stat = (mean_return - rf_m) / se

    print(
        f"  {asset:5s} "
        f"Mean: {mean_return:.5f} "
        f"SE: {se:.5f} "
        f"95% CI: [{lower:+.5f}, {upper:+.5f}] "
        f"t(excess): {t_stat:.2f}"
    )


# ------------------------------------------------------------
# (h) Bootstrap Maximum-Sharpe Portfolio Weights
# ------------------------------------------------------------

# Locate June 2008 to examine the effect of including this month
# in the bootstrap samples.
i_jun08 = df.index.get_loc(pd.Timestamp("2008-06-30"))

# Run the bootstrap under the same three portfolio constraints
# used in part (c).
for name, bounds in [
    ("Unconstrained", None),
    ("Long only", [(0, 1)] * n),
    ("Long only, PUTW<=20%", cap_bounds)
]:

    # Generate bootstrap estimates of the optimal portfolio weights.
    # Each resample contains 240 monthly observations.
    Wb, ix = bootstrap_max_sharpe(
        R,
        rf_m,
        bounds,
        n_boot=N_BOOT,
        seed=545
    )

    # Summarize the bootstrap distribution of each asset's weight.
    summ = pd.DataFrame({
        "Point Estimate": W[name].values,
        "Bootstrap Mean": Wb.mean(axis=0),
        "Bootstrap SD": Wb.std(axis=0, ddof=1),
        "5%": np.percentile(Wb, 5, axis=0),
        "95%": np.percentile(Wb, 95, axis=0),
        "P(w < 1%)": (Wb < 0.01).mean(axis=0),
    }, index=assets)

    print(
        f"\n(h) Bootstrap ({N_BOOT} resamples of 240 months): "
        f"{name}"
    )

    # Convert portfolio weights into percentages for display.
    print((summ * 100).round(1).to_string())

    # For the long-only portfolio, examine whether including
    # June 2008 changes the average PUTW allocation.
    if name == "Long only":

        has_june_2008 = (ix == i_jun08).any(axis=1)

        putw_index = assets.index("PUTW")

        print(
            f"  Resamples without June 2008: "
            f"{(~has_june_2008).mean():.1%}; "
            f"Mean PUTW weight without: "
            f"{Wb[~has_june_2008, putw_index].mean():.1%} "
            f"| With: "
            f"{Wb[has_june_2008, putw_index].mean():.1%}"
        )
