"""
Problem 3: Portfolio Risk Analysis

This script covers:
(a)-(b) Portfolio Greeks and 10-day P&L predictions.
(c)    Moments of daily stock returns and distribution fitting.
(d)    Delta-normal VaR and ES.
(e)    Monte Carlo simulation with full portfolio revaluation.
(f)    Comparison of full revaluation and delta-only P&L.
(g)    Delta, gamma, and theta approximations.
(h)    Comparison of predicted and simulated mean P&L.
(i)    Comparison of Normal and Student t simulation results.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from p3_lib import (
    four_moments,
    fit_normal,
    fit_t,
    var_es,
    delta_normal_var_es,
    position_greeks,
    position_value,
    delta_normal_var,
    book_value,
    taylor_pnl,
)


# ============================================================
# 1. Basic settings and data
# ============================================================

R = 0.0425          # Annual risk-free interest rate
DPY = 255           # Trading days per year
H = 10              # Holding period in trading days
NSIM = 100_000      # Number of Monte Carlo simulations
SEED = 42           # Random seed for reproducibility

ALPHAS = [0.05, 0.01]   # Confidence levels for VaR and ES

pd.set_option("display.width", 200)
pd.set_option("display.float_format", "{:,.4f}".format)

# Load historical stock prices and portfolio positions.
prices = pd.read_csv("problem3.csv", parse_dates=["Date"])
port = pd.read_csv("problem3_portfolios.csv")

# Current stock price.
S0 = prices["ABC"].iloc[-1]

# Calculate daily arithmetic returns, r_t = P_t / P_{t-1} - 1.
# Convention: the assignment requires arithmetic returns, so this one
# series is used in EVERY part of Problem 3: the volatility behind the
# predictions in (a)-(b), the delta-normal VaR/ES in (d), and the
# distribution fitting and simulation in (c) and (e).
rets = prices["ABC"].pct_change().dropna().values
n = len(rets)

# Daily volatility of arithmetic returns (sample std, ddof=1).
# Computed once here and reused in (a)-(b) and (d), so the delta-normal
# VaR in the part (a) table matches part (d) exactly.
sig_d = rets.std(ddof=1)

# Separate positions into individual portfolios.
books = {
    name: book
    for name, book in port.groupby("Portfolio", sort=False)
}


# ============================================================
# 2. Parts (a)-(b): Greeks and 10-day P&L predictions
# ============================================================

# The historical volatility sig_d comes from the arithmetic returns
# computed in section 1 (not log returns), as the assignment requires.

# Estimate the one-standard-deviation stock price movement
# over the 10-day holding period.
dS = S0 * sig_d * np.sqrt(H)

print("=== Parts (a)-(b): Portfolio Greeks and predictions ===")
print(
    f"Current stock price: {S0:.2f}\n"
    f"Daily arithmetic-return volatility: {sig_d:.4%}\n"
    f"Annualized volatility: {sig_d * np.sqrt(DPY):.2%}\n"
    f"10-day one-standard-deviation price move: {dS:.2f}\n"
)

rows = []

for name, book in books.items():

    # Calculate the Greeks for each position and sum them
    # to obtain the total portfolio delta, gamma, and theta.
    g = pd.DataFrame([
        position_greeks(r, S0, R, DPY)
        for _, r in book.iterrows()
    ]).sum()

    # Calculate the current portfolio value.
    v0 = sum(
        position_value(r, S0, R, DPY)
        for _, r in book.iterrows()
    )

    # Calculate the portfolio value after 10 days,
    # assuming the stock price remains unchanged.
    v10 = sum(
        position_value(r, S0, R, DPY, shift_days=H)
        for _, r in book.iterrows()
    )

    rows.append({
        "Book": name,
        "Value": v0,

        # Portfolio Greeks.
        "Delta": g["delta"],
        "Gamma": g["gamma"],
        "Theta/yr": g["theta"],
        "Theta/day": g["theta"] / DPY,
        "Theta 10d": g["theta"] * H / DPY,

        # Exact P&L caused by the passage of time,
        # with the underlying stock price held constant.
        "Unchanged-price P&L 10d": v10 - v0,

        # 95% delta-normal VaR.
        "DeltaNormal VaR95": delta_normal_var(
            g["delta"], S0, sig_d, H
        ),

        # Absolute first-order delta P&L for a 1-sd price move.
        "|Delta|*dS": abs(g["delta"]) * dS,

        # Second-order gamma P&L for a 1-sd price move.
        "0.5*Gamma*dS^2": 0.5 * g["gamma"] * dS**2,
    })

res = pd.DataFrame(rows).set_index("Book")

# Compare the size of the gamma contribution with the delta contribution.
# A larger ratio means gamma has a greater relative impact.
res["Gamma/Delta ratio"] = (
    res["0.5*Gamma*dS^2"].abs()
    / res["|Delta|*dS"].replace(0, np.nan)
)

print(res.T.to_string())


# ============================================================
# 3. Part (c): Return moments and distribution fitting
# ============================================================

print(f"\n=== Part (c): Moments of {n} daily arithmetic returns ===")

# Calculate the four moments: mean, standard deviation,
# skewness, and excess kurtosis.
print(pd.Series(four_moments(rets)).to_string(), "\n")

# Fit Normal and Student t distributions to historical returns.
fn = fit_normal(rets)
ft = fit_t(rets)

# Compare the estimated parameters and model fit statistics.
fits = pd.DataFrame({
    "Normal": {
        "mu": fn["mu"],
        "sigma (scale)": fn["sigma"],
        "nu": np.nan,
        "std dev": fn["sigma"],
        "loglik": fn["loglik"],
        "AICc": fn["aicc"],
    },
    "Student t": {
        "mu": ft["mu"],
        "sigma (scale)": ft["sigma"],
        "nu": ft["nu"],
        "std dev": ft["sigma"] * np.sqrt(
            ft["nu"] / (ft["nu"] - 2)
        ),
        "loglik": ft["loglik"],
        "AICc": ft["aicc"],
    },
})

print(fits.to_string())

# Select the distribution with the lower AICc.
best = "Student t" if ft["aicc"] < fn["aicc"] else "Normal"

print(
    f"\nAICc difference (Normal - Student t) = "
    f"{fn['aicc'] - ft['aicc']:.2f}"
    f" -> Choose {best}\n"
)


# ============================================================
# 4. Part (d): Delta-normal VaR and ES
# ============================================================

# Use the same arithmetic-return volatility sig_d defined in section 1,
# so (a) and (d) are based on one consistent volatility estimate.

rows = []

for name, book in books.items():

    # Calculate total portfolio delta.
    delta = sum(
        position_greeks(r, S0, R, DPY)["delta"]
        for _, r in book.iterrows()
    )

    row = {"Book": name, "Delta": delta}

    # Calculate VaR and ES at the 5% and 1% tail levels.
    for a in ALPHAS:
        v, e = delta_normal_var_es(delta, S0, sig_d, H, a)

        row[f"VaR {a:.0%}"] = v
        row[f"ES {a:.0%}"] = e

    rows.append(row)

dn = pd.DataFrame(rows).set_index("Book")

print(
    f"=== Part (d): 10-day delta-normal VaR / ES "
    f"(daily volatility = {sig_d:.4%}) ==="
)
print(dn.to_string(float_format="{:,.2f}".format), "\n")


# ============================================================
# 5. Part (e): Monte Carlo simulation with full revaluation
# ============================================================

rng = np.random.default_rng(SEED)

# Simulate 10 days of daily returns under each fitted distribution.
# Both distributions use zero mean, so the simulation focuses
# on the impact of volatility and tail behavior.
sims = {
    "Normal": rng.normal(
        0.0, fn["sigma"], size=(NSIM, H)
    ),

    "Student t": stats.t.rvs(
        ft["nu"],
        loc=0.0,
        scale=ft["sigma"],
        size=(NSIM, H),
        random_state=rng,
    ),
}

# Compound daily arithmetic returns to obtain the stock price
# at the end of the 10-day holding period.
S10 = {
    dist: S0 * np.prod(1 + daily_returns, axis=1)
    for dist, daily_returns in sims.items()
}

rows = []
pnl_store = {}

for name, book in books.items():

    # Current portfolio value.
    v0 = book_value(book, S0, R, DPY)

    for dist, stock_prices in S10.items():

        # Fully revalue the portfolio at each simulated stock price.
        # Also account for the passage of 10 trading days.
        pnl = (
            book_value(book, stock_prices, R, DPY, shift_days=H)
            - v0
        )

        # Save simulated P&L for later comparisons and plots.
        pnl_store[(name, dist)] = pnl

        row = {
            "Book": name,
            "Dist": dist,
            "Mean P&L": pnl.mean(),
        }

        # Calculate VaR and ES at both tail levels.
        for a in ALPHAS:
            v, e = var_es(pnl, a)

            row[f"VaR {a:.0%}"] = v
            row[f"ES {a:.0%}"] = e

        # Compare expected shortfall with VaR at the 5% level.
        row["ES/VaR 5%"] = row["ES 5%"] / row["VaR 5%"]

        rows.append(row)

mc = pd.DataFrame(rows).set_index(["Book", "Dist"])

print(
    f"=== Part (e): Monte Carlo full revaluation "
    f"({NSIM:,} paths, {H} days, zero mean) ==="
)
print(mc.to_string(float_format="{:,.2f}".format), "\n")


# ============================================================
# 6. Compare delta-normal estimates with Monte Carlo results
# ============================================================

# Compare 5% VaR and 1% ES from the different approaches.
cmp = pd.DataFrame({
    "Delta-normal VaR 5%": dn["VaR 5%"],

    "MC VaR 5% (Normal)": (
        mc.xs("Normal", level="Dist")["VaR 5%"]
    ),

    "MC VaR 5% (t)": (
        mc.xs("Student t", level="Dist")["VaR 5%"]
    ),

    "Delta-normal ES 1%": dn["ES 1%"],

    "MC ES 1% (t)": (
        mc.xs("Student t", level="Dist")["ES 1%"]
    ),
})

print("=== Delta-normal vs full revaluation ===")
print(cmp.to_string(float_format="{:,.2f}".format))


# ============================================================
# 7. Part (f): Full revaluation vs delta-only P&L
# ============================================================

BLUE = "#2a78d6"
ORANGE = "#eb6834"
INK = "#0b0b0b"
MUTED = "#52514e"

# Use the distribution selected by AICc in part (c).
dist = best

fig, axes = plt.subplots(2, 2, figsize=(12, 9))

# Select 5,000 simulation paths to make the scatter plots easier to read.
idx = rng.choice(NSIM, 5000, replace=False)

for ax, (name, book) in zip(axes.ravel(), books.items()):

    # Simulated stock prices and full-revaluation P&L.
    s = S10[dist][idx]
    pnl = pnl_store[(name, dist)][idx]

    # Portfolio delta.
    delta = dn.loc[name, "Delta"]

    # Create a range of stock prices for the delta-only line.
    grid = np.linspace(s.min(), s.max(), 200)

    # Plot fully revalued P&L.
    ax.scatter(
        s, pnl,
        s=8,
        color=BLUE,
        alpha=0.35,
        linewidths=0,
        label="Full revaluation",
    )

    # Plot the first-order delta approximation.
    ax.plot(
        grid,
        delta * (grid - S0),
        color=ORANGE,
        lw=2,
        label="Delta-only",
    )

    # Mark zero P&L and the current stock price.
    ax.axhline(0, color=MUTED, lw=0.6)
    ax.axvline(S0, color=MUTED, lw=0.6, ls=":")

    ax.set_title(
        f"{name} (Delta = {delta:.1f})",
        color=INK,
        fontsize=12,
        loc="left",
    )

    ax.set_xlabel("ABC price in 10 days", color=MUTED)
    ax.set_ylabel("10-day P&L ($)", color=MUTED)

    # Simplify the plot appearance.
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    ax.grid(alpha=0.2)
    ax.legend(frameon=False, fontsize=9)

fig.suptitle(
    f"10-day P&L vs ABC price, {dist} simulation "
    f"(S0 = {S0:.2f})",
    fontsize=14,
    color=INK,
)

fig.tight_layout()


# ============================================================
# 8. Part (g): Taylor approximations vs full revaluation
# ============================================================

# Calculate and store the total Greeks for each portfolio.
greeks = {}

for name, book in books.items():

    greeks[name] = pd.DataFrame([
        position_greeks(r, S0, R, DPY)
        for _, r in book.iterrows()
    ]).sum()

rows = []

for name in books:

    g = greeks[name]

    for dist, stock_prices in S10.items():

        # Stock price changes across all simulated paths.
        dS = stock_prices - S0

        # Compare increasingly detailed Taylor approximations:
        # 1. Delta only.
        # 2. Delta and gamma.
        # 3. Delta, gamma, and theta.
        # 4. Full portfolio revaluation.
        approx = {
            "Delta": taylor_pnl(
                dS, g["delta"]
            ),

            "Delta+Gamma": taylor_pnl(
                dS, g["delta"], g["gamma"]
            ),

            "Delta+Gamma+Theta": taylor_pnl(
                dS,
                g["delta"],
                g["gamma"],
                g["theta"],
                H / DPY,
            ),

            "Full": pnl_store[(name, dist)],
        }

        # Calculate the mean P&L, VaR, and ES for each method.
        for method, pnl in approx.items():

            v5, e5 = var_es(pnl, 0.05)
            v1, e1 = var_es(pnl, 0.01)

            rows.append({
                "Book": name,
                "Dist": dist,
                "Method": method,
                "Mean": pnl.mean(),
                "VaR 5%": v5,
                "ES 5%": e5,
                "VaR 1%": v1,
                "ES 1%": e1,
            })

tay = pd.DataFrame(rows).set_index(
    ["Book", "Dist", "Method"]
)

print(
    "\n=== Part (g): Taylor approximations vs full revaluation "
    "(same simulated paths) ==="
)

# Display the Student t results for comparison.
print(
    tay.xs("Student t", level="Dist").to_string(
        float_format="{:,.2f}".format
    )
)


# ============================================================
# 9. Part (h): Mean P&L vs unchanged-price prediction
# ============================================================

rows = []

for name, book in books.items():

    g = greeks[name]

    # Current portfolio value.
    v0 = book_value(book, S0, R, DPY)

    # Exact P&L after 10 days if the stock price does not change.
    flat = (
        book_value(book, S0, R, DPY, shift_days=H) - v0
    )

    # Estimate E[(dS)^2] using the Normal simulation.
    # The simulated returns have zero mean.
    var_dS = (S10["Normal"] - S0).var()

    # Compare theta, gamma, and Monte Carlo mean P&L.
    rows.append({
        "Book": name,

        "Theta 10d": g["theta"] * H / DPY,

        "Unchanged-price P&L": flat,

        "Gamma cost 0.5*G*E[dS^2]": (
            0.5 * g["gamma"] * var_dS
        ),

        "Theta + gamma cost": (
            g["theta"] * H / DPY
            + 0.5 * g["gamma"] * var_dS
        ),

        "MC mean (Normal)": mc.loc[
            (name, "Normal"), "Mean P&L"
        ],

        "MC mean (t)": mc.loc[
            (name, "Student t"), "Mean P&L"
        ],
    })

print("\n=== Part (h): Mean P&L vs part (b) prediction ===")

print(
    pd.DataFrame(rows).set_index("Book").to_string(
        float_format="{:,.2f}".format
    )
)


# ============================================================
# 10. Part (i): Compare Normal and Student t results
# ============================================================

# Calculate the percentage change in each risk measure
# when moving from the Normal distribution to Student t.
normal_mc = mc.xs("Normal", level="Dist")
t_mc = mc.xs("Student t", level="Dist")

chg = (t_mc / normal_mc - 1) * 100

print("\n=== Part (i): Percentage change from Normal to Student t ===")

print(
    chg[["VaR 5%", "ES 5%", "VaR 1%", "ES 1%"]].to_string(
        float_format="{:+.1f}%".format
    )
)

# Also compare the simulated 10-day return distributions.
for dist in ("Normal", "Student t"):

    r10 = S10[dist] / S0 - 1

    print(
        f"{dist:9s} 10-day return: "
        f"sd = {r10.std():.4f}, "
        f"5% q = {np.quantile(r10, 0.05):.4f}, "
        f"1% q = {np.quantile(r10, 0.01):.4f}, "
        f"excess kurt = {stats.kurtosis(r10):.3f}"
    )


# ============================================================
# 11. Display the figure from part (f)
# ============================================================

# Show the figure after all numerical results have been printed.
plt.show()