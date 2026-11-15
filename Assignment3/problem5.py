import numpy as np
import pandas as pd

import p5_lib as fl


# ============================================================
# Settings
# ============================================================

ANN = 252                         # Trading days per year
NOTIONAL = 10_000_000              # Portfolio value ($10 million)
FUNDS = ["FUND1", "FUND2"]

# Factor specifications for each regression model
MODELS = {
    "CAPM": ["Mkt-RF"],
    "FF3": ["Mkt-RF", "SMB", "HML"],
    "Carhart4": ["Mkt-RF", "SMB", "HML", "Mom"],
}

F4 = MODELS["Carhart4"]


# ============================================================
# Load and Prepare Data
# ============================================================

# Load fund returns and Fama-French daily factors
funds = fl.load_funds("problem5.csv")

factors = fl.load_ff("F-F_Research_Data_Factors_daily.CSV").join(
    fl.load_ff("F-F_Momentum_Factor_daily.CSV"),
    how="inner"
)

# Keep dates available in both datasets
data = funds.join(factors, how="inner")

# Calculate fund excess returns: fund return minus risk-free rate
excess = data[FUNDS].sub(data["RF"], axis=0)

print(
    "Sample:",
    data.index[0].date(),
    "to",
    data.index[-1].date(),
    "| Observations:",
    len(data)
)


# ============================================================
# (a) Summary Statistics and CAPM Regression
# ============================================================

print("\n(a) Summary Statistics and CAPM")

# Display annualized summary statistics for fund excess returns
print(fl.summary(excess, ANN))

# Estimate CAPM separately for each fund
capm = {}

for fund in FUNDS:
    result = fl.ols(excess[fund], data[["Mkt-RF"]])
    capm[fund] = result

    alpha = result["coef"]["alpha"] * ANN
    alpha_t = result["t"]["alpha"]
    beta = result["coef"]["Mkt-RF"]
    r_squared = result["r2"]

    print(
        fund,
        "| Annual alpha:", round(alpha, 4),
        "| Alpha t-stat:", round(alpha_t, 2),
        "| Market beta:", round(beta, 3),
        "| R-squared:", round(r_squared, 3)
    )


# ============================================================
# (b) Correlation Between CAPM Residuals and Other Factors
# ============================================================

print("\n(b) Residual Correlations")

# Check whether CAPM residuals are correlated with omitted factors
# A high correlation may indicate that these factors explain
# additional variation not captured by the CAPM.
corr = pd.DataFrame({
    fund: data[["SMB", "HML", "Mom"]].corrwith(capm[fund]["resid"])
    for fund in FUNDS
})

print(corr)

# Display annualized average returns of all Carhart four factors
print("\nAnnualized Factor Means (Full Sample):")
print(data[F4].mean() * ANN)


# ============================================================
# (c) Compare CAPM, Fama-French Three-Factor, and Carhart Four-Factor
# ============================================================

print("\n(c) Regression Model Comparison")

fits = {}

for fund in FUNDS:
    fits[fund] = {}

    for model_name, factor_cols in MODELS.items():

        # Estimate the regression using the specified factor set
        result = fl.ols(excess[fund], data[factor_cols])
        fits[fund][model_name] = result

        print(f"\n{fund} - {model_name}")

        # Display estimated coefficients, standard errors, and t-statistics
        regression_table = pd.DataFrame({
            "Coefficient": result["coef"],
            "Standard Error": result["se"],
            "t-statistic": result["t"]
        })

        print(regression_table)

        # Compare annualized alpha and model fit
        print(
            "Annual alpha:", round(result["coef"]["alpha"] * ANN, 4),
            "| Alpha t-stat:", round(result["t"]["alpha"], 2),
            "| R-squared:", round(result["r2"], 4),
            "| AICc:", round(result["aicc"], 1)
        )


# ============================================================
# (d) Expected Returns Using Factor Means from the Last 10 Years
# ============================================================

print("\n(d) Expected Returns")

# Select observations within the most recent 10-year period
recent = factors[
    factors.index > factors.index[-1] - pd.DateOffset(years=10)
]

# Calculate average factor premiums and risk-free rate
prem = recent[F4].mean()
rf10 = recent["RF"].mean()

print(
    "10-Year Window:",
    recent.index[0].date(),
    "to",
    recent.index[-1].date()
)

print("\nAnnualized Factor Premia:")
print(prem * ANN)

print("\nAnnualized Risk-Free Rate:", round(rf10 * ANN, 4))

expected = {}

for fund in FUNDS:

    # Use Carhart four-factor regression coefficients
    beta = fits[fund]["Carhart4"]["coef"]

    # Expected return without alpha:
    # Risk-free rate + sum(beta_i * factor premium_i)
    no_alpha = (rf10 + (beta[F4] * prem).sum()) * ANN

    # Expected return including estimated annualized alpha
    with_alpha = no_alpha + beta["alpha"] * ANN

    expected[fund] = [no_alpha, with_alpha]

    print(
        fund,
        "| Without alpha:", round(no_alpha, 4),
        "| With alpha:", round(with_alpha, 4)
    )


# ============================================================
# (e) Fund Exposure to the Worst Momentum Day
# ============================================================

print("\n(e) Worst Momentum Day")

# Identify the date with the lowest daily momentum factor return
worst_day = factors["Mom"].idxmin()
worst = factors["Mom"].min()

print(
    "Worst Momentum Day:",
    worst_day.date(),
    "| Momentum Return:",
    round(worst, 4)
)

for fund in FUNDS:

    # Estimate the dollar impact of the momentum shock:
    # Momentum beta * worst momentum return * portfolio notional
    momentum_beta = fits[fund]["Carhart4"]["coef"]["Mom"]
    loss = momentum_beta * worst * NOTIONAL

    # Compare the absolute dollar impact with one daily standard deviation
    daily_sd = data[fund].std()
    sd_multiple = abs(loss) / (daily_sd * NOTIONAL)

    print(
        fund,
        "| Estimated Dollar Impact: $", round(loss),
        "| Impact in Daily SDs:", round(sd_multiple, 2)
    )


# ============================================================
# (f) Explain the Change in Alpha from CAPM to Carhart Four-Factor
# ============================================================

print("\n(f) Alpha Decomposition")

# Use full-sample average factor returns to measure the contribution
# from changes in factor exposures between the two models.
means = data[F4].mean()

for fund in FUNDS:

    # CAPM has only a market beta.
    # Treat the coefficients on SMB, HML, and Mom as zero.
    beta_capm = fits[fund]["CAPM"]["coef"].reindex(F4).fillna(0)

    # Factor exposures estimated by the Carhart four-factor model
    beta_carhart = fits[fund]["Carhart4"]["coef"][F4]

    # Annualized contribution from each change in factor exposure
    contribution = (
        (beta_carhart - beta_capm) * means * ANN
    )

    # Compare annualized alpha estimates across the two models
    alpha_capm = fits[fund]["CAPM"]["coef"]["alpha"] * ANN
    alpha_carhart = fits[fund]["Carhart4"]["coef"]["alpha"] * ANN

    print(
        f"\n{fund}: Alpha {alpha_capm:.4f} -> {alpha_carhart:.4f}",
        f"| Alpha Reduction: {alpha_capm - alpha_carhart:.4f}"
    )

    # Show the contribution from each factor
    decomposition = pd.DataFrame({
        "Change in Beta": beta_carhart - beta_capm,
        "Annualized Factor Mean": means * ANN,
        "Alpha Contribution": contribution
    })

    print(decomposition)

    # Total contribution from changes in factor exposures
    print("Total Contribution:", round(contribution.sum(), 4))


# ============================================================
# (h) Compare Realized Returns with Expected Returns
# ============================================================

print("\n(h) Realized vs. Expected Returns")

# Estimate the sample length in years
years = len(data) / ANN

for fund in FUNDS:

    # Annualized arithmetic average return
    realized_arithmetic = data[fund].mean() * ANN

    # Annualized geometric return (compound annual growth rate)
    realized_geometric = (
        (1 + data[fund]).prod() ** (1 / years) - 1
    )

    print(
        fund,
        "| Realized Arithmetic:", round(realized_arithmetic, 4),
        "| Realized Geometric:", round(realized_geometric, 4),
        "| Expected (No Alpha):", round(expected[fund][0], 4),
        "| Expected (With Alpha):", round(expected[fund][1], 4)
    )

# Compare annualized factor means over the full sample and recent 10 years
print("\nFactor Means: Full Sample vs. Last 10 Years")

factor_comparison = pd.DataFrame({
    "Full Sample": means * ANN,
    "Last 10 Years": prem * ANN
})

print(factor_comparison)
