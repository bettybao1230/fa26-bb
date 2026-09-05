# ==================================================
# Problem 5: Identifying an AR or MA Order
# ==================================================

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA


# ==================================================
# 1. Load the data
# ==================================================

data5 = pd.read_csv("problem5.csv")

# Extract the x series and remove missing observations.
x = data5["x"].dropna().to_numpy()

# Number of observations.
n = len(x)

print(f"Sample size: n = {n}")


# ==================================================
# 2. Plot the series, ACF, and PACF
# ==================================================

# The ACF measures the correlation between x_t and
# x_(t-k) at different lag values k.
#
# The PACF measures the relationship between x_t and
# x_(t-k) after controlling for the intermediate lags.
#
# For an AR(p) process:
#   - the ACF typically decays gradually
#   - the PACF typically cuts off after lag p
#
# For an MA(q) process:
#   - the ACF typically cuts off after lag q
#   - the PACF typically decays gradually
#
# The 95% confidence bands help determine whether the
# sample autocorrelations are statistically significant.

# Use at most 20 lags, while avoiding too many lags
# when the sample size is small.
max_lag = min(20, n // 2 - 1)

fig, axes = plt.subplots(
    3, 1,
    figsize=(10, 12)
)


# ==================================================
# 2.1. Original time series
# ==================================================

axes[0].plot(x)

axes[0].set_title(
    "Time Series of x",
    fontsize=10,
    loc="left"
)

axes[0].set_xlabel("Time")
axes[0].set_ylabel("x")

axes[0].grid(alpha=0.3)


# ==================================================
# 2.2. Autocorrelation function (ACF)
# ==================================================

plot_acf(
    x,
    lags=max_lag,
    alpha=0.05,
    ax=axes[1]
)

axes[1].set_title(
    "Autocorrelation Function (ACF)",
    fontsize=10,
    loc="left"
)

axes[1].set_xlabel("Lag")
axes[1].set_ylabel("Autocorrelation")

axes[1].grid(alpha=0.3)


# ==================================================
# 2.3. Partial autocorrelation function (PACF)
# ==================================================

plot_pacf(
    x,
    lags=max_lag,
    alpha=0.05,
    method="ywm",
    ax=axes[2]
)

axes[2].set_title(
    "Partial Autocorrelation Function (PACF)",
    fontsize=10,
    loc="left"
)

axes[2].set_xlabel("Lag")
axes[2].set_ylabel("Partial Autocorrelation")

axes[2].grid(alpha=0.3)


# ==================================================
# 3. Adjust spacing and display
# ==================================================

plt.tight_layout(pad=2.0)
plt.show()


# ==================================================
# Problem 5(c): Fit AR and MA Models
# ==================================================

# We fit:
#   AR(1), AR(2), AR(3)
#   MA(1), MA(2), MA(3)
#
# The models are compared using AICc.


# ==================================================
# 4. Define the AICc function
# ==================================================

# AICc is the small-sample corrected version of AIC:
#
# AICc = AIC + [2k(k + 1)] / (n - k - 1),
#
# where:
#   k = number of estimated parameters
#   n = number of observations used by the model
#
# For the AR models, the estimated parameters are:
#   p AR coefficients + 1 intercept + 1 error variance
#
# Therefore:
#   AR(p): k = p + 2
#
# For the MA models, the estimated parameters are:
#   q MA coefficients + 1 intercept + 1 error variance
#
# Therefore:
#   MA(q): k = q + 2

def calculate_aicc(model, k):
    n_obs = model.nobs

    return (
        model.aic
        + (2 * k * (k + 1))
        / (n_obs - k - 1)
    )


# ==================================================
# 5. Fit AR(1), AR(2), and AR(3)
# ==================================================

ar_models = {}

for p in [1, 2, 3]:

    model = AutoReg(
        x,
        lags=p,
        trend="c"
    ).fit()

    ar_models[f"AR({p})"] = model


# ==================================================
# 6. Fit MA(1), MA(2), and MA(3)
# ==================================================

ma_models = {}

for q in [1, 2, 3]:

    model = ARIMA(
        x,
        order=(0, 0, q),
        trend="c"
    ).fit()

    ma_models[f"MA({q})"] = model


# ==================================================
# 7. Calculate AICc for all models
# ==================================================

results = []

for name, model in {**ar_models, **ma_models}.items():

    # Extract the model order from the name.
    order = int(name.split("(")[1].split(")")[0])

    # Both AR(p) and MA(q) include:
    #   - p or q dynamic parameters
    #   - 1 intercept
    #   - 1 error variance
    k = order + 2

    aicc = calculate_aicc(
        model,
        k
    )

    results.append({
        "Model": name,
        "AICc": aicc
    })


# ==================================================
# 8. Display the results
# ==================================================

results_df = pd.DataFrame(results)

print("\nAICc Comparison:")
print(results_df.to_string(index=False))