# ==================================================
# Problem 1: Reading the Shape of a Sample.
# ==================================================

import pandas as pd
import numpy as np
from scipy import stats


# ==================================================
# 1. Load the data
# ==================================================

data1 = pd.read_csv("problem1.csv")

# Extract the x series and remove any missing observations.
x = data1["x"].dropna().to_numpy()
n = len(x)


# ==================================================
# 2. Compute the first four moments
# ==================================================

# The first moment is the sample mean, which measures
# the center of the distribution.
mean = np.mean(x)

# The second central moment is the sample variance,
# which measures the dispersion of the observations
# around the mean.
variance = np.var(x, ddof=1)

# Skewness measures the asymmetry of the distribution.
# bias=False applies the bias correction for the sample.
skewness = stats.skew(x, bias=False)

# Excess kurtosis measures tail heaviness relative to
# a Normal distribution. A Normal distribution has
# excess kurtosis equal to zero.
excess_kurtosis = stats.kurtosis(
    x,
    fisher=True,
    bias=False
)


print(f"Number of observations: {n}")

print("\nFirst four moments:")
print(f"Mean = {mean:.6f}")
print(f"Variance = {variance:.6f}")
print(f"Skewness = {skewness:.6f}")
print(f"Excess kurtosis = {excess_kurtosis:.6f}")


# ==================================================
# 3. Fit a Normal distribution
# ==================================================

# Fit a Normal distribution using the sample mean and
# sample standard deviation as the estimated parameters.
sample_mean = mean
sample_std = np.sqrt(variance)

normal = stats.norm(
    loc=sample_mean,
    scale=sample_std
)


# ==================================================
# 4. Evaluate the 1% lower tail
# ==================================================

# Compute the 1% lower quantile of the fitted
# Normal distribution.
q01 = normal.ppf(0.01)

# Count the number of observed values below this
# fitted 1% Normal quantile.
observed = np.sum(x < q01)

# If the fitted Normal model were correct, approximately
# 1% of the observations would be expected to fall below
# this quantile.
expected = 0.01 * n


print("\nFitted Normal distribution:")
print(f"Mean = {sample_mean:.6f}")
print(f"Standard deviation = {sample_std:.6f}")

print("\n1% lower tail:")
print(f"Normal 1% quantile = {q01:.6f}")
print(f"Observed below 1% quantile = {observed}")
print(f"Expected below 1% quantile = {expected:.2f}")