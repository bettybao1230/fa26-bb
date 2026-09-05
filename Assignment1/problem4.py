# ==================================================
# Problem 4:  Conditional Distributions.
# ==================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ==================================================
# 1. Load the data
# ==================================================

data4 = pd.read_csv("problem4.csv")

x1 = data4["x1"].to_numpy()
x2 = data4["x2"].to_numpy()


# ==================================================
# 2. Compute sample means and covariance matrix
# ==================================================

mu1 = np.mean(x1)
mu2 = np.mean(x2)

# np.cov() uses the sample covariance estimator,
# dividing by n - 1.
sigma = np.cov(
    np.column_stack((x1, x2)),
    rowvar=False
)

print("Sample covariance matrix:")
print(sigma)
print()


# ==================================================
# 3. Partition the covariance matrix
# ==================================================

# Since X1 and X2 are scalars, each block of the
# partitioned covariance matrix is a scalar:
#
# Sigma = [ sigma11  sigma12 ]
#         [ sigma21  sigma22 ]

sigma11 = sigma[0, 0]   # Var(X1)
sigma12 = sigma[0, 1]   # Cov(X1, X2)
sigma21 = sigma[1, 0]   # Cov(X2, X1)
sigma22 = sigma[1, 1]   # Var(X2)


# ==================================================
# 4. Compute E[X2 | X1 = x1]
# ==================================================

# Under the joint normal assumption, the conditional
# expectation is
#
# E[X2 | X1 = x1]
# = mu2 + (sigma21 / sigma11) * (x1 - mu1).
#
# The coefficient sigma21 / sigma11 is the slope of
# the conditional expectation as a function of x1.

beta_cond = sigma21 / sigma11

conditional_mean = (
    mu2 + beta_cond * (x1 - mu1)
)


# ==================================================
# 5. Compute Var(X2 | X1)
# ==================================================

# Under the joint normal assumption,
#
# Var(X2 | X1)
# = sigma22 - sigma21 * sigma11^(-1) * sigma12.
#
# Because X1 and X2 are scalars, this simplifies to
#
# Var(X2 | X1)
# = sigma22 - (sigma21 * sigma12) / sigma11.
#
# This variance does not depend on the value of x1,
# so the resulting conditional band has constant width.

conditional_variance = (
    sigma22 - (sigma21 * sigma12) / sigma11
)

if conditional_variance < 0:
    raise ValueError("Conditional variance is negative.")

conditional_sd = np.sqrt(conditional_variance)


# ==================================================
# 6. Construct the 95% conditional prediction band
# ==================================================

# Under joint normality,
#
# X2 | X1 = x1
# ~ N(conditional_mean, conditional_variance).
#
# Therefore, an approximate 95% conditional interval is
#
# E[X2 | X1 = x1] +/- 1.96 * sqrt(Var(X2 | X1)).

z_95 = 1.96

lower = conditional_mean - z_95 * conditional_sd
upper = conditional_mean + z_95 * conditional_sd


# ==================================================
# 7. Sort x1 values for plotting
# ==================================================

# The observations are not necessarily ordered by x1.
# Sorting them allows the conditional expectation and
# band boundaries to be plotted as continuous lines.

order = np.argsort(x1)

x1_sorted = x1[order]
conditional_mean_sorted = conditional_mean[order]
lower_sorted = lower[order]
upper_sorted = upper[order]


# ==================================================
# 8. Plot the data and conditional distribution
# ==================================================

plt.figure(figsize=(8, 5))

# Observed data
plt.scatter(
    x1,
    x2,
    alpha=0.6,
    label="Observed data"
)

# Estimated conditional expectation
plt.plot(
    x1_sorted,
    conditional_mean_sorted,
    linewidth=2,
    label=r"Conditional expectation, $E[X_2 \mid X_1=x_1]$"
)

# Lower and upper limits of the 95% conditional band
plt.plot(
    x1_sorted,
    lower_sorted,
    linestyle="--",
    label="Lower limit of 95% conditional band"
)

plt.plot(
    x1_sorted,
    upper_sorted,
    linestyle="--",
    label="Upper limit of 95% conditional band"
)

plt.xlabel(r"$x_1$")
plt.ylabel(r"$x_2$")
plt.title(r"Conditional Expectation and 95% Conditional Band")
plt.legend()
plt.tight_layout()
plt.show()


# ==================================================
# 9. Compute overall coverage
# ==================================================

# An observation is covered when its observed x2 value
# falls within the conditional interval corresponding
# to its observed x1 value.

inside_band = (
    (x2 >= lower) &
    (x2 <= upper)
)

overall_coverage = np.mean(inside_band)

print(
    f"Overall coverage of the 95% conditional band: "
    f"{overall_coverage:.4f}"
)


# ==================================================
# 10. Compute coverage by distance from the mean of x1
# ==================================================

# Standardize x1 using its sample mean and standard
# deviation. This measures how far each observation is
# from the center of the x1 distribution in standard
# deviation units.

sd_x1 = np.sqrt(sigma11)

z_x1 = np.abs((x1 - mu1) / sd_x1)


# Divide observations into three regions:
#   1. Within 1 standard deviation
#   2. Between 1 and 2 standard deviations
#   3. Beyond 2 standard deviations

bucket_1 = z_x1 <= 1
bucket_2 = (z_x1 > 1) & (z_x1 <= 2)
bucket_3 = z_x1 > 2


# Compute the fraction of observations inside the
# 95% conditional band within each region.

coverage_1 = np.mean(inside_band[bucket_1])
coverage_2 = np.mean(inside_band[bucket_2])
coverage_3 = np.mean(inside_band[bucket_3])


# ==================================================
# 11. Report coverage by x1 region
# ==================================================
print()
print(
    f"Coverage within 1 standard deviation: "
    f"{coverage_1:.4f}"
)

print(
    f"Coverage between 1 and 2 standard deviations: "
    f"{coverage_2:.4f}"
)

print(
    f"Coverage beyond 2 standard deviations: "
    f"{coverage_3:.4f}"
)