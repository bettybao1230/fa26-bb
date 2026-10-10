import numpy as np
import pandas as pd

from risklib import (
    pairwise_observation_counts,
    complete_case_correlation,
    pairwise_correlation,
    eigenvalues_of_correlation,
    is_psd,
    covariance_from_correlation,
    portfolio_variance,
    rebonato_jackel,
    higham_nearest_correlation,
)


# ---------------------------------------------------------
# Load the data
# ---------------------------------------------------------

df = pd.read_csv("problem1.csv")

print("Data:")
print(df)

print("\nNumber of rows and columns:")
print(df.shape)

# The Day column identifies the observation date and is not
# a return series, so exclude it from all calculations.
returns = df[["A", "B", "C", "D", "IDX"]]

print("\nReturn series used in the analysis:")
print(returns.columns.tolist())


# =========================================================
# Part 1: Jointly observed days
# =========================================================

# Count the number of days on which each pair of series
# has an observation.
counts = pairwise_observation_counts(returns)

print("\nPairwise number of jointly observed days:")
print(counts)

# Count the number of days on which all five series
# have observations.
complete_days = returns.dropna().shape[0]

print("\nNumber of days when all five assets trade:")
print(complete_days)


# =========================================================
# Part d: Complete-case correlation
# =========================================================

# Use only rows where all five series are observed.
complete_corr = complete_case_correlation(returns)

print("\nComplete-row correlation matrix:")
print(complete_corr)

# Calculate the eigenvalues of the complete-row
# correlation matrix.
complete_eigenvalues = eigenvalues_of_correlation(
    complete_corr
)

print("\nEigenvalues of complete-row correlation matrix:")
print(complete_eigenvalues)

print("\nSmallest eigenvalue:")
print(complete_eigenvalues[0])

# Check whether the complete-row correlation matrix
# is positive semidefinite.
print("\nIs the complete-row correlation matrix PSD?")
print(is_psd(complete_corr))


# Attempt Cholesky factorization.
# Cholesky requires the matrix to be positive definite.
try:
    complete_cholesky = np.linalg.cholesky(
        complete_corr.to_numpy()
    )

    print("\nComplete-row Cholesky factorization:")
    print("Succeeded.")
    print(complete_cholesky)

except np.linalg.LinAlgError as error:
    print("\nComplete-row Cholesky factorization:")
    print("Failed.")
    print("Error:", error)


# =========================================================
# Part d: Pairwise correlation
# =========================================================

# Each pairwise correlation uses all observations available
# for that particular pair.
pairwise_corr = pairwise_correlation(returns)

print("\nPairwise correlation matrix:")
print(pairwise_corr)

# Calculate the eigenvalues of the pairwise
# correlation matrix.
pairwise_eigenvalues = eigenvalues_of_correlation(
    pairwise_corr
)

print("\nEigenvalues of pairwise correlation matrix:")
print(pairwise_eigenvalues)

print("\nSmallest eigenvalue:")
print(pairwise_eigenvalues[0])

# Check whether the pairwise correlation matrix
# is positive semidefinite.
print("\nIs the pairwise correlation matrix PSD?")
print(is_psd(pairwise_corr))


# Attempt Cholesky factorization.
# Cholesky requires the matrix to be positive definite.
try:
    pairwise_cholesky = np.linalg.cholesky(
        pairwise_corr.to_numpy()
    )

    print("\nPairwise Cholesky factorization:")
    print("Succeeded.")
    print(pairwise_cholesky)

except np.linalg.LinAlgError as error:
    print("\nPairwise Cholesky factorization:")
    print("Failed.")
    print("Error:", error)


# =========================================================
# Part e: Covariance matrix and tracking portfolio variance
# =========================================================

# Use the full history of each individual series to
# estimate its standard deviation.
#
# ddof=1 gives the sample standard deviation.
full_history_std = returns.std(ddof=1)

print("\nFull-history standard deviations:")
print(full_history_std)

# Convert the pairwise correlation matrix into a covariance
# matrix using:
#
# Sigma = D R D
#
# where D is the diagonal matrix of standard deviations
# and R is the pairwise correlation matrix.
pairwise_covariance = covariance_from_correlation(
    pairwise_corr,
    full_history_std,
)

print(
    "\nCovariance matrix from pairwise correlations "
    "and full-history standard deviations:"
)
print(pairwise_covariance)


# The portfolio is:
#
#   +1.00 IDX
#   -0.40 A
#   -0.30 B
#   -0.20 C
#   -0.10 D
#
# The order matches the columns:
# [A, B, C, D, IDX].
weights = np.array([
    -0.40,
    -0.30,
    -0.20,
    -0.10,
     1.00,
])

# Portfolio variance is:
#
#   Var(P) = w' Sigma w
#
# This measures the variance of the tracking-error
# portfolio: long IDX and short the weighted basket.
tracking_variance = portfolio_variance(
    pairwise_covariance,
    weights,
)

print("\nTracking portfolio weights:")
print(weights)

print("\nTracking portfolio variance:")
print(tracking_variance)


# =========================================================
# Part f: Rebonato-Jackel repair
# =========================================================

# Repair the pairwise correlation matrix using the
# Rebonato-Jackel method.
#
# The repair adjusts negative eigenvalues and then
# rescales the matrix so that its diagonal remains one.
rebonato_corr = rebonato_jackel(pairwise_corr)

print("\nRebonato-Jackel repaired correlation matrix:")
print(rebonato_corr)

# Calculate the eigenvalues after the repair.
rebonato_eigenvalues = eigenvalues_of_correlation(
    rebonato_corr
)

print("\nRebonato-Jackel eigenvalues:")
print(rebonato_eigenvalues)

print("\nRebonato-Jackel smallest eigenvalue:")
print(rebonato_eigenvalues[0])

# Measure how far the repaired matrix is from the
# original pairwise correlation matrix.
#
# Frobenius distance:
#
#   ||R_repaired - R_pairwise||_F
rebonato_distance = np.linalg.norm(
    rebonato_corr.to_numpy() - pairwise_corr.to_numpy(),
    ord="fro",
)

print("\nRebonato-Jackel Frobenius distance:")
print(rebonato_distance)


# Build a covariance matrix using the repaired
# correlation matrix and the same full-history
# standard deviations from Part e.
rebonato_covariance = covariance_from_correlation(
    rebonato_corr,
    full_history_std,
)

# Recompute the tracking portfolio variance.
rebonato_tracking_variance = portfolio_variance(
    rebonato_covariance,
    weights,
)

print("\nRebonato-Jackel tracking portfolio variance:")
print(rebonato_tracking_variance)


# =========================================================
# Part f: Higham repair
# =========================================================

# Repair the pairwise correlation matrix using
# Higham's nearest correlation matrix algorithm.
#
# The algorithm finds a nearby matrix that is PSD
# and has ones on the diagonal.
higham_corr = higham_nearest_correlation(pairwise_corr)

print("\nHigham repaired correlation matrix:")
print(higham_corr)

# Calculate the eigenvalues after the repair.
higham_eigenvalues = eigenvalues_of_correlation(
    higham_corr
)

print("\nHigham eigenvalues:")
print(higham_eigenvalues)

print("\nHigham smallest eigenvalue:")
print(higham_eigenvalues[0])

# Calculate the Frobenius distance between the
# Higham matrix and the original pairwise matrix.
higham_distance = np.linalg.norm(
    higham_corr.to_numpy() - pairwise_corr.to_numpy(),
    ord="fro",
)

print("\nHigham Frobenius distance:")
print(higham_distance)


# Build a covariance matrix from the Higham-repaired
# correlation matrix and the same full-history
# standard deviations.
higham_covariance = covariance_from_correlation(
    higham_corr,
    full_history_std,
)

# Recompute the tracking portfolio variance.
higham_tracking_variance = portfolio_variance(
    higham_covariance,
    weights,
)

print("\nHigham tracking portfolio variance:")
print(higham_tracking_variance)


