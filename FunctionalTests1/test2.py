# --------------------------------------------------------
# Test 2
# --------------------------------------------------------

import numpy as np
import pandas as pd


# --------------------------------------------------------
# Read input data
# --------------------------------------------------------

data = pd.read_csv("test2.csv")
labels = data.columns
X = data.to_numpy(dtype=float)


# --------------------------------------------------------
# EW Covariance Function
# --------------------------------------------------------

def ew_cov(X, lam):
    """
    Calculate the exponentially weighted covariance matrix.

    Parameters:
        X   : data matrix (T observations x N variables)
        lam : EW decay factor

    Returns:
        EW covariance matrix
    """

    T = len(X)

    # EW weights:
    # w_t = lambda^(T-t) / sum_{j=1}^T lambda^(T-j)
    # The newest observation receives the largest weight.
    weights = lam ** np.arange(T - 1, -1, -1)

    # Normalize weights so that they sum to 1.
    weights = weights / weights.sum()

    # EW mean:
    # mu_EW = sum_{t=1}^T w_t * x_t
    ew_mean = np.sum(X * weights[:, None], axis=0)

    # Deviations from the EW mean:
    # d_t = x_t - mu_EW
    deviations = X - ew_mean

    # EW covariance:
    # Sigma_EW = sum_{t=1}^T w_t * d_t * d_t^T
    covariance = deviations.T @ (
        deviations * weights[:, None]
    )

    return covariance


# --------------------------------------------------------
# Test 2.1: EW Covariance, lambda = 0.97
# --------------------------------------------------------

cov_ew97 = ew_cov(X, lam=0.97)

# Convert covariance matrix to DataFrame
cov_ew97 = pd.DataFrame(
    cov_ew97,
    columns=labels
)

print(cov_ew97)


# --------------------------------------------------------
# Test 2.2: EW Correlation, lambda = 0.94
# --------------------------------------------------------

cov_ew94 = ew_cov(X, lam=0.94)

# EW standard deviations:
# sigma_i = sqrt(Sigma_EW[i,i])
std_ew94 = np.sqrt(np.diag(cov_ew94))

# EW correlation:
# rho_ij = Sigma_EW[i,j] / (sigma_i * sigma_j)
corr_ew94 = cov_ew94 / np.outer(std_ew94, std_ew94)

# Convert correlation matrix to DataFrame
corr_ew94 = pd.DataFrame(
    corr_ew94,
    columns=labels
)

print(corr_ew94)


# --------------------------------------------------------
# Test 2.3: Covariance with EW Variance (lambda = 0.97)
#          and EW Correlation (lambda = 0.94)
# --------------------------------------------------------

# EW standard deviations using lambda = 0.97
# sigma_ew97[i] = sqrt(Cov_ew97[i,i])
std_ew97 = np.sqrt(np.diag(cov_ew97))

# Combine the EW correlation from Test 2.2
# with the EW standard deviations from Test 2.1.
#
# covariance[i,j] =
#     corr_ew94[i,j] * std_ew97[i] * std_ew97[j]

result = corr_ew94.to_numpy() * np.outer(
    std_ew97,
    std_ew97
)

# Convert final covariance matrix to DataFrame
result = pd.DataFrame(
    result,
    columns=labels
)

print(result)



