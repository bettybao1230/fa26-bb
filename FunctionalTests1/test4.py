# input near_psd covariance
# output chol_psd

import numpy as np
import pandas as pd


def chol_psd(matrix):
    """
    Input:
        matrix: near-PSD covariance matrix

    Output:
        root: lower-triangular matrix such that
              matrix ≈ root @ root.T
    """

    n = matrix.shape[0]

    # Symmetrize the matrix
    matrix = (matrix + matrix.T) / 2

    root = np.zeros((n, n))
    tol = 1e-12

    for i in range(n):

        # ------------------------------------------------
        # Diagonal:
        # L_ii^2 = Sigma_ii - sum(L_ik^2)
        # ------------------------------------------------
        value = matrix[i, i]

        for k in range(i):
            value -= root[i, k] ** 2

        if value < -tol:
            raise ValueError("Matrix is not PSD")

        if value < 0:
            value = 0.0

        root[i, i] = np.sqrt(value)

        # ------------------------------------------------
        # Below diagonal:
        # L_ji =
        # [Sigma_ji - sum(L_jk L_ik)] / L_ii
        # ------------------------------------------------
        for j in range(i + 1, n):

            value = matrix[j, i]

            for k in range(i):
                value -= root[j, k] * root[i, k]

            if root[i, i] > tol:
                root[j, i] = value / root[i, i]
            else:
                root[j, i] = 0.0

    return root



covariance = pd.read_csv("testout_3.1.csv")
labels = covariance.columns
covariance = covariance.to_numpy()

chol = chol_psd(covariance)
chol_psd = pd.DataFrame(
    chol,
    columns=labels
)

print(chol_psd)