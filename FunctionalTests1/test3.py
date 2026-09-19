
import numpy as np
import pandas as pd


# ========================================================
# Test 3.1: Near-PSD Covariance Matrix
# ========================================================

# --------------------------------------------------------
# Step 1: Read the covariance matrix
# --------------------------------------------------------
# Read the covariance matrix from the CSV file.
#
# The input matrix is:
#     Sigma = covariance matrix
#
data = pd.read_csv("testout_1.3.csv")


# --------------------------------------------------------
# Step 2: Convert the DataFrame to a NumPy array
# --------------------------------------------------------
# Convert the pandas DataFrame to a NumPy array so that
# matrix operations can be performed.
covariance = data.to_numpy(dtype=float)


# --------------------------------------------------------
# Step 3: Calculate the original standard deviations
# --------------------------------------------------------
# The diagonal elements of a covariance matrix are variances:
#
#     Sigma_ii = sigma_i^2
#
# Therefore, the standard deviation of variable i is:
#
#     sigma_i = sqrt(Sigma_ii)
#
std = np.sqrt(np.diag(covariance))


# --------------------------------------------------------
# Step 4: Convert covariance matrix to correlation matrix
# --------------------------------------------------------
# Convert each covariance into a correlation:
#
#     R_ij = Sigma_ij / (sigma_i * sigma_j)
#
# np.outer(std, std) creates the matrix:
#
#     [sigma_i * sigma_j]
#
# for every pair of variables i and j.
correlation = covariance / np.outer(std, std)


# --------------------------------------------------------
# Step 5: Eigenvalue decomposition
# --------------------------------------------------------
# Decompose the correlation matrix:
#
#     R = Q Lambda Q^T
#
# where:
#     Q      = matrix of eigenvectors
#     Lambda = diagonal matrix of eigenvalues
#
# A matrix is positive semidefinite (PSD) when:
#
#     lambda_i >= 0
#
# for every eigenvalue.
eigenvalues, eigenvectors = np.linalg.eigh(correlation)


# --------------------------------------------------------
# Step 6: Replace negative eigenvalues with zero
# --------------------------------------------------------
# If an eigenvalue is negative, replace it with zero:
#
#     lambda_i* = max(lambda_i, 0)
#
# This ensures that all adjusted eigenvalues are
# non-negative and therefore removes the negative
# eigenvalues that prevent the matrix from being PSD.
eigenvalues = np.maximum(eigenvalues, 0)


# --------------------------------------------------------
# Step 7: Apply Rebonato-Jackel scaling
# --------------------------------------------------------
# After changing the eigenvalues, the reconstructed
# correlation matrix may no longer have 1s on the diagonal.
#
# For each variable i, calculate:
#
#     t_i =
#     sqrt(1 / sum_j(q_ij^2 * lambda_j*))
#
# The scaling factors are then used to form the
# diagonal scaling matrix T.
scaling = 1 / (eigenvectors**2 @ eigenvalues)
scaling = np.sqrt(scaling)


# --------------------------------------------------------
# Step 8: Construct the B matrix
# --------------------------------------------------------
# Construct B using the Rebonato-Jackel scaling:
#
#     B = T Q Lambda*^(1/2)
#
# where:
#     T            = scaling matrix
#     Q            = eigenvector matrix
#     Lambda*^(1/2) = diagonal matrix of sqrt(adjusted
#                    eigenvalues)
#
B = (
    scaling[:, None]
    * eigenvectors
    * np.sqrt(eigenvalues)[None, :]
)


# --------------------------------------------------------
# Step 9: Reconstruct the PSD correlation matrix
# --------------------------------------------------------
# Reconstruct the adjusted correlation matrix:
#
#     R_PSD = B B^T
#
# The resulting matrix is positive semidefinite because
# it is constructed as B times its transpose.
#
# The Rebonato-Jackel scaling also makes the diagonal
# approximately equal to 1.
correlation_psd = B @ B.T


# --------------------------------------------------------
# Step 10: Convert correlation back to covariance
# --------------------------------------------------------
# Convert the adjusted correlation matrix back to
# a covariance matrix:
#
#     Sigma_PSD,ij = R_PSD,ij * sigma_i * sigma_j
#
# In matrix form:
#
#     Sigma_PSD = D R_PSD D
#
# where D is the diagonal matrix of original
# standard deviations.
covariance_psd = (
    correlation_psd
    * np.outer(std, std)
)


# --------------------------------------------------------
# Step 11: Convert the result to a DataFrame
# --------------------------------------------------------
# Convert the NumPy array back to a pandas DataFrame.
# The original column names are used for the variables.
near_psd_covariance = pd.DataFrame(
    covariance_psd,
    columns=data.columns
)


# --------------------------------------------------------
# Step 12: Print the Near-PSD covariance matrix
# --------------------------------------------------------
print(near_psd_covariance)


# ========================================================
# Test 3.2: Near-PSD Correlation Matrix
# ========================================================

# --------------------------------------------------------
# Step 1: Read the correlation matrix
# --------------------------------------------------------
# This input is already a correlation matrix, so we do NOT
# need to convert it from covariance to correlation.
#
# A correlation matrix has:
#
#     R_ii = 1
#
# on its diagonal.
data = pd.read_csv("testout_1.4.csv")


# --------------------------------------------------------
# Step 2: Convert the DataFrame to a NumPy array
# --------------------------------------------------------
correlation = data.to_numpy(dtype=float)


# --------------------------------------------------------
# Step 3: Eigenvalue decomposition
# --------------------------------------------------------
# Decompose the correlation matrix:
#
#     R = Q Lambda Q^T
#
# where Q contains the eigenvectors and Lambda contains
# the eigenvalues.
eigenvalues, eigenvectors = np.linalg.eigh(correlation)


# --------------------------------------------------------
# Step 4: Replace negative eigenvalues with zero
# --------------------------------------------------------
# For a PSD matrix:
#
#     lambda_i >= 0
#
# Replace every negative eigenvalue with zero:
#
#     lambda_i* = max(lambda_i, 0)
#
eigenvalues = np.maximum(eigenvalues, 0)


# --------------------------------------------------------
# Step 5: Apply Rebonato-Jackel scaling
# --------------------------------------------------------
# The eigenvalue adjustment may change the diagonal
# elements of the correlation matrix.
#
# For each variable i:
#
#     t_i =
#     sqrt(1 / sum_j(q_ij^2 * lambda_j*))
#
# These scaling factors ensure that the reconstructed
# correlation matrix has diagonal elements approximately
# equal to 1.
scaling = 1 / (eigenvectors**2 @ eigenvalues)
scaling = np.sqrt(scaling)


# --------------------------------------------------------
# Step 6: Construct the B matrix
# --------------------------------------------------------
# Construct:
#
#     B = T Q Lambda*^(1/2)
#
# using the scaled eigenvectors and the square roots
# of the adjusted eigenvalues.
B = (
    scaling[:, None]
    * eigenvectors
    * np.sqrt(eigenvalues)[None, :]
)


# --------------------------------------------------------
# Step 7: Reconstruct the Near-PSD correlation matrix
# --------------------------------------------------------
# Reconstruct the adjusted correlation matrix:
#
#     R_PSD = B B^T
#
# This produces a PSD correlation matrix with diagonal
# elements approximately equal to 1.
correlation_psd = B @ B.T


# --------------------------------------------------------
# Step 8: Convert the result to a DataFrame
# --------------------------------------------------------
# Convert the NumPy array back to a pandas DataFrame
# using the original variable names.
near_psd_correlation = pd.DataFrame(
    correlation_psd,
    columns=data.columns
)


# --------------------------------------------------------
# Step 9: Print the Near-PSD correlation matrix
# --------------------------------------------------------
print(near_psd_correlation)










# ========================================================
# Test 3.3: Higham Covariance Matrix
# ========================================================

# --------------------------------------------------------
# Step 1: Read the covariance matrix
# --------------------------------------------------------
data = pd.read_csv("testout_1.3.csv")

# Convert to NumPy array
covariance = data.to_numpy(dtype=float)

# --------------------------------------------------------
# Step 2: Calculate the original standard deviations
# --------------------------------------------------------
# sigma_i = sqrt(Sigma_ii)
std = np.sqrt(np.diag(covariance))

# --------------------------------------------------------
# Step 3: Convert covariance matrix to correlation matrix
# --------------------------------------------------------
# R_ij = Sigma_ij / (sigma_i * sigma_j)
correlation = covariance / np.outer(std, std)

# Make the matrix symmetric
correlation = (correlation + correlation.T) / 2

# --------------------------------------------------------
# Step 4: Initialize Higham algorithm
# --------------------------------------------------------
# Y^(0) = R
Y = correlation.copy()

# Dykstra correction matrix
Delta_S = np.zeros_like(correlation)

# Convergence settings
tol = 1e-9
max_iter = 100

# --------------------------------------------------------
# Step 5: Iterate Higham's algorithm
# --------------------------------------------------------
for iteration in range(max_iter):

    # Save previous matrix for convergence check
    Y_old = Y.copy()

    # ----------------------------------------------------
    # Step 5.1: Apply Dykstra correction
    # ----------------------------------------------------
    # R^(k) = Y^(k) - Delta_S
    R = Y - Delta_S

    # ----------------------------------------------------
    # Step 5.2: Project onto PSD matrices
    # ----------------------------------------------------
    # R = Q Lambda Q^T
    eigenvalues, eigenvectors = np.linalg.eigh(R)

    # Replace negative eigenvalues with zero
    # lambda_i* = max(lambda_i, 0)
    eigenvalues = np.maximum(eigenvalues, 0)

    # Reconstruct the PSD matrix
    X = (
        eigenvectors
        @ np.diag(eigenvalues)
        @ eigenvectors.T
    )

    # ----------------------------------------------------
    # Step 5.3: Update Dykstra correction
    # ----------------------------------------------------
    # Delta_S = X - R
    Delta_S = X - R

    # ----------------------------------------------------
    # Step 5.4: Project onto unit-diagonal matrices
    # ----------------------------------------------------
    # Set all diagonal elements equal to 1
    Y = X.copy()
    np.fill_diagonal(Y, 1.0)

    # ----------------------------------------------------
    # Step 5.5: Check convergence
    # ----------------------------------------------------
    difference = np.linalg.norm(Y - Y_old, ord="fro")

    if difference < tol:
        break

# --------------------------------------------------------
# Step 6: Final Higham correlation matrix
# --------------------------------------------------------
correlation_higham = (Y + Y.T) / 2

# --------------------------------------------------------
# Step 7: Convert correlation back to covariance
# --------------------------------------------------------
# Sigma_Higham = D R_Higham D
covariance_higham = (
    correlation_higham
    * np.outer(std, std)
)

# --------------------------------------------------------
# Step 8: Convert result to DataFrame
# --------------------------------------------------------
higham_covariance = pd.DataFrame(
    covariance_higham,
    columns=data.columns
)

# --------------------------------------------------------
# Step 9: Print the Higham covariance matrix
# --------------------------------------------------------
print(higham_covariance)






# ========================================================
# Test 3.4: Higham correlation
# ========================================================

# --------------------------------------------------------
# Step 1: Read data
# --------------------------------------------------------

data = pd.read_csv("testout_1.4.csv")
R = data.to_numpy(dtype=float)
labels = data.columns


# --------------------------------------------------------
# Step 2: Higham algorithm
# --------------------------------------------------------

Y = R.copy()

# Dykstra correction
Delta_S = np.zeros_like(R)

# Convergence settings
tol = 1e-9
max_iter = 100


for iteration in range(max_iter):

    # Save previous matrix
    Y_old = Y.copy()


    # ----------------------------------------------------
    # 2.1 Dykstra correction
    # ----------------------------------------------------
    R_k = Y - Delta_S


    # ----------------------------------------------------
    # 2.2 Project onto PSD matrices
    # ----------------------------------------------------
    #
    # R_k = Q Lambda Q^T
    #

    eigenvalues, eigenvectors = np.linalg.eigh(R_k)


    # Replace negative eigenvalues by zero
    eigenvalues = np.maximum(eigenvalues, 0)


    # Reconstruct PSD matrix
    X = (
        eigenvectors
        @ np.diag(eigenvalues)
        @ eigenvectors.T
    )


    # ----------------------------------------------------
    # 2.3 Update Dykstra correction
    # ----------------------------------------------------

    Delta_S = X - R_k


    # ----------------------------------------------------
    # 2.4 Project onto correlation matrices
    # ----------------------------------------------------
    # Force diagonal elements to equal 1

    Y = X.copy()

    np.fill_diagonal(Y, 1.0)


    # ----------------------------------------------------
    # 2.5 Check convergence
    # ----------------------------------------------------

    difference = np.linalg.norm(
        Y - Y_old,
        ord="fro"
    )

    if difference < tol:
        break


# --------------------------------------------------------
# Step 3: Final Higham correlation matrix
# --------------------------------------------------------

correlation_higham = (Y + Y.T) / 2


# --------------------------------------------------------
# Step 4: Convert to DataFrame
# --------------------------------------------------------

correlation_higham = pd.DataFrame(
    correlation_higham,
    columns=labels
)


# --------------------------------------------------------
# Step 5: Print result
# --------------------------------------------------------

print(correlation_higham)
