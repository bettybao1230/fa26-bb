# Test 5.1. Normal Simulation PD Input 0 mean - 100,000 simulations, compare input vs output covariance
# ---------------------------------------------------------------------------------------------------------
"""
Normal simulation from a covariance matrix (PD input).

Steps:
1. Read the positive-definite covariance matrix.
2. Cholesky-decompose it: cov = L @ L.T.
3. Draw Z ~ N(0, I), with 100,000 simulations.
4. Generate X = Z @ L.T, so X ~ N(0, cov).
5. Calculate the sample covariance of X.
6. Compare the simulated covariance with the input covariance.
"""

import pandas as pd
import numpy as np


def simulate_normal(cov: np.ndarray, n_sim: int, seed: int = None) -> np.ndarray:
    """
    Simulate observations from N(0, cov) using Cholesky decomposition.

    Parameters
    ----------
    cov : np.ndarray
        Positive-definite covariance matrix.
    n_sim : int
        Number of simulations.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        Simulated observations with mean approximately 0
        and covariance approximately equal to cov.
    """

    n = cov.shape[0]

    # Random number generator
    rng = np.random.default_rng(seed)

    # Cholesky decomposition:
    # cov = L @ L.T
    L = np.linalg.cholesky(cov)

    # Generate independent standard normal variables:
    # Z ~ N(0, I)
    Z = rng.standard_normal((n_sim, n))

    # Transform independent normals into correlated normals:
    # X = Z @ L.T
    X = Z @ L.T

    return X




# ---------------------------------------------------------
# Settings
# ---------------------------------------------------------
input_path = "test5_1.csv"

n_sim = 100000
seed = 0

# ---------------------------------------------------------
# 1. Read input covariance matrix
# ---------------------------------------------------------
df = pd.read_csv(input_path)
cov_in = df.values

# ---------------------------------------------------------
# 2. Simulate N(0, cov)
# ---------------------------------------------------------
X = simulate_normal(
    cov=cov_in,
    n_sim=n_sim,
    seed=seed
)

# ---------------------------------------------------------
# 3. Calculate simulated covariance
# ---------------------------------------------------------
cov_out = np.cov(X, rowvar=False)

# ---------------------------------------------------------
# 4. Save output covariance
# ---------------------------------------------------------
out_pd = pd.DataFrame(
    cov_out,
    columns=df.columns
)

print(out_pd)





# ---------------------------------------------------------------------------------------------------------
# Test 5.2. Normal Simulation PSD Input 0 mean - 100,000 simulations, compare input vs output covariance
# ---------------------------------------------------------------------------------------------------------
"""
Normal Simulation from PSD Input, Mean = 0, 100,000 simulations.

Steps:
1. Read the PSD covariance matrix.
2. Eigenvalue-decompose the covariance matrix.
3. Set negative eigenvalues to zero.
4. Generate Z ~ N(0, I).
5. Transform Z into X ~ N(0, covariance).
6. Calculate the output covariance.
7. Compare input vs. output covariance.
"""



# ---------------------------------------------------------
# Settings
# ---------------------------------------------------------
input_path = "test5_2.csv"

n_sim = 100_000
seed = 0


# ---------------------------------------------------------
# 1. Read PSD covariance matrix
# ---------------------------------------------------------
df = pd.read_csv(input_path)
cov_in = df.values.astype(float)


# ---------------------------------------------------------
# 2. Eigenvalue decomposition
#
#     cov = Q @ Lambda @ Q.T
# ---------------------------------------------------------
eigenvalues, eigenvectors = np.linalg.eigh(cov_in)


# ---------------------------------------------------------
# 3. Set negative eigenvalues to zero
#
# PSD matrix should have eigenvalues >= 0.
# Small negative values can occur because of numerical error.
# ---------------------------------------------------------
eigenvalues = np.maximum(eigenvalues, 0.0)


# ---------------------------------------------------------
# 4. Construct square-root matrix
#
#     L = Q @ sqrt(Lambda)
#
# Then:
#     L @ L.T = cov
# ---------------------------------------------------------
L = eigenvectors @ np.diag(np.sqrt(eigenvalues))


# ---------------------------------------------------------
# 5. Generate independent standard normal variables
#
#     Z ~ N(0, I)
# ---------------------------------------------------------
rng = np.random.default_rng(seed)

n = cov_in.shape[0]

Z = rng.standard_normal(
    size=(n_sim, n)
)


# ---------------------------------------------------------
# 6. Generate correlated normal simulations
#
#     X = Z @ L.T
#
#     X ~ N(0, cov)
# ---------------------------------------------------------
X = Z @ L.T


# ---------------------------------------------------------
# 7. Calculate output covariance
# ---------------------------------------------------------
cov_out = np.cov(
    X,
    rowvar=False
)


# ---------------------------------------------------------
# 8. Save output covariance
# ---------------------------------------------------------
out_psd = pd.DataFrame(
    cov_out,
    columns=df.columns
)

print(out_psd)





# ---------------------------------------------------------------------------------------------------------------------------
# Test 5.3. Normal Simulation nonPSD Input, 0 mean, near_psd fix - 100,000 simulations
# Compare input vs fixed vs simulated covariance
# ---------------------------------------------------------------------------------------------------------------------------


# ------------------------------------------------------------
# Load input covariance matrix
# ------------------------------------------------------------
data = pd.read_csv("test5_3.csv")
a = data.values
cols = data.columns
n = a.shape[0]

np.random.seed(42)

# ------------------------------------------------------------
# Step 1: near-PSD fix
# Rebonato-Jackel eigenvalue clipping
# ------------------------------------------------------------
out = a.copy()
invSD = None

# If input is not a correlation matrix, convert to correlation matrix
if not np.allclose(np.diag(out), 1.0):
    invSD = np.diag(1.0 / np.sqrt(np.diag(out)))
    out = invSD @ out @ invSD

# Eigenvalue decomposition
vals, vecs = np.linalg.eigh(out)

# Clip negative eigenvalues to zero
vals = np.maximum(vals, 0.0)

# Rebonato-Jackel scaling
T = 1.0 / ((vecs * vecs) @ vals)
T = np.diag(np.sqrt(T))

l = np.diag(np.sqrt(vals))

# Construct near-PSD matrix
B = T @ vecs @ l
out = B @ B.T

# Convert back to covariance matrix
if invSD is not None:
    invSD2 = np.diag(1.0 / np.diag(invSD))
    out = invSD2 @ out @ invSD2

fixed = out

# ------------------------------------------------------------
# Step 2: Cholesky decomposition
# fixed ≈ L @ L.T
# ------------------------------------------------------------
L = np.zeros((n, n))

for j in range(n):

    if j > 0:
        s = L[j, :j] @ L[j, :j].T
    else:
        s = 0.0

    temp = fixed[j, j] - s

    # Handle small negative numerical errors
    if -1e-8 < temp < 0:
        temp = 0.0

    L[j, j] = np.sqrt(temp) if temp > 0 else 0.0

    if L[j, j] == 0.0:
        continue

    ir = 1.0 / L[j, j]

    for i in range(j + 1, n):

        if j > 0:
            s = L[i, :j] @ L[j, :j].T
        else:
            s = 0.0

        L[i, j] = (fixed[i, j] - s) * ir

# ------------------------------------------------------------
# Step 3: Simulate 100,000 multivariate normal observations
# Mean = 0
# ------------------------------------------------------------
n_sims = 100000

z = np.random.normal(
    loc=0.0,
    scale=1.0,
    size=(n, n_sims)
)

# X = LZ
# Each row after transpose is one simulation
sim = (L @ z).T

# ------------------------------------------------------------
# Step 4: Calculate empirical covariance
# ------------------------------------------------------------
sim_cov = np.cov(sim, rowvar=False)


print(pd.DataFrame(sim_cov, columns=cols))






# ---------------------------------------------------------------------------------------------------------------------------
# Test 5.4. Normal Simulation nonPSD Input, 0 mean, Higham fix - 100,000 simulations, compare input vs output covariance
# ---------------------------------------------------------------------------------------------------------------------------

np.random.seed(42)
data = pd.read_csv("test5_3.csv")
cols = data.columns

# ---- Load input covariance matrix ----
a = data.values
n = a.shape[0]


# ---------------------------------------------------------------------------------------------------------------------------
# Step 1: Convert covariance matrix to correlation matrix
# ---------------------------------------------------------------------------------------------------------------------------

invSD = np.diag(1.0 / np.sqrt(np.diag(a)))

corr = invSD @ a @ invSD


# ---------------------------------------------------------------------------------------------------------------------------
# Step 2: Higham (2002) nearest PSD correlation matrix fix
# ---------------------------------------------------------------------------------------------------------------------------

# Initialize
Y = corr.copy()

# Dykstra correction matrix
deltaS = np.zeros_like(Y)

# Convergence settings
tol = 1e-9
max_iter = 100
prev_gamma = np.inf


for i in range(max_iter):

    # R = Y - DeltaS
    R = Y - deltaS

    # Eigenvalue decomposition
    vals, vecs = np.linalg.eigh(R)

    # Set negative eigenvalues to zero
    vals_clipped = np.maximum(vals, 0.0)

    # Projection onto PSD matrices
    X = vecs @ np.diag(vals_clipped) @ vecs.T

    # Update Dykstra correction
    deltaS = X - R

    # Projection onto correlation matrices:
    # force diagonal elements to 1
    Y = X.copy()
    np.fill_diagonal(Y, 1.0)

    # Check convergence
    gamma = np.linalg.norm(Y - corr, 'fro')

    if abs(gamma - prev_gamma) < tol:
        break

    prev_gamma = gamma


# ---------------------------------------------------------------------------------------------------------------------------
# Step 3: Convert fixed correlation matrix back to covariance matrix
# ---------------------------------------------------------------------------------------------------------------------------

# D = inverse(invSD)
invSD2 = np.diag(1.0 / np.diag(invSD))

# Fixed covariance matrix
fixed = invSD2 @ Y @ invSD2


# ---------------------------------------------------------------------------------------------------------------------------
# Step 4: PSD-safe Cholesky decomposition
# ---------------------------------------------------------------------------------------------------------------------------

root = np.zeros((n, n))


for j in range(n):

    # Calculate sum of previous terms
    if j > 0:
        s = root[j, :j] @ root[j, :j].T
    else:
        s = 0.0

    # Diagonal element
    temp = fixed[j, j] - s

    # Numerical tolerance
    if -1e-8 < temp < 0:
        temp = 0.0

    # Cholesky diagonal
    root[j, j] = np.sqrt(temp) if temp > 0 else 0.0

    # If diagonal is zero, skip this column
    if root[j, j] == 0.0:
        continue

    # Calculate off-diagonal elements
    ir = 1.0 / root[j, j]

    for i in range(j + 1, n):

        if j > 0:
            s = root[i, :j] @ root[j, :j].T
        else:
            s = 0.0

        root[i, j] = (fixed[i, j] - s) * ir


# Lower-triangular Cholesky factor
L = root


# ---------------------------------------------------------------------------------------------------------------------------
# Step 5: Simulate 100,000 multivariate Normal observations with mean 0
# ---------------------------------------------------------------------------------------------------------------------------

n_sims = 100000

# Independent standard Normal variables
z = np.random.normal(
    size=(n, n_sims)
)

# Generate correlated Normal observations
sim = (L @ z).T


# ---------------------------------------------------------------------------------------------------------------------------
# Step 6: Calculate covariance of simulated output
# ---------------------------------------------------------------------------------------------------------------------------

sim_cov = np.cov(sim, rowvar=False)

sim_cov_df = pd.DataFrame(sim_cov, columns=cols)
print(sim_cov_df)





# ---------------------------------------------------------------------------------------------------------------------------
# Test 5.5. PCA Simulation, 99% explained, 0 mean - 100,000 simulations, compare input vs output covariance
# ---------------------------------------------------------------------------------------------------------------------------

np.random.seed(42)
data = pd.read_csv("test5_2.csv")
cols = data.columns

# ---- Load input covariance matrix ----
a = data.values
n = a.shape[0]


# ---------------------------------------------------------------------------------------------------------------------------
# Step 1: Eigenvalue decomposition of covariance matrix
# ---------------------------------------------------------------------------------------------------------------------------
vals, vecs = np.linalg.eigh(a)

# Sort eigenvalues from largest to smallest
idx = np.argsort(vals)[::-1]

vals = vals[idx]
vecs = vecs[:, idx]


# ---------------------------------------------------------------------------------------------------------------------------
# Step 2: Determine number of principal components explaining at least 99% variance
# ---------------------------------------------------------------------------------------------------------------------------

total_variance = np.sum(vals)

explained_variance_ratio = vals / total_variance

cumulative_variance = np.cumsum(explained_variance_ratio)

# Smallest number of PCs explaining at least 99%
k = np.searchsorted(cumulative_variance, 0.99) + 1


# ---------------------------------------------------------------------------------------------------------------------------
# Step 3: Keep the first k principal components
# ---------------------------------------------------------------------------------------------------------------------------

eigenvalues = vals[:k]
eigenvectors = vecs[:, :k]


# ---------------------------------------------------------------------------------------------------------------------------
# Step 4: Construct PCA simulation matrix
# ---------------------------------------------------------------------------------------------------------------------------

# Independent standard Normal variables
n_sims = 100000

z = np.random.normal(
    size=(k, n_sims)
)

# Scale each principal component by sqrt(eigenvalue)
sqrt_eigenvalues = np.sqrt(eigenvalues)

# PCA loading matrix
B = eigenvectors @ np.diag(sqrt_eigenvalues)


# ---------------------------------------------------------------------------------------------------------------------------
# Step 5: Simulate 100,000 observations with mean 0
# ---------------------------------------------------------------------------------------------------------------------------

# X = BZ
sim = (B @ z).T


# ---------------------------------------------------------------------------------------------------------------------------
# Step 6: Calculate covariance of simulated output
# ---------------------------------------------------------------------------------------------------------------------------

sim_cov = np.cov(
    sim,
    rowvar=False
)

sim_cov_df = pd.DataFrame(
    sim_cov,
    columns=cols
)

print(sim_cov_df)