import numpy as np
import pandas as pd

df = pd.read_csv("problem1.csv")
cols = df.columns

# part a. Pairwise correlation matrix
def pairwise_corr(data: pd.DataFrame) -> pd.DataFrame:
    n = data.shape[1]
    R = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            both = data.iloc[:, [i, j]].dropna()
            R[i, j] = R[j, i] = np.corrcoef(both.iloc[:, 0], both.iloc[:, 1])[0, 1]
    return pd.DataFrame(R, index=data.columns, columns=data.columns)

R = pairwise_corr(df)
assert np.allclose(R.values, df.corr().values)  # sanity check vs pandas
print("a. Pairwise correlation matrix\n", R, "\n")

# Overlapping observation counts per pair (useful to report)
counts = df.notna().astype(int).T @ df.notna().astype(int)
print("Pairwise observation counts\n", counts, "\n")



# part b. Variance of each series using all of its own observations
var = df.var(ddof=1)  # NaNs skipped per column
print("b. Variances (ddof=1)")
print(pd.DataFrame({"N": df.count(), "variance": var}), "\n")



# part c. Positive semi-definiteness check
def is_psd(A: np.ndarray, tol: float = 1e-8):
    eigvals, eigvecs = np.linalg.eigh(A)  # symmetric -> eigh, sorted ascending
    return eigvals.min() >= -tol, eigvals, eigvecs

psd, eigvals, eigvecs = is_psd(R.values)
print("c. Eigenvalues of R:", eigvals)
print("   PSD?", psd)
if not psd:
    v = eigvecs[:, 0]
    print("   Eigenvector of min eigenvalue:", v)
    print("   v' R v =", v @ R.values @ v, "(< 0, so R is not PSD)")
try:
    np.linalg.cholesky(R.values)
    print("   Cholesky succeeded")
except np.linalg.LinAlgError:
    print("   Cholesky fails -> R is not positive definite")
print()



# part d. Higham (2002) nearest correlation matrix, W = I (Frobenius norm)
def proj_psd(A: np.ndarray) -> np.ndarray:
    """Project onto the PSD cone: zero out negative eigenvalues."""
    w, V = np.linalg.eigh(A)
    return V @ np.diag(np.maximum(w, 0.0)) @ V.T

def proj_unit_diag(A: np.ndarray) -> np.ndarray:
    """Project onto symmetric matrices with unit diagonal."""
    B = A.copy()
    np.fill_diagonal(B, 1.0)
    return B

def higham_nearest_corr(A: np.ndarray, tol: float = 1e-10, max_iter: int = 1000):
    Y = A.copy()
    dS = np.zeros_like(A)          # Dykstra correction
    for k in range(1, max_iter + 1):
        Y_prev = Y
        Rk = Y - dS
        X = proj_psd(Rk)
        dS = X - Rk
        Y = proj_unit_diag(X)
        if np.linalg.norm(Y - Y_prev, "fro") < tol:
            break
    return Y, k

if not psd:
    C, n_iter = higham_nearest_corr(R.values)
else:
    C, n_iter = R.values.copy(), 0

C_df = pd.DataFrame(C, index=cols, columns=cols)
print(f"d. Nearest PSD correlation matrix (Higham, {n_iter} iterations)\n", C_df)
print("   Eigenvalues:", np.linalg.eigvalsh(C))

diff = C_df - R
print("\n   Change (Higham - original)\n", diff)
print("   Frobenius distance:", np.linalg.norm(diff.values, "fro"))

# Rank the off-diagonal changes by absolute size
iu = np.triu_indices(len(cols), k=1)
changes = pd.Series(
    diff.values[iu],
    index=[f"{cols[i]}-{cols[j]}" for i, j in zip(*iu)],
).sort_values(key=np.abs, ascending=False)
print("\n   Largest changes\n", changes, "\n")



# part e. Covariance = D C D, D = diag(std devs from part b)
sd = np.sqrt(var.values)
Sigma = np.diag(sd) @ C @ np.diag(sd)
Sigma_df = pd.DataFrame(Sigma, index=cols, columns=cols)
print("e. Covariance matrix\n", Sigma_df.map(lambda x: f"{x:.4e}"))
print("   Eigenvalues:", np.linalg.eigvalsh(Sigma))
print("   Diagonal matches part b:", np.allclose(np.diag(Sigma), var.values))
