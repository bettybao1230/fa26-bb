import numpy as np
import pandas as pd


def load_ff(path):
    # Ken French daily file: yyyymmdd dates, values in percent
    df = pd.read_csv(path, index_col=0)
    df.columns = df.columns.str.strip()
    df = df[df.index.astype(str).str.strip().str.len() == 8]
    df.index = pd.to_datetime(df.index.astype(str).str.strip(), format="%Y%m%d")
    return df.astype(float) / 100


def load_funds(path):
    return pd.read_csv(path, index_col=0, parse_dates=True)


def summary(excess, ann=252):
    mean = excess.mean() * ann
    vol = excess.std() * np.sqrt(ann)
    return pd.DataFrame({"mean": mean, "vol": vol, "sharpe": mean / vol})


def ols(y, X):
    # regress y on a constant and X, return coefficients, standard errors, t, R2, AICc, residuals
    Xc = np.column_stack([np.ones(len(y)), X.values])
    n, k = Xc.shape
    XtX_inv = np.linalg.inv(Xc.T @ Xc)
    b = XtX_inv @ Xc.T @ y.values
    resid = y.values - Xc @ b
    ssr = resid @ resid
    se = np.sqrt(np.diag(XtX_inv) * ssr / (n - k))
    r2 = 1 - ssr / ((y - y.mean()) ** 2).sum()

    p = k + 1  # coefficients plus error variance
    aic = n * np.log(2 * np.pi * ssr / n) + n + 2 * p
    aicc = aic + 2 * p * (p + 1) / (n - p - 1)

    names = ["alpha"] + list(X.columns)
    return {
        "coef": pd.Series(b, index=names),
        "se": pd.Series(se, index=names),
        "t": pd.Series(b / se, index=names),
        "r2": r2,
        "aicc": aicc,
        "resid": pd.Series(resid, index=y.index),
    }