import numpy as np
import pandas as pd
from scipy.stats import norm, t, norminvgauss, kendalltau
from scipy.linalg import cholesky
from scipy.optimize import minimize_scalar

from risklib import information_criteria, ensure_pd, fmt_params, gaussian_copula_loglik, t_copula_loglik, R_for_nu, var_es

N_SIM = 5000
ALPHA = 0.05
SEED = 545

# 1. Load the data
prices = pd.read_csv("problem3.csv")
if not pd.api.types.is_numeric_dtype(prices.iloc[:, 0]):
    prices = prices.set_index(prices.columns[0])
holdings = pd.read_csv("problem3_holdings.csv").set_index("Stock")
stocks = list(holdings.index)
prices = prices[stocks]

returns = prices.pct_change().dropna()          # arithmetic returns, 1,000 rows
last_price = prices.iloc[-1]
position_value = holdings["Holding"] * last_price
print("Position values at last price\n",
      pd.DataFrame({"Book": holdings["Book"], "Shares": holdings["Holding"],
                    "Last price": last_price, "Value": position_value}).round(2), "\n")



# part a. Marginal fits: Normal, Student-t, NIG
DISTS = {"Normal": (norm, 2), "Student-t": (t, 3), "NIG": (norminvgauss, 4)}
n_obs = len(returns)

fit_rows, params = [], {}
for s in stocks:
    x = returns[s].to_numpy()
    for name, (dist, k) in DISTS.items():
        p = dist.fit(x)
        ll = float(np.sum(dist.logpdf(x, *p)))
        aicc, bic = information_criteria(ll, n_obs, k)
        params[(s, name)] = p
        fit_rows.append({"Stock": s, "Dist": name, "Params": p,
                         "LogLik": ll, "AICc": aicc, "BIC": bic})

fits = pd.DataFrame(fit_rows)

fits["Parameters"] = fits.apply(fmt_params, axis=1)
print("a. Fitted marginal distributions")
print(fits[["Stock", "Dist", "Parameters", "LogLik", "AICc", "BIC"]]
      .round({"LogLik": 2, "AICc": 2, "BIC": 2}).to_string(index=False), "\n")

best = fits.loc[fits.groupby("Stock")["AICc"].idxmin()].set_index("Stock").loc[stocks]
print("Best fit by AICc (BIC choice shown for comparison)")
bic_best = fits.loc[fits.groupby("Stock")["BIC"].idxmin()].set_index("Stock")["Dist"]
print(pd.DataFrame({"AICc choice": best["Dist"], "BIC choice": bic_best.loc[stocks]}), "\n")

marginals = {s: (DISTS[best.loc[s, "Dist"]][0], best.loc[s, "Params"]) for s in stocks}



# part b. Copula fits
U = np.column_stack([marginals[s][0].cdf(returns[s], *marginals[s][1]) for s in stocks])
U = np.clip(U, 1e-10, 1 - 1e-10)
d = U.shape[1]

# Gaussian copula: MLE of R is (approximately) the correlation of Z = Phi^-1(U)
Z = norm.ppf(U)
R_gauss = ensure_pd(np.corrcoef(Z, rowvar=False))

res = minimize_scalar(lambda lnu: -t_copula_loglik(U, R_for_nu(U, np.exp(lnu)), np.exp(lnu)),
                      bounds=(np.log(2.01), np.log(500)), method="bounded")
nu_hat = float(np.exp(res.x))
R_t = R_for_nu(U, nu_hat)

# Cross-check against the Kendall-tau correlation (reported, not used)
R_tau = np.eye(d)
for i in range(d):
    for j in range(i + 1, d):
        R_tau[i, j] = R_tau[j, i] = np.sin(np.pi * kendalltau(U[:, i], U[:, j]).statistic / 2)
print(f"Kendall-tau R: min eigenvalue {np.linalg.eigvalsh(R_tau).min():.4f}, "
      f"max |R_tau - R_t| = {np.abs(R_tau - R_t).max():.4f}")

ll_g = gaussian_copula_loglik(U, R_gauss)
ll_t = t_copula_loglik(U, R_t, nu_hat)
k_corr = d * (d - 1) // 2
aicc_g, bic_g = information_criteria(ll_g, n_obs, k_corr)
aicc_t, bic_t = information_criteria(ll_t, n_obs, k_corr + 1)

print("b. Copula comparison")
print(pd.DataFrame({
    "LogLik": [ll_g, ll_t], "k": [k_corr, k_corr + 1],
    "AICc": [aicc_g, aicc_t], "BIC": [bic_g, bic_t],
}, index=["Gaussian", f"Student-t (nu={nu_hat:.2f})"]).round(2))
use_t = aicc_t < aicc_g
print("Chosen copula:", f"Student-t, nu={nu_hat:.2f}" if use_t else "Gaussian", "\n")

print("Copula correlation matrix used\n",
      pd.DataFrame(R_t if use_t else R_gauss, index=stocks, columns=stocks).round(3), "\n")



# part c. Simulation, VaR and ES
rng = np.random.default_rng(SEED)
R_use = R_t if use_t else R_gauss
L = cholesky(R_use, lower=True)
Zs = rng.standard_normal((N_SIM, d)) @ L.T
if use_t:
    W = rng.chisquare(nu_hat, N_SIM)
    Us = t.cdf(Zs / np.sqrt(W / nu_hat)[:, None], df=nu_hat)
else:
    Us = norm.cdf(Zs)
Us = np.clip(Us, 1e-10, 1 - 1e-10)

sim_ret = np.column_stack([marginals[s][0].ppf(Us[:, j], *marginals[s][1])
                           for j, s in enumerate(stocks)])
pnl = pd.DataFrame(sim_ret * position_value.to_numpy(), columns=stocks)

rows = []
for s in stocks:
    v, e = var_es(pnl[s], ALPHA)
    rows.append({"Level": "Stock", "Name": s, "Book": holdings.loc[s, "Book"],
                 "Value": position_value[s], "VaR95": v, "ES95": e})

book_sum = {}
for b in sorted(holdings["Book"].unique()):
    members = holdings.index[holdings["Book"] == b]
    v, e = var_es(pnl[members].sum(axis=1), ALPHA)
    book_sum[b] = (sum(r["VaR95"] for r in rows if r["Book"] == b),
                   sum(r["ES95"] for r in rows if r["Book"] == b))
    rows.append({"Level": "Book", "Name": f"Book {b}", "Book": b,
                 "Value": position_value[members].sum(), "VaR95": v, "ES95": e})

v, e = var_es(pnl.sum(axis=1), ALPHA)
rows.append({"Level": "Total", "Name": "Portfolio", "Book": "A+B",
             "Value": position_value.sum(), "VaR95": v, "ES95": e})

result = pd.DataFrame(rows)
result["VaR % of value"] = 100 * result["VaR95"] / result["Value"]
result["ES % of value"] = 100 * result["ES95"] / result["Value"]
print(f"c. 95% absolute VaR and ES ($), {N_SIM:,} simulations")
print(result.round(2).to_string(index=False), "\n")

# Diversification diagnostics
book_rows = result[result["Level"] == "Book"].set_index("Book")
total = result[result["Level"] == "Total"].iloc[0]
print("Diversification benefit (sum of parts - combined)")
for b, (sv, se) in book_sum.items():
    print(f"  Book {b}: VaR {sv - book_rows.loc[b, 'VaR95']:.2f}, "
          f"ES {se - book_rows.loc[b, 'ES95']:.2f}")
print(f"  Books -> total: VaR {book_rows['VaR95'].sum() - total['VaR95']:.2f}, "
      f"ES {book_rows['ES95'].sum() - total['ES95']:.2f}")

