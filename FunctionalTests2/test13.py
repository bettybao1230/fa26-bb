"""
Question 13: Kendall Correlation, Multivariate t Fit, Copulas,
Tail Dependence, and Portfolio VaR/ES.

All calculations use functions from risk_lib_13.
"""

import pandas as pd
import risk_lib_13 as rl


# ============================================================
# 1. Settings and Input Data
# ============================================================
ALPHA = 0.05
N_SIM = 1_000_000

returns = pd.read_csv("test13_returns.csv")
portfolio = pd.read_csv("test13_portfolio.csv")

asset_names = list(returns.columns)
return_data = returns.to_numpy()

n_observations, n_assets = return_data.shape


# ============================================================
# 2. Kendall Correlation (Question 13.1)
# ============================================================
correlation = rl.kendall_correlation(return_data)


# ============================================================
# 3. Multivariate t Distribution Fit (Questions 13.2–13.4)
# ============================================================
mvt_mean, mvt_scale, mvt_df, mvt_loglik = rl.fit_mvt_with_corr(
    return_data,
    correlation
)


# ============================================================
# 4. Fit t Marginal Distributions (Question 13.5)
# ============================================================
marginals = [
    rl.FittedMarginal.fit(return_data[:, j], "T")
    for j in range(n_assets)
]

uniform_data = rl.to_uniforms(return_data, marginals)


# ============================================================
# 5. Gaussian Copula Log Likelihood (Question 13.5)
# ============================================================
gaussian_loglik = rl.gaussian_copula_loglik(
    uniform_data,
    correlation
)


# ============================================================
# 6. Fit t Copula Degrees of Freedom (Question 13.6)
# ============================================================
copula_df, t_copula_loglik = rl.fit_t_copula_nu(
    uniform_data,
    correlation
)


# ============================================================
# 7. Copula Model Selection (Question 13.7)
# ============================================================
model_comparison = pd.DataFrame({
    "Copula": ["Gaussian", "T"],
    "LL": [gaussian_loglik, t_copula_loglik],
    "K": [0, 1]
})

model_comparison["AICC"] = [
    rl.aicc(loglik, parameters, n_observations)
    for loglik, parameters in zip(
        model_comparison["LL"],
        model_comparison["K"]
    )
]

model_comparison["BIC"] = [
    rl.bic(loglik, parameters, n_observations)
    for loglik, parameters in zip(
        model_comparison["LL"],
        model_comparison["K"]
    )
]


# ============================================================
# 8. Lower Tail Dependence (Question 13.8)
# ============================================================
tail_dependence = []

for i in range(n_assets):
    for j in range(i + 1, n_assets):

        rho = correlation[i, j]

        tail_dependence.append({
            "I": i + 1,
            "J": j + 1,
            "Rho": rho,
            "Lambda": rl.t_tail_dependence(rho, copula_df)
        })

tail_results = pd.DataFrame(tail_dependence)


# ============================================================
# 9. Portfolio VaR and ES under the t Copula (Question 13.9)
# ============================================================
portfolio_values = (
    portfolio.set_index("Stock")
    .loc[asset_names, "currentValue"]
    .to_numpy()
)

simulated_returns = rl.t_copula_simulate(
    marginals,
    correlation,
    copula_df,
    n=N_SIM
)

simulated_pnl = simulated_returns * portfolio_values

portfolio_risk = rl.portfolio_var_es(
    simulated_pnl,
    asset_names,
    portfolio_values,
    ALPHA
)


# ============================================================
# 10. Print Results
# ============================================================

# 13.1 Kendall Correlation
print("\nQuestion 13.1: Kendall Correlation")
print("=" * 70)

display_names = [
    f"x{i + 1}" for i in range(n_assets)
]

correlation_table = pd.DataFrame(
    correlation,
    columns=display_names
)

print(
    correlation_table.to_string(
        index=False,
        float_format=lambda x: repr(float(x))
    )
)


# Question 13.2: Mean Vector
print("\nQuestion 13.2: Mean Vector")
print("=" * 70)

mean_table = pd.DataFrame({
    "mu": mvt_mean
})

print(
    mean_table.to_string(
        index=False,
        float_format=lambda x: repr(float(x))
    )
)

# Question 13.3: Scale Matrix
print("\nQuestion 13.3: Scale Matrix")
print("=" * 70)

display_names = [
    f"x{i + 1}" for i in range(n_assets)
]

scale_table = pd.DataFrame(
    mvt_scale,
    columns=display_names
)

print(
    scale_table.to_string(
        index=False,
        float_format=lambda x: repr(float(x))
    )
)

# Question 13.4: Degrees of Freedom and Log Likelihood
print("\nQuestion 13.4: Multivariate t Parameters")
print("=" * 70)

result_13_4 = pd.DataFrame({
    "nu": [mvt_df],
    "ll": [mvt_loglik]
})

print(
    result_13_4.to_string(
        index=False,
        float_format=lambda x: repr(float(x))
    )
)


# Question 13.5: Gaussian Copula Log Likelihood
print("\nQuestion 13.5: Gaussian Copula")
print("=" * 70)

result_13_5 = pd.DataFrame({
    "ll": [gaussian_loglik]
})

print(
    result_13_5.to_string(
        index=False,
        float_format=lambda x: repr(float(x))
    )
)


# Question 13.6: t Copula Parameters
print("\nQuestion 13.6: t Copula")
print("=" * 70)

result_13_6 = pd.DataFrame({
    "nu": [copula_df],
    "ll": [t_copula_loglik]
})

print(
    result_13_6.to_string(
        index=False,
        float_format=lambda x: repr(float(x))
    )
)


# 13.7 Model Comparison
print("\nQuestion 13.7: Copula Model Comparison")
print("=" * 70)

print(model_comparison.to_string(
    index=False,
    float_format=lambda x: repr(float(x))
))


# 13.8 Lower Tail Dependence
print("\nQuestion 13.8: Lower Tail Dependence")
print("=" * 70)

print(tail_results.to_string(
    index=False,
    float_format=lambda x: repr(float(x))
))


# 13.9 Portfolio VaR and ES
print("\nQuestion 13.9: Portfolio VaR and ES under the t Copula")
print("=" * 70)

print(portfolio_risk.to_string(
    index=False,
    float_format=lambda x: repr(float(x))
))
