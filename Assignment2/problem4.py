import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import norm, t, rankdata

from risklib import (
    four_moments,
    fit_distribution,
    calculate_log_likelihood,
    calculate_aicc,
    transform_to_uniform,
    kendall_tau_to_correlation,
    gaussian_copula_loglik,
    fit_t_copula_df,
    information_criteria,
    simulate_gaussian_copula,
    simulate_t_copula,
    transform_uniform_to_margins,
    compute_pnl,
    historical_es,
    historical_var,
    joint_tail_days,
    copula_daily_loglik_difference,
)


# ============================================================
# Load Data
# ============================================================

assets = ["X1", "X2", "X3"]

df = pd.read_csv("problem4.csv")
df[assets] = df[assets].apply(pd.to_numeric, errors="coerce")


# ============================================================
# Part (a): Sample Moments and Extreme-Return Sensitivity
# ============================================================
for asset in assets:
    # Exclude missing observations before calculating moments.
    x = df[asset].dropna().to_numpy()

    original = four_moments(x)

    print(f"\n{asset}")
    print("Original moments (mean, variance, skewness, kurtosis):")
    print(np.round(original, 6))

    # Remove the observation with the largest absolute return.
    extreme_idx = np.argmax(np.abs(x))
    x_reduced = np.delete(x, extreme_idx)

    revised = four_moments(x_reduced)

    print(f"Extreme return:      {x[extreme_idx]:.6f}")
    print(f"Original kurtosis:    {original[3]:.4f}")
    print(f"Revised kurtosis:     {revised[3]:.4f}")
    print(f"Change in kurtosis:   {revised[3] - original[3]:.4f}")


# ============================================================
# Part (b): Historical Joint-Tail Dependence
# ============================================================
# Convert returns to scaled ranks in (0, 1), giving each asset
# a comparable uniform scale without assuming a marginal model.
U_ranks = pd.DataFrame({
    asset: rankdata(df[asset].dropna()) / (df[asset].count() + 1)
    for asset in assets
})

# Use complete observations so that all pairs share the same
# sample size and observation dates.
complete_returns = df[assets].dropna()
U_ranks = pd.DataFrame(
    {
        asset: rankdata(complete_returns[asset])
        / (len(complete_returns) + 1)
        for asset in assets
    },
    index=complete_returns.index,
)

pairs = [("X1", "X2"), ("X1", "X3"), ("X2", "X3")]
tail = 0.025
n = len(U_ranks)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for ax, (a, b) in zip(axes, pairs):
    u = U_ranks[a]
    v = U_ranks[b]

    # Compare the ranks and mark the lower and upper 2.5% tails.
    ax.scatter(u, v, s=12, alpha=0.5)
    ax.axvline(tail, color="red", linestyle="--")
    ax.axhline(tail, color="red", linestyle="--")
    ax.axvline(1 - tail, color="green", linestyle="--")
    ax.axhline(1 - tail, color="green", linestyle="--")

    ax.set_xlabel(a)
    ax.set_ylabel(b)
    ax.set_title(f"{a} vs {b}")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # Count days when both assets fall in the same extreme tail.
    lower = ((u <= tail) & (v <= tail)).sum()
    upper = ((u >= 1 - tail) & (v >= 1 - tail)).sum()

    # Under independence, the joint-tail probability is tail^2.
    expected = n * tail**2

    print(f"\n{a} and {b}")
    print(f"Joint worst 2.5% days: {lower}")
    print(f"Joint best 2.5% days:  {upper}")
    print(f"Expected under independence: {expected:.3f}")

plt.tight_layout()
plt.show()


# ============================================================
# Part (c): Fit Marginal Distributions and Compare AICc
# ============================================================
# Fit a Normal distribution and a Student-t distribution to
# each asset's returns. The lower AICc indicates the preferred fit.
marginal_choices = {}
marginal_specs = {}

for asset in assets:
    x = df[asset].dropna().to_numpy()

    results = {}

    for name, distribution, k in [
        ("Normal", norm, 2),
        ("Student-t", t, 3),
    ]:
        params = fit_distribution(x, distribution)
        loglik = calculate_log_likelihood(x, distribution, params)
        aicc = calculate_aicc(loglik, len(x), k)

        results[name] = {
            "distribution": distribution,
            "params": params,
            "loglik": loglik,
            "aicc": aicc,
        }

    preferred = min(results, key=lambda name: results[name]["aicc"])
    marginal_choices[asset] = preferred

    selected = results[preferred]
    marginal_specs[asset] = (
        selected["distribution"],
        selected["params"],
    )

    print(f"\n{asset}")
    for name, result in results.items():
        print(f"{name} AICc: {result['aicc']:.4f}")

    print(f"Preferred distribution: {preferred}")


# ============================================================
# Part (d): Fit Gaussian and Student-t Copulas
# ============================================================
# Use complete observations to estimate dependence consistently
# across all three assets.
complete_df = df[assets].dropna()

# Transform each return series through its fitted marginal CDF.
U = transform_to_uniform(complete_df, marginal_specs)

# Convert Kendall's tau estimates into a copula correlation matrix.
R = kendall_tau_to_correlation(U)

print("\nSelected marginal distributions:")
print(pd.Series(marginal_choices, name="Distribution"))

print("\nEstimated copula correlation matrix:")
print(pd.DataFrame(R, index=assets, columns=assets))

n, d = U.shape
k_corr = d * (d - 1) // 2

# The Gaussian copula estimates the dependence matrix only.
ll_gaussian = gaussian_copula_loglik(U, R)
ic_gaussian = information_criteria(ll_gaussian, n, k_corr)

# The Student-t copula also estimates its degrees of freedom.
nu_hat, ll_t = fit_t_copula_df(U, R)
ic_t = information_criteria(ll_t, n, k_corr + 1)

print("\nGaussian copula")
print(f"Log-likelihood: {ll_gaussian:.4f}")
print(f"AICc:           {ic_gaussian['AICc']:.4f}")
print(f"BIC:            {ic_gaussian['BIC']:.4f}")

print("\nStudent-t copula")
print(f"Log-likelihood: {ll_t:.4f}")
print(f"Estimated nu:   {nu_hat:.4f}")
print(f"AICc:           {ic_t['AICc']:.4f}")
print(f"BIC:            {ic_t['BIC']:.4f}")


# ============================================================
# Part (e): Simulate Copulas and Calculate VaR / ES
# ============================================================
investment = 1_000_000
n_simulations = 100_000
alphas = [0.05, 0.01]

# Simulate uniform observations from both fitted copulas.
U_gaussian = simulate_gaussian_copula(
    n_simulations, R, seed=123
)

U_t = simulate_t_copula(
    n_simulations, R, nu_hat, seed=456
)

# Transform simulated uniforms into returns using the fitted margins.
sim_gaussian = transform_uniform_to_margins(
    U_gaussian, marginal_specs
)

sim_t = transform_uniform_to_margins(
    U_t, marginal_specs
)

# Each asset has a $1 million position.
# Compute P&L in dollars, assuming simple returns and fixed
# dollar exposures over the one-day horizon.
historical_pnl = investment * complete_df[assets].to_numpy().sum(axis=1)
gaussian_pnl = investment * sim_gaussian.sum(axis=1)
t_pnl = investment * sim_t.sum(axis=1)

portfolio_value = len(assets) * investment

# Express portfolio P&L as a return on the total $3 million value.
historical_returns = historical_pnl / portfolio_value
gaussian_returns = gaussian_pnl / portfolio_value
t_returns = t_pnl / portfolio_value

# Compare historical estimates with estimates from both copulas.
results = []

for alpha in alphas:
    for method, returns, pnl in [
        ("Historical", historical_returns, historical_pnl),
        ("Gaussian copula", gaussian_returns, gaussian_pnl),
        ("Student-t copula", t_returns, t_pnl),
    ]:
        var = historical_var(returns, portfolio_value, alpha)
        es = historical_es(pnl, alpha)

        results.append({
            "Method": method,
            "Confidence level": f"{100 * (1 - alpha):.0f}%",
            "VaR ($)": round(var, 2),
            "ES ($)": round(es, 2),
        })

results_df = pd.DataFrame(results)
print("\nPortfolio VaR and ES:")
print(results_df.to_string(index=False))


# ============================================================
# Part (f): Compare Joint-Tail Events
# ============================================================
# Count simulated or historical observations in which asset pairs
# experience joint extreme losses or gains.
historical_results = joint_tail_days(
    complete_df[assets].to_numpy(),
    0.025,
    1000,
    assets=assets,
)
historical_results.insert(0, "Method", "Historical")

gaussian_results = joint_tail_days(
    sim_gaussian,
    0.025,
    1000,
    assets=assets,
)
gaussian_results.insert(0, "Method", "Gaussian copula")

t_results = joint_tail_days(
    sim_t,
    0.025,
    1000,
    assets=assets,
)
t_results.insert(0, "Method", "Student-t copula")

joint_tail_results = pd.concat(
    [historical_results, gaussian_results, t_results],
    ignore_index=True,
)

print("\nJoint-tail comparison:")
print(joint_tail_results.round(2).to_string(index=False))

print(
    "\nExpected joint-tail days under independence:",
    1000 * 0.025**2,
)


# ============================================================
# Part (g): Contribution of Central Observations to the
#           Log-Likelihood Difference
# ============================================================
# Calculate each observation's contribution to the difference
# between the fitted Student-t and Gaussian copula log-likelihoods.
delta_ll = copula_daily_loglik_difference(U, R, nu_hat)

# Identify observations for which every asset lies between
# its 5th and 95th percentiles.
central = np.all((U >= 0.05) & (U <= 0.95), axis=1)

total_difference = delta_ll.sum()
central_difference = delta_ll[central].sum()

central_share = (
    100 * central_difference / total_difference
    if total_difference != 0
    else np.nan
)

print(f"\nTotal log-likelihood difference: {total_difference:.4f}")
print(f"Central-day contribution:        {central_difference:.4f}")
print(f"Central-day share:               {central_share:.2f}%")

