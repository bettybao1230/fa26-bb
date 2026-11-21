"""VaR and ES calculations: Questions 8.1-8.6."""

import numpy as np
import pandas as pd
from scipy import stats

from risk_library_8_9 import (
    fit_normal,
    fit_student_t,
    var_normal,
    var_student_t,
    es_normal,
    es_student_t,
    var_es_from_simulation,
)

# Significance level
ALPHA = 0.05

# Load data
x1 = pd.read_csv("test7_1.csv")["x1"].to_numpy()
x2 = pd.read_csv("test7_2.csv")["x1"].to_numpy()

# Fit normal distribution
mu_n, sigma_n = fit_normal(x1)

# Fit Student-t distribution
mu_t, sigma_t, nu = fit_student_t(x2)

print(f"Student-t parameters: mu={mu_t:.6f}, "
      f"sigma={sigma_t:.6f}, nu={nu:.4f}")

# Simulate 1,000,000 observations from the fitted Student-t distribution
rng = np.random.default_rng(42)

sim = stats.t.rvs(
    df=nu,
    loc=mu_t,
    scale=sigma_t,
    size=1_000_000,
    random_state=rng,
)

# Question 8.1: VaR - Normal distribution
result_8_1 = var_normal(mu_n, sigma_n, ALPHA)

# Question 8.2: VaR - Student-t distribution
result_8_2 = var_student_t(mu_t, sigma_t, nu, ALPHA)

# Question 8.3: VaR - Simulation
result_8_3, result_8_6 = var_es_from_simulation(sim, ALPHA)

# Question 8.4: ES - Normal distribution
result_8_4 = es_normal(mu_n, sigma_n, ALPHA)

# Question 8.5: ES - Student-t distribution
result_8_5 = es_student_t(mu_t, sigma_t, nu, ALPHA)

# 8.1 VaR - Normal
table_8_1 = pd.DataFrame(
    {"VaR Absolute": [result_8_1[0]],
     "VaR Diff from Mean": [result_8_1[1]]}
)
print("\n8.1 VaR (Normal)")
print(table_8_1.to_string(index=False))


# 8.2 VaR - Student-t
table_8_2 = pd.DataFrame(
    {"VaR Absolute": [result_8_2[0]],
     "VaR Diff from Mean": [result_8_2[1]]}
)
print("\n8.2 VaR (Student-t)")
print(table_8_2.to_string(index=False))


# 8.3 VaR - Simulation
table_8_3 = pd.DataFrame(
    {"VaR Absolute": [result_8_3[0]],
     "VaR Diff from Mean": [result_8_3[1]]}
)
print("\n8.3 VaR (Simulation)")
print(table_8_3.to_string(index=False))


# 8.4 ES - Normal
table_8_4 = pd.DataFrame(
    {"ES Absolute": [result_8_4[0]],
     "ES Diff from Mean": [result_8_4[1]]}
)
print("\n8.4 ES (Normal)")
print(table_8_4.to_string(index=False))


# 8.5 ES - Student-t
table_8_5 = pd.DataFrame(
    {"ES Absolute": [result_8_5[0]],
     "ES Diff from Mean": [result_8_5[1]]}
)
print("\n8.5 ES (Student-t)")
print(table_8_5.to_string(index=False))


# 8.6 ES - Simulation
table_8_6 = pd.DataFrame(
    {"ES Absolute": [result_8_6[0]],
     "ES Diff from Mean": [result_8_6[1]]}
)
print("\n8.6 ES (Simulation)")
print(table_8_6.to_string(index=False))