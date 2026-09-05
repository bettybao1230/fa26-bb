# ==================================================
# Problem 3:  Pearson Against Spearman.
# ==================================================
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from itertools import combinations


# ==================================================
# 1. Load the data
# ==================================================

data3 = pd.read_csv("problem3.csv")

# Variables used for the pairwise correlation analysis.
variables = ["x1", "x2", "x3", "x4"]


# ==================================================
# 2. Compute pairwise Pearson and Spearman correlations
# ==================================================

# Generate every unique pair of variables.
# For four variables, this produces six pairs:
# (x1, x2), (x1, x3), (x1, x4),
# (x2, x3), (x2, x4), and (x3, x4).

results = []

for x_name, y_name in combinations(variables, 2):

    x = data3[x_name]
    y = data3[y_name]

    # Pearson correlation measures the strength of
    # the linear relationship between the two variables.
    pearson = pearsonr(x, y).statistic

    # Spearman correlation measures the strength of
    # the monotonic relationship using the ranks of
    # the observations.
    spearman = spearmanr(x, y).statistic

    # A large difference between the two correlations
    # may indicate that the relationship is not well
    # described by a simple linear relationship.
    difference = abs(pearson - spearman)

    results.append({
        "Pair": f"{x_name} vs {y_name}",
        "Pearson": pearson,
        "Spearman": spearman,
        "Difference": difference
    })


# Store the results in a DataFrame so that they can
# be easily printed, compared, and sorted.
results_df = pd.DataFrame(results)


# ==================================================
# 3. Display the pairwise correlations
# ==================================================

print("Pairwise correlations:")
print()

for _, row in results_df.iterrows():

    print(row["Pair"])
    print(f"  Pearson:    {row['Pearson']:.4f}")
    print(f"  Spearman:   {row['Spearman']:.4f}")
    print(f"  Difference: {row['Difference']:.4f}")
    print()


# ==================================================
# 4. Plot all pairwise relationships
# ==================================================

# There are six unique pairs, so arrange the
# scatterplots in a 2 x 3 grid.
pairs = list(combinations(variables, 2))

fig, axes = plt.subplots(
    2,
    3,
    figsize=(12, 8)
)

for ax, (x_name, y_name) in zip(axes.flat, pairs):

    # Plot the observed values for each pair.
    # The scatterplots help visualize whether the
    # relationship appears linear, nonlinear, or
    # otherwise unusual.
    ax.scatter(
        data3[x_name],
        data3[y_name],
        alpha=0.6
    )

    ax.set_xlabel(x_name)
    ax.set_ylabel(y_name)
    ax.set_title(f"{x_name} vs {y_name}")


plt.tight_layout()
plt.show()


# ==================================================
# 5. Compute the correlation matrices
# ==================================================

# Pearson correlation matrix summarizes the linear
# association between every pair of variables.
pearson_matrix = data3[variables].corr(method="pearson")

# Spearman correlation matrix summarizes the
# monotonic association between every pair of variables.
spearman_matrix = data3[variables].corr(method="spearman")

print("Pearson correlation matrix:")
print(pearson_matrix)
print()

print("Spearman correlation matrix:")
print(spearman_matrix)
print()


# ==================================================
# 6. Identify the largest Pearson-Spearman gap
# ==================================================

# Find the pair for which Pearson and Spearman
# correlations differ the most.
largest_gap = results_df.loc[
    results_df["Difference"].idxmax()
]

print(
    f"Largest Pearson-Spearman gap: "
    f"{largest_gap['Difference']:.4f}"
)

print(
    f"Pair with the largest gap: "
    f"{largest_gap['Pair']}"
)