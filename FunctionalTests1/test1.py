# --------------------------------------------------------
# Test 1
# --------------------------------------------------------

import numpy as np
import pandas as pd

# --------------------------------------------------------
# Test 1.1: Covariance - Missing data, skip missing rows
# --------------------------------------------------------

# Read input
data = pd.read_csv("test1.csv")

# Save column names before converting to NumPy
labels = data.columns

# Remove rows containing any missing values
no_missing_value_data = data[~data.isna().any(axis=1)]

# Sample covariance matrix
cov_no_missing_value = np.cov(
    no_missing_value_data,
    rowvar=False
)

# Convert covariance matrix back to DataFrame
cov_1_1 = pd.DataFrame(
    cov_no_missing_value,
    columns=labels
)

print(cov_1_1)
# cov_1_1.to_csv("cov_1_1.csv", index=False)


# --------------------------------------------------------
# Test 1.2: Correlation - Missing data, skip missing rows
# --------------------------------------------------------

corr_no_missing_value = no_missing_value_data.corr()

# Convert to table
corr_1_2 = pd.DataFrame(
    corr_no_missing_value,
    columns=labels
)

print(corr_1_2)
# corr_1_2.to_csv("corr_1_2.csv", index=False)


# --------------------------------------------------------
# Test 1.3: Covariance - Missing data, Pairwise
# --------------------------------------------------------

# Calculate covariance for each pair of columns,
# using all available observations for that pair
cov_data = data.cov()

# Convert to table
cov_1_3 = pd.DataFrame(
    cov_data,
    columns=labels
)

print(cov_1_3)
# cov_1_3.to_csv("cov_1_3.csv", index=False)


# --------------------------------------------------------
# Test 1.4: Correlation - Missing data, Pairwise
# --------------------------------------------------------

# pandas .corr() automatically uses pairwise complete observations
corr_data = data.corr()

# Convert to table
corr_1_4 = pd.DataFrame(
    corr_data,
    columns=labels
)

print(corr_1_4)
# corr_1_4.to_csv("corr_1_4.csv", index=False)