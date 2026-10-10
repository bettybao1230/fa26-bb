# Assignment 2 --- Covariance, VaR, and Copulas

FinTech 545 --- Quantitative Risk Management

This folder contains the code that reproduces every number in the written responses (`responses.pdf`). Each problem has its own script, and all shared functions live in one library, `risklib.py`.

## Folder contents

```         
Assignment2/
├── README.md        # this file
├── risklib.py       # shared library functions (correlation repair, VaR/ES, copulas, market model)
├── problem1.py      # Correlations from mismatched histories
├── problem2.py      # Volatility estimate after a regime change
├── problem3.py      # When diversification raises VaR
├── problem4.py      # Gaussian vs. t copula
├── problem5.py      # Model-based simulation and residual correlation
├── problem1.csv     # input data for problem 1
├── problem2.csv     # input data for problem 2
├── problem3.csv     # input data for problem 3
├── problem4.csv     # input data for problem 4
├── problem5.csv     # input data for problem 5
└── responses.pdf    # written responses to all five problems
```

## Requirements

-   Python 3.9 or newer
-   Packages: `numpy`, `pandas`, `scipy`, `matplotlib`

Install with:

``` bash
pip install numpy pandas scipy matplotlib
```

## How to run

Run each script from inside the `Assignment2` folder. The scripts read their CSV files with relative paths (e.g. `pd.read_csv("problem1.csv")`) and import `risklib` from the same directory, so the working directory matters.

``` bash
cd Assignment2

python problem1.py
python problem2.py
python problem3.py
python problem4.py
python problem5.py
```

Each script prints its results to the terminal, labeled by the lettered part of the problem it answers (for example "Part (c)" or "PART (b)").

### Plots

`problem2.py`, `problem3.py`, and `problem4.py` also open matplotlib windows. The script pauses until the window is closed. To run without a display, set a non-interactive backend first:

``` bash
MPLBACKEND=Agg python problem2.py
```

To save the output of a script to a file:

``` bash
python problem1.py > problem1_output.txt
```

## Problem-by-problem guide

| Script        | Parts it produces       | What it prints                                                                                                                                                                                                                                                                  |
|------------------------|------------------------|------------------------|
| `problem1.py` | 1, d, e, f              | Pairwise observation counts, complete-case and pairwise correlation matrices, eigenvalues, Cholesky success/failure, tracking portfolio variance, Rebonato--Jackel and Higham repairs (smallest eigenvalue, Frobenius distance, repaired variance)                              |
| `problem2.py` | b (returns), c, d, f, g | Demeaned-return plot, first four moments, the five one-day VaRs on \$1M, fitted Student-t parameters, standard deviation inside each regime, EWMA volatilities                                                                                                                  |
| `problem3.py` | a, b, c, d, e           | P&L histograms, four moments per bond, large-loss counts, historical VaR/ES at 5%, normal VaR at 5%, historical VaR/ES at 1%                                                                                                                                                    |
| `problem4.py` | a--g                    | Moments with and without the most extreme day, rank scatter plots and joint-tail counts, marginal AICc comparison, copula log-likelihood/AICc/BIC and fitted ν, VaR/ES by method at 5% and 1%, joint-tail counts per 1,000 days, central-day share of the likelihood difference |
| `problem5.py` | b, c, d                 | Regression alpha/beta/residual SD and residual correlation, simulated 95% VaR for P1 and P2 under full and zero residual covariance, delta-normal 95% VaR                                                                                                                       |

Parts that are explanations only (the Predict and Reconcile sections) are answered in `responses.pdf`, using the numbers printed by these scripts. The tail dependence coefficients in problem 4 part (j) follow from the fitted correlation matrix and ν printed by `problem4.py`.

## Conventions and choices

-   **Returns.** Arithmetic returns, `P_t / P_{t-1} - 1`, as the assignment specifies. 501 prices give 500 returns.
-   **VaR and ES.** Reported as positive dollar losses. VaR and ES are at α = 5% unless stated; problem 5 uses 95% confidence, which is the same tail.
-   **Standard deviations.** Sample standard deviation (`ddof=1`) everywhere, including the full-history standard deviations used to build the covariance matrix in problem 1.
-   **Missing data (problem 1).** The `Day` column is dropped. Complete-case correlation uses only rows where all five series are observed; pairwise correlation uses every row available to each pair.
-   **Demeaning (problem 2).** Returns are demeaned before fitting each VaR model, so every estimate is for a zero-mean return.
-   **Copula estimation (problem 4).** Margins are fitted by maximum likelihood (Normal or Student-t, chosen by AICc). The copula correlation matrix comes from Kendall's τ, and ν for the t copula is found by maximizing the copula log-likelihood over the range 2.01 to 200.
-   **Random seeds.** Simulations are seeded, so results are reproducible:
    -   problem 4: Gaussian copula seed 123, t copula seed 456
    -   problem 5: seed 585 for both scenarios (the same seed is used so the full and zero-covariance runs differ only in the residual covariance)
-   **Simulation size.** 100,000 draws in problems 4 and 5. Joint-tail counts in problem 4 part (f) are scaled to 1,000 days to match the historical sample.

## Reproducibility notes

-   Exact digits may differ slightly across versions of NumPy and SciPy, mainly in the Student-t and copula optimizers. The conclusions do not depend on these differences.
-   If `import risklib` fails, confirm you are running from inside the `Assignment2` folder and that `risklib.py` is in it.
