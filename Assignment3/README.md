# Assignment 3 — Options, Option Risk, and Portfolio Construction

FinTech 545 — Quantitative Risk Management

The written answers are in **`assignment3_responses.pdf`**. Each problem has one
script, `problemN.py`, which prints every number reported for that problem and draws
its figure. The option pricing (generalized Black–Scholes–Merton, its Greeks, the
binomial trees) and the risk and regression helpers are written in the
matching `pN_lib.py` file.

---

## 1. Setup

Python 3 with NumPy, pandas, SciPy and Matplotlib.

```bash
cd Assignment3

# Install the packages
pip install numpy pandas scipy matplotlib
```

Every script reads its data from the current folder, so **run them from inside
`Assignment3/`**.

---

## 2. Reproduce every number in the write-up

```bash
python problem1.py   # Problem 1: put-call parity, implied vols, vega, smile plot
python problem2.py   # Problem 2: GBSM, binomial trees, dividends, Greeks
python problem3.py   # Problem 3: option-book Greeks, delta-normal and Monte Carlo VaR/ES
python problem4.py   # Problem 4: moments, max-Sharpe portfolios, VaR/ES, CAPM, bootstrap
python problem5.py   # Problem 5: CAPM, Fama-French 3, Carhart 4, stress test
```

Each script prints its results to the terminal in part order, (a), (b), (c), ...,
matching the sections of the PDF. Scripts that make a figure open it in a window at
the end; close the window to let the script finish.

| Script | Parts | Output | Figure in the PDF |
|---|---|---|---|
| `problem1.py` | 1(a)–(h) | parity table, bid/mid/ask implied vols, vega table | `problem1d_iv_smile.png` (Figure 1) |
| `problem2.py` | 2(a)–(i) | prices, convergence table, finite-difference Greeks, dividend sensitivities, critical price | — |
| `problem3.py` | 3(a)–(j) | Greeks, delta-normal and simulated VaR/ES, Taylor comparison | `problem3_pnl_vs_price.png` (Figure 2) |
| `problem4.py` | 4(a)–(h) | moments, portfolio weights, VaR/ES, CAPM, bootstrap | `problem4_frontier.png` (Figure 3) |
| `problem5.py` | 5(a)–(h) | regressions with standard errors and AICc, expected returns, stress loss | — |

**Run time.** Problems 1, 4 and 5 take a few seconds. Problem 3 runs 100,000
Monte Carlo paths and takes a little longer. Problem 2 is the slowest: the
non-recombining tree at N = 1000 and 1005 in part (d) takes about 25 seconds
each.

**Random numbers.** Problem 3 uses a fixed seed (`SEED = 42`), so the Monte Carlo
results are the same on every run. The Problem 4 bootstrap (1,000 resamples) uses a
seeded generator too.

---

## 3. Files

| File | Contents |
|---|---|
| `README.md` | This file: how to run the code and the conventions used |
| `assignment3_responses.pdf` | Written answers |
| `Assignment 3.pdf` | Assignment handout |
| `problem1.py` … `problem5.py` | One driver script per problem |
| `p1_lib.py` … `p5_lib.py` | Functions used by each driver (pricing, Greeks, trees, fitting, VaR/ES, regressions) |
| `problem1.csv` | XYZ option chain: bid and ask for calls and puts at 18 strikes |
| `problem3.csv` | 501 daily ABC prices |
| `problem3_portfolios.csv` | Four books of ABC stock and option positions |
| `problem4.csv` | 240 monthly returns for EQ, BOND, STK and PUTW |
| `problem5.csv` | Five years of daily returns for FUND1 and FUND2 |
| `F-F_Research_Data_Factors_daily.CSV` | Fama–French market, SMB, HML and risk-free rate (percent) |
| `F-F_Momentum_Factor_daily.CSV` | Momentum factor (percent) |
| `*.png` | Figures used in the PDF |

---

## 4. Conventions chosen

Where the assignment leaves a choice, these are the conventions used. They are also
commented in the code where they apply.

**General**
- Arithmetic returns, r_t = P_t / P_{t-1} − 1, unless a problem says otherwise.
- VaR and ES are reported as positive numbers for a loss.

**Problem 1**
- T = calendar days / 365 = 35 / 365.
- Quotes are valued at their mid price. Implied vol is found by root-finding on the
  GBSM price, with cost of carry b = r − q.
- A quote below the no-arbitrage lower bound has no implied vol and is reported as N/A.
- A strike is flagged when |parity gap| is larger than the call spread plus the put spread.

**Problem 2**
- CRR tree: u = e^(σ√Δt), d = 1/u. N is a multiple of 5 so that tree steps fall
  exactly on t = 0.10 and t = 0.40.
- Dividends are discrete cash amounts. The tree is non-recombining: the stock drops
  by D at each dividend date. The American holder compares exercising at the
  cum-dividend price with holding at the ex-dividend price.
- The early-exercise premium is the American tree value minus the European tree
  value at the same N, so the tree error cancels.
- Greeks use central finite differences with bumps of 1, 0.1, 0.01 and 0.001.
  Dividend sensitivities use a central bump of 0.01.

**Problem 3**
- 255 trading days a year. `DaysToExpiry` is in trading days.
- Each option keeps its own implied vol over the horizon.
- The 10-day simulation draws 10 zero-mean daily returns, compounds them, and fully
  revalues every position with 10 fewer days to expiry.
- One historical daily volatility (arithmetic returns, sample std, ddof = 1) is used
  for the predictions in (a)–(b) and the delta-normal VaR/ES in (d).
- AICc = 2k − 2 ln L + 2k(k+1)/(n−k−1), with k = 2 for the normal and k = 3 for the
  Student t (location, scale, degrees of freedom).

**Problem 4**
- Monthly risk-free rate = (1.03)^(1/12) − 1. Sharpe ratios are annualized by √12.
- Unconstrained tangency weights use the closed form. The long-only and capped
  portfolios are solved numerically.
- Historical ES at 1% averages the worst 2.4 months, giving the third-worst month a
  weight of 0.4.
- The bootstrap resamples whole months, keeping each month's four returns together.

**Problem 5**
- Fama–French values are divided by 100. Fund and factor dates are matched on
  common trading days.
- Means are annualized by 252 and volatilities by √252.
- OLS with an intercept and classical standard errors.
- AICc counts the regression coefficients plus the error variance.
- The expected returns in (d) use the factor averages over the last 10 years in the
  factor files.
