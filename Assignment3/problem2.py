""""
Question 2: American Options and Discrete Dividends

Stock price: 48.50
Strike price: 50.00
Maturity: 0.5 years
Risk-free rate: 4.5%
Volatility: 30%

Dividends:
    0.25 at t = 0.10
    1.50 at t = 0.40

The number of tree steps must be a multiple of 5
so that both dividend dates fall on tree steps.
"""

import time

from p2_lib import (
    early_exercise_check,
    early_exercise_steps,
    price_all_four,
    binomial_price,
    nonrecomb_price,
    black_scholes,
    gbsm,
    pv_dividends,
    fd_delta_gamma
)


# --------------------------------------------------
# Set up the option parameters
# --------------------------------------------------

S0 = 48.50
X = 50.00
T = 0.50
r = 0.045
sigma = 0.30

# Dividend dates and amounts
DIVS = [(0.10, 0.25), (0.40, 1.50)]

# Number of steps for the convergence test
N_LIST = [5, 10, 50, 100, 500, 1000]

# Main tree size
N = 500


# A function to calculate option prices using
# the non-recombining tree
def option_price(option, american, n=N, dividends=DIVS, stock=S0):
    return nonrecomb_price(
        stock, X, T, r, sigma, n,
        dividends, option, american
    )


# --------------------------------------------------
# Part (a): Early exercise condition
# --------------------------------------------------

print("\n" + "=" * 65)
print("Part (a): Early Exercise Condition")
print("=" * 65)

# Check the early exercise condition for each dividend
results = early_exercise_check(DIVS, X, r, T)

for item in results:
    print(
        f"Dividend date: {item['t']:.2f}, "
        f"Dividend amount: {item['D']:.2f}, "
        f"Threshold: {item['threshold_to_T']:.4f}"
    )

    if item["possible_vs_T"]:
        print("Early exercise may be optimal.")
    else:
        print("Early exercise is not optimal.")

# Use the binomial tree to check which nodes
# actually exercise the American call
_, tree = binomial_price(
    S0, X, T, r, sigma, N, DIVS,
    "call", True, return_tree=True
)

print(f"\nEarly exercise nodes for the American call (N = {N}):")

for step, t, count in early_exercise_steps(tree):
    print(
        f"Step {step:4d}, "
        f"Time = {t:.4f}, "
        f"Number of nodes = {count}"
    )


# --------------------------------------------------
# Part (b): Effect of dividends on option prices
# --------------------------------------------------

print("\n" + "=" * 65)
print("Part (b): Effect of Dividends")
print("=" * 65)

# Compare European and American option prices
# with and without dividends for different tree sizes

for n in N_LIST:

    no_div = price_all_four(
        S0, X, T, r, sigma, n, ()
    )

    with_div = price_all_four(
        S0, X, T, r, sigma, n, DIVS
    )

    print(f"\nNumber of steps: {n}")

    print(
        f"{'Option':<18}"
        f"{'No Div':>10}"
        f"{'With Div':>12}"
        f"{'Change':>12}"
    )

    for option in [
        "EC", "AC", "EP", "AP",
        "call_premium", "put_premium"
    ]:

        change = with_div[option] - no_div[option]

        print(
            f"{option:<18}"
            f"{no_div[option]:>10.4f}"
            f"{with_div[option]:>12.4f}"
            f"{change:>+12.4f}"
        )


# Compare European option prices with Black-Scholes
print("\nBlack-Scholes European Option Prices")

for option in ["call", "put"]:

    no_div = black_scholes(
        S0, X, T, r, sigma, option
    )

    with_div = black_scholes(
        S0, X, T, r, sigma, option, DIVS
    )

    print(
        f"{option.capitalize()}: "
        f"No dividends = {no_div:.4f}, "
        f"With dividends = {with_div:.4f}"
    )


# --------------------------------------------------
# Part (c): Compare European and American options
# --------------------------------------------------

print("\n" + "=" * 65)
print("Part (c): European and American Option Prices")
print("=" * 65)

# First, calculate option prices without dividends
print("\nWithout dividends:")

for option in ["call", "put"]:

    european = gbsm(
        S0, X, T, r, r, sigma, option
    )

    american = binomial_price(
        S0, X, T, r, sigma, N,
        (), option, True
    )

    print(f"\n{option.capitalize()}:")
    print(f"  European option: {european:.4f}")
    print(f"  American option: {american:.4f}")


# Calculate the present value of the dividends
pv = pv_dividends(DIVS, r)

print("\nWith dividends:")
print(f"Present value of dividends: {pv:.4f}")
print(f"Adjusted stock price: {S0 - pv:.4f}")


# Compare the non-recombining tree with GBSM
for option in ["call", "put"]:

    european = option_price(option, False)
    american = option_price(option, True)

    # Use the adjusted stock price in GBSM
    benchmark = gbsm(
        S0 - pv, X, T, r, r, sigma, option
    )

    print(f"\n{option.capitalize()}:")

    print(f"  European tree price: {european:.4f}")
    print(f"  GBSM price:          {benchmark:.4f}")
    print(f"  Difference:          {european - benchmark:+.4f}")
    print(f"  American tree price: {american:.4f}")

    # The early exercise premium is the difference
    # between American and European option prices
    premium = american - european

    print(f"  Early exercise premium: {premium:.4f}")




# --------------------------------------------------
# Part (d): Convergence of American option prices
# --------------------------------------------------

print("\n" + "=" * 65)
print("Part (d): Convergence of American Option Prices")
print("=" * 65)

print(
    f"{'N':>8}"
    f"{'American Call':>18}"
    f"{'American Put':>18}"
    f"{'Time (sec)':>14}"
)

# Compare different tree sizes, including neighbouring values,
# to observe possible oscillations in the option prices.
N_VALUES = [10, 20, 50, 100, 105, 200, 205, 500, 505, 1000, 1005]

for n in N_VALUES:

    start = time.time()

    call = option_price("call", True, n)
    put = option_price("put", True, n)

    elapsed = time.time() - start

    print(
        f"{n:8d}"
        f"{call:18.4f}"
        f"{put:18.4f}"
        f"{elapsed:14.2f}"
    )



# --------------------------------------------------
# Part (e): Delta and Gamma using finite differences
# --------------------------------------------------

print("\n" + "=" * 65)
print("Part (e): Delta and Gamma")
print("=" * 65)

print(
    f"{'h':>8}"
    f"{'Call Delta':>14}"
    f"{'Call Gamma':>14}"
    f"{'Put Delta':>14}"
    f"{'Put Gamma':>14}"
)

# Calculate Delta and Gamma using different bump sizes
for h in [1, 0.1, 0.01, 0.001]:

    # American call sensitivities
    call_delta, call_gamma = fd_delta_gamma(
        lambda stock: option_price(
            "call", True, stock=stock
        ),
        S0, h
    )

    # American put sensitivities
    put_delta, put_gamma = fd_delta_gamma(
        lambda stock: option_price(
            "put", True, stock=stock
        ),
        S0, h
    )

    print(
        f"{h:8g}"
        f"{call_delta:14.4f}"
        f"{call_gamma:14.4f}"
        f"{put_delta:14.4f}"
        f"{put_gamma:14.4f}"
    )


# --------------------------------------------------
# Part (f): Sensitivity to dividend amounts
# --------------------------------------------------

print("\n" + "=" * 65)
print("Part (f): Dividend Sensitivity")
print("=" * 65)

# Use a central difference with a bump of 0.01
epsilon = 0.01

print(
    f"{'Option':<18}"
    f"{'dV/dD1':>12}"
    f"{'dV/dD2':>12}"
)

for option in ["call", "put"]:

    for american in [False, True]:

        sensitivities = []

        # Calculate sensitivity to each dividend separately
        for i in range(len(DIVS)):

            dividends_up = [list(d) for d in DIVS]
            dividends_down = [list(d) for d in DIVS]

            dividends_up[i][1] += epsilon
            dividends_down[i][1] -= epsilon

            price_up = option_price(
                option,
                american,
                dividends=dividends_up
            )

            price_down = option_price(
                option,
                american,
                dividends=dividends_down
            )

            # Central difference approximation
            sensitivity = (
                price_up - price_down
            ) / (2 * epsilon)

            sensitivities.append(sensitivity)

        if american:
            option_name = "American "
        else:
            option_name = "European "

        option_name += option.capitalize()

        print(
            f"{option_name:<18}"
            f"{sensitivities[0]:12.4f}"
            f"{sensitivities[1]:12.4f}"
        )


# --------------------------------------------------
# Part (g): Early exercise premium and critical price
# --------------------------------------------------

print("\n" + "=" * 65)
print("Part (g): Early Exercise Premium")
print("=" * 65)

# Compare early exercise premiums with and without dividends
for option in ["call", "put"]:

    # Without dividends
    american_no_div = binomial_price(
        S0, X, T, r, sigma, N,
        (), option, True
    )

    european_no_div = binomial_price(
        S0, X, T, r, sigma, N,
        (), option, False
    )

    # With dividends
    american_div = option_price(option, True)
    european_div = option_price(option, False)

    premium_no_div = american_no_div - european_no_div
    premium_div = american_div - european_div

    print(f"\n{option.capitalize()}:")
    print(f"  Without dividends: {premium_no_div:.4f}")
    print(f"  With dividends:    {premium_div:.4f}")


# Find the critical stock price immediately before
# the second dividend at t = 0.40

D2 = 1.50
tau = T - 0.40


# Bisection method to find the critical stock price
def bisection(function, lower, upper):

    for _ in range(100):

        middle = (lower + upper) / 2

        if function(middle) > 0:
            upper = middle
        else:
            lower = middle

    return (lower + upper) / 2


# Critical price using the GBSM continuation value
# Exercise value = S - X
# Continuation value = call price on S - D2

critical_bs = bisection(
    lambda stock:
        (stock - X) - gbsm(
            stock - D2,
            X,
            tau,
            r,
            r,
            sigma,
            "call"
        ),
    X,
    2 * X
)


# Critical price using the binomial tree
# There are no additional dividends after the second dividend

steps = round(N * tau / T)

critical_tree = bisection(
    lambda stock:
        (stock - X) - binomial_price(
            stock - D2,
            X,
            tau,
            r,
            sigma,
            steps,
            (),
            "call",
            True
        ),
    X,
    2 * X
)


# Display the critical stock prices
print("\nCritical Stock Price Before the Second Dividend")
print(f"GBSM critical price: {critical_bs:.4f}")
print(f"Tree critical price: {critical_tree:.4f}")
print(f"Remaining tree steps: {steps}")
