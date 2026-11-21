"""
Question 12: European and American Option Pricing with Greeks
Using functions from risklib.
"""

import pandas as pd
import risk_lib_12 as rl


# ============================================================
# 1. Settings and Input Data
# ============================================================
N_STEPS = 500

options_12_1 = pd.read_csv("test12_1.csv").dropna(subset=["ID"])
options_12_3 = pd.read_csv("test12_3.csv").dropna(subset=["ID"])


# ============================================================
# 2. European Options (GBSM) with Greeks - Question 12.1
# ============================================================
results_12_1 = []

for _, option in options_12_1.iterrows():

    is_call = option["Option Type"] == "Call"
    maturity = option["DaysToMaturity"] / option["DayPerYear"]

    risk_free_rate = option["RiskFreeRate"]
    cost_of_carry = risk_free_rate - option["DividendRate"]

    underlying = option["Underlying"]
    strike = option["Strike"]
    volatility = option["ImpliedVol"]

    # Calculate option value and Greeks
    option_value = rl.gbsm(
        is_call, underlying, strike, maturity,
        risk_free_rate, cost_of_carry, volatility
    )

    greeks = rl.gbsm_greeks(
        is_call, underlying, strike, maturity,
        risk_free_rate, cost_of_carry, volatility
    )

    results_12_1.append({
        "ID": int(option["ID"]),
        "Value": option_value,
        **greeks
    })

result_12_1 = pd.DataFrame(results_12_1)


# ============================================================
# 3. American Options with Continuous Dividends - Question 12.2
# ============================================================
results_12_2 = []

for _, option in options_12_1.iterrows():

    is_call = option["Option Type"] == "Call"
    maturity = option["DaysToMaturity"] / option["DayPerYear"]

    risk_free_rate = option["RiskFreeRate"]
    cost_of_carry = risk_free_rate - option["DividendRate"]

    greeks = rl.american_binomial_greeks(
        is_call,
        option["Underlying"],
        option["Strike"],
        maturity,
        risk_free_rate,
        cost_of_carry,
        option["ImpliedVol"],
        N_STEPS
    )

    results_12_2.append({
        "ID": int(option["ID"]),
        **greeks
    })

result_12_2 = pd.DataFrame(results_12_2)


# ============================================================
# 4. American Options with Discrete Dividends - Question 12.3
# ============================================================
results_12_3 = []

for _, option in options_12_3.iterrows():

    is_call = option["Option Type"] == "Call"

    days_to_maturity = option["DaysToMaturity"]
    maturity = days_to_maturity / option["DayPerYear"]

    # Convert dividend dates and amounts into lists
    dividend_days = [
        float(x) for x in str(option["DividendDates"]).split(",")
    ]

    dividend_amounts = [
        float(x) for x in str(option["DividendAmts"]).split(",")
    ]

    # Convert dividend dates into binomial tree steps
    dividend_steps = [
        int(round(day / days_to_maturity * N_STEPS))
        for day in dividend_days
    ]

    # Calculate American option value
    option_value = rl.american_binomial_discrete_div(
        is_call,
        option["Underlying"],
        option["Strike"],
        maturity,
        option["RiskFreeRate"],
        dividend_amounts,
        dividend_steps,
        option["ImpliedVol"],
        N_STEPS
    )

    results_12_3.append({
        "ID": int(option["ID"]),
        "Value": option_value
    })

result_12_3 = pd.DataFrame(results_12_3)


# ============================================================
# 5. Print Results
# ============================================================
results = {
    "12.1 European Options (GBSM) with Greeks": result_12_1,
    "12.2 American Options with Continuous Dividends": result_12_2,
    "12.3 American Options with Discrete Dividends": result_12_3
}

for title, result in results.items():

    print(f"\n{title}")
    print("=" * 70)

    print(
        result.to_string(
            index=False,
            float_format=lambda x: repr(float(x))
        )
    )
