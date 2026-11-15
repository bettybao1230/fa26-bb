"""
Assignment 3 - Problem 1(a)

This script:
1. Sets the market information.
2. Loads the option-chain data.
3. Checks for crossed bid-ask quotes.
4. Tests put-call parity.
5. Calculates call and put butterfly costs.
6. Identifies flagged strikes.
"""

import pandas as pd
from datetime import date

from p1_lib import check_put_call_parity, butterfly_cost


# ===========================================================================
# 1. MARKET INFORMATION
# ===========================================================================

S = 182.40                  # Current stock price
r = 0.0425                  # Risk-free rate
q = 0.0060                  # Continuous dividend yield
b = r - q                  # Cost of carry

today = date(2026, 11, 13)
expiry = date(2026, 12, 18)

days = (expiry - today).days
T = days / 365.0            # Time to maturity in years


# ===========================================================================
# 2. LOAD MARKET DATA
# ===========================================================================

df = pd.read_csv("problem1.csv")

# Check whether any bid price is higher than its ask price.
crossed_quotes = df[df["Bid"] > df["Ask"]]

if not crossed_quotes.empty:
    print("Warning: Crossed quotes detected.")
    print(crossed_quotes.to_string(index=False))
else:
    print("No crossed quotes detected.")


# ===========================================================================
# 3. CHECK PUT-CALL PARITY
# ===========================================================================

# Match calls and puts by strike and compare market prices with
# the theoretical put-call parity relationship.
results = check_put_call_parity(df, S, r, q, T)


# ===========================================================================
# 4. CALCULATE BUTTERFLY COSTS
# ===========================================================================

# The results are already sorted by strike.
# Butterfly cost = V(K-h) - 2V(K) + V(K+h)

results["Call_Fly"] = butterfly_cost(
    results["Mid_C"].to_numpy()
)

results["Put_Fly"] = butterfly_cost(
    results["Mid_P"].to_numpy()
)


# ===========================================================================
# 5. DISPLAY PUT-CALL PARITY RESULTS
# ===========================================================================

print("\nMarket Information")
print(f"Spot price:       {S:.2f}")
print(f"Risk-free rate:   {r:.4%}")
print(f"Dividend yield:   {q:.4%}")
print(f"Cost of carry:    {b:.4%}")
print(f"Days to expiry:   {days}")
print(f"Time to maturity: {T:.6f} years")

report_columns = [
    "Mid_C",
    "Mid_P",
    "Market",
    "Theoretical",
    "Gap",
    "Spread_C",
    "Spread_P",
    "Band",
    "Flag"
]

print("\nPut-Call Parity Results")
print(results[report_columns].round(4).to_string())


# ===========================================================================
# 6. IDENTIFY FLAGGED STRIKES
# ===========================================================================

flagged = results[results["Flag"]]

print("\nFlagged Strikes")

if flagged.empty:
    print("No strikes violate the bid-ask spread criterion.")

else:
    print(flagged[["Gap", "Band"]].round(4).to_string())


# ===========================================================================
# 7. DISPLAY BUTTERFLY COSTS AT ALL STRIKES
# ===========================================================================

print("\nButterfly Costs by Strike")

butterfly_report = results[["Call_Fly", "Put_Fly"]]

print(butterfly_report.round(4).to_string())




"""
Assignment 3 - Problem 1(d) and 1(e)

Part (d):
1. Calculate implied volatility using bid, mid, and ask prices.
2. Identify quotes with no implied volatility.
3. Plot call and put implied volatility against strike.

Part (e):
1. Calculate vega at each quote's mid implied volatility.
2. Calculate the IV change caused by a one-cent price change.

The pricing functions are imported from p1_lib.py.
"""

import sys
from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from p1_lib import implied_vol, gbsm_vega, price_bounds, gbsm_price


# ===========================================================================
# 1. MARKET INFORMATION
# ===========================================================================

S = 182.40
r = 0.0425
q = 0.0060
b = r - q

today = date(2026, 11, 13)
expiry = date(2026, 12, 18)

days = (expiry - today).days
T = days / 365.0

ONE_CENT = 0.01


# ===========================================================================
# 2. LOAD OPTION-CHAIN DATA
# ===========================================================================

# Use problem1.csv in the same folder by default.
# An alternative CSV path can be provided when running the script.

script_folder = Path(__file__).resolve().parent

if len(sys.argv) > 1:
    csv_path = Path(sys.argv[1])
else:
    csv_path = script_folder / "problem1.csv"

chain = pd.read_csv(csv_path)

# Standardize option types and calculate mid prices.
chain["Type"] = chain["Type"].str.strip().str.capitalize()
chain["Mid"] = (chain["Bid"] + chain["Ask"]) / 2.0

# Check for crossed quotes.
crossed_quotes = chain[chain["Bid"] > chain["Ask"]]

if not crossed_quotes.empty:
    print("Warning: Crossed quotes detected.")
    print(crossed_quotes.to_string(index=False))


# ===========================================================================
# 3. PART (d): CALCULATE IMPLIED VOLATILITY
# ===========================================================================

iv_rows = []

for _, quote in chain.iterrows():

    option_type = quote["Type"]
    is_call = option_type == "Call"
    K = float(quote["Strike"])

    # Calculate theoretical no-arbitrage price bounds.
    lower_bound, upper_bound = price_bounds(
        is_call, S, K, T, r, b
    )

    # Store the basic information for this option.
    row = {
        "Type": option_type,
        "Strike": K,
        "Bid": float(quote["Bid"]),
        "Mid": float(quote["Mid"]),
        "Ask": float(quote["Ask"]),
        "LowerBound": lower_bound,
        "UpperBound": upper_bound
    }

    # Calculate IV separately for bid, mid, and ask.
    no_iv_reasons = []

    for price_type in ["Bid", "Mid", "Ask"]:

        market_price = float(quote[price_type])

        sigma, reason = implied_vol(
            is_call, S, K, T, r, b, market_price
        )

        row["IV_" + price_type] = sigma

        # Record the reason if an implied volatility cannot be found.
        if sigma is None:
            no_iv_reasons.append(price_type + ": " + reason)

    row["NoIV_Reason"] = "; ".join(no_iv_reasons)

    iv_rows.append(row)


iv = pd.DataFrame(iv_rows)


# ===========================================================================
# 4. CLASSIFY OPTIONS AS ITM, ATM, OR OTM
# ===========================================================================

moneyness = []

for _, row in iv.iterrows():

    K = row["Strike"]
    option_type = row["Type"]

    if abs(K - S) < 1e-8:
        label = "ATM"

    elif option_type == "Call":
        label = "ITM" if K < S else "OTM"

    else:
        label = "ITM" if K > S else "OTM"

    moneyness.append(label)

iv["Moneyness"] = moneyness


# ===========================================================================
# 5. PART (e): CALCULATE VEGA AND IV PRICE SENSITIVITY
# ===========================================================================

vega_values = []
vega_per_vol_point_values = []
iv_change_values = []

for _, row in iv.iterrows():

    sigma = row["IV_Mid"]

    # If mid IV does not exist, vega cannot be calculated.
    if pd.isna(sigma):

        vega = float("nan")
        vega_per_vol_point = float("nan")
        iv_change = float("nan")

    else:

        # Calculate vega using the mid implied volatility.
        vega = gbsm_vega(
            S, row["Strike"], T, r, b, sigma
        )

        # Raw vega is the price change for a 1.00 change in volatility.
        # Divide by 100 to get the price change for one volatility point.
        vega_per_vol_point = vega / 100.0

        # Approximate IV change for a one-cent option-price change.
        # Convert the volatility change into percentage points.
        iv_change = 100.0 * ONE_CENT / vega

    vega_values.append(vega)
    vega_per_vol_point_values.append(vega_per_vol_point)
    iv_change_values.append(iv_change)


iv["Vega"] = vega_values
iv["Vega_per_volpt"] = vega_per_vol_point_values
iv["dIV_per_cent_volpts"] = iv_change_values


# ===========================================================================
# 6. CREATE A VEGA TABLE WITH ONE OPTION PER STRIKE
# ===========================================================================

# Use the OTM option at each strike.
# The 175 put is excluded because it was identified as a parity outlier
# in part (a). Therefore, use the 175 call instead.

strike_rows = []

for K in sorted(iv["Strike"].unique()):

    same_strike = iv[iv["Strike"] == K]

    # Find the OTM option at this strike.
    otm_options = same_strike[
        same_strike["Moneyness"] == "OTM"
    ]

    if abs(K - 175.0) < 1e-8:

        # Use the call instead of the parity-outlier put.
        selected = same_strike[
            same_strike["Type"] == "Call"
        ]

    elif not otm_options.empty:

        selected = otm_options

    else:

        selected = same_strike

    if selected.empty:
        continue

    option = selected.iloc[0]

    strike_rows.append({
        "Strike": K,
        "Option_used": option["Type"],
        "IV_mid": option["IV_Mid"],
        "Vega": option["Vega"],
        "Vega_per_volpt": option["Vega_per_volpt"],
        "dIV_per_cent_volpts": option["dIV_per_cent_volpts"]
    })


strike_view = pd.DataFrame(strike_rows).set_index("Strike")


# ===========================================================================
# 7. DISPLAY PART (d): IMPLIED VOLATILITY RESULTS
# ===========================================================================

print("\nPart (d): Implied Volatility (%)")

# Convert IV from decimal form to percentage form for display.
iv_display = iv.copy()

for column in ["IV_Bid", "IV_Mid", "IV_Ask"]:
    iv_display[column] = iv_display[column] * 100.0

display_columns = [
    "Strike",
    "Moneyness",
    "Bid",
    "Mid",
    "Ask",
    "IV_Bid",
    "IV_Mid",
    "IV_Ask"
]

# Display calls and puts separately.
for option_type in ["Call", "Put"]:

    print(f"\n{option_type}s")

    option_rows = iv_display[
        iv_display["Type"] == option_type
    ]

    print(
        option_rows[display_columns]
        .round(4)
        .to_string(index=False)
    )


# ===========================================================================
# 8. REPORT QUOTES WITH NO IMPLIED VOLATILITY
# ===========================================================================

print("\nQuotes with No Implied Volatility")

missing = iv[iv["NoIV_Reason"] != ""]

if missing.empty:

    print("All quotes have implied volatilities.")

else:

    for _, row in missing.iterrows():

        print(
            f"{row['Type']} K={row['Strike']:.0f}: "
            f"{row['NoIV_Reason']}"
        )


# ===========================================================================
# 9. DISPLAY PART (e): VEGA RESULTS
# ===========================================================================

print("\nPart (e): Vega and IV Change for a One-Cent Price Change")

vega_display = iv[
    [
        "Type",
        "Strike",
        "IV_Mid",
        "Vega",
        "Vega_per_volpt",
        "dIV_per_cent_volpts"
    ]
].copy()

# Display IV as a percentage.
vega_display["IV_Mid"] = vega_display["IV_Mid"] * 100.0

# Display calls and puts separately.
for option_type in ["Call", "Put"]:

    print(f"\n{option_type}s")

    option_rows = vega_display[
        vega_display["Type"] == option_type
    ]

    print(
        option_rows.round(4).to_string(index=False)
    )


# One option per strike.
print("\nPart (e): Vega Summary by Strike")
print("Using OTM options, except K=175 where the Call is used.")

strike_display = strike_view.copy()
strike_display["IV_mid"] = strike_display["IV_mid"] * 100.0

print(
    strike_display.reset_index().round(4).to_string(index=False)
)

# ===========================================================================
# 10. SAVE OUTPUT TABLES
# ===========================================================================

iv.to_csv(
    script_folder / "problem1d_implied_vols.csv",
    index=False
)

strike_view.to_csv(
    script_folder / "problem1e_vega.csv"
)


# ===========================================================================
# 11. PLOT THE IMPLIED-VOLATILITY SMILE
# ===========================================================================

fig, ax = plt.subplots(figsize=(9, 5.5))
styles = {"Call": dict(marker="o", linestyle="-", color="black"),
          "Put": dict(marker="s", linestyle="--", color="gray")}
for t, st in styles.items():
    sub = iv[iv["Type"] == t].sort_values("Strike")
    ax.plot(sub["Strike"], sub["IV_Mid"] * 100, label=f"{t} (mid)", **st)
    # Vertical bar from bid IV to ask IV where both exist; offset slightly so
    # call and put bars at the same strike don't overlap.
    both = sub.dropna(subset=["IV_Bid", "IV_Ask"])
    ax.vlines(both["Strike"] + (-0.4 if t == "Call" else 0.4),
              both["IV_Bid"] * 100, both["IV_Ask"] * 100,
              color=st["color"], alpha=0.6, linewidth=1)
ax.axvline(S, color="black", linewidth=0.8, linestyle=":")
ax.text(S + 0.5, ax.get_ylim()[1] * 0.97, f"spot {S}", fontsize=8, va="top")
ax.set_xlabel("Strike")
ax.set_ylabel("Implied volatility (%)")
ax.set_title("XYZ implied volatility by strike, expiry 12/18/2026\n"
             "markers = mid; vertical bars = bid-to-ask IV range")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
plt.show()


# ===========================================================================
# 12. PART (f): THE FLAGGED STRIKE, IN VOL POINTS
# ===========================================================================

# A parity gap in dollars becomes a call-put IV gap of about
#     sigma_C - sigma_P ~= Gap / Vega
# because the call and the put at one strike share the same vega.

print("\nPart (f): Parity Gap Translated into Vol Points")


def quote_at(option_type, K):
    """Return the IV-table row for one option (type + strike)."""
    return iv[(iv["Type"] == option_type) & (iv["Strike"] == K)].iloc[0]


for K in flagged.index:

    call = quote_at("Call", K)
    put = quote_at("Put", K)
    gap = results.loc[K, "Gap"]

    # Vega is evaluated at each option's own mid IV, so the call and put
    # vegas differ slightly; the true answer lies between the two estimates.
    vega_call = call["Vega"]
    vega_put = put["Vega"]
    vega_avg = 0.5 * (vega_call + vega_put)

    est_call = 100.0 * gap / vega_call      # vol points, using call vega
    est_put = 100.0 * gap / vega_put        # vol points, using put vega
    est_avg = 100.0 * gap / vega_avg        # vol points, using average vega

    actual = 100.0 * (call["IV_Mid"] - put["IV_Mid"])   # actual IV gap

    print(f"\nStrike {K:.0f}")
    print(f"Parity gap:                 {gap:.4f}")
    print(f"Call vega / put vega:       {vega_call:.4f} / {vega_put:.4f}")
    print(f"Gap / call vega:            {est_call:.2f} vol points")
    print(f"Gap / put vega:             {est_put:.2f} vol points")
    print(f"Gap / average vega:         {est_avg:.2f} vol points")
    print(f"Actual call IV - put IV:    {actual:.2f} vol points "
          f"({100 * call['IV_Mid']:.4f}% - {100 * put['IV_Mid']:.4f}%)")

    # Do the bid-to-ask IV ranges of the call and put overlap? If not, the
    # gap cannot be explained by the spread.
    call_lo, call_hi = 100 * call["IV_Bid"], 100 * call["IV_Ask"]
    put_lo, put_hi = 100 * put["IV_Bid"], 100 * put["IV_Ask"]
    overlap = not (put_hi < call_lo or call_hi < put_lo)
    print(f"Call IV range (bid-ask):    {call_lo:.2f}% to {call_hi:.2f}%")
    print(f"Put IV range (bid-ask):     {put_lo:.2f}% to {put_hi:.2f}%")
    print(f"Ranges overlap:             {overlap}")

    # IV at the neighbouring strikes, to show the dip in the plot.
    neighbours = iv[iv["Strike"].isin([K - 5, K, K + 5])]
    neighbour_table = neighbours.pivot(
        index="Strike", columns="Type", values="IV_Mid"
    ) * 100.0
    print("\nMid IV (%) at the flagged strike and its neighbours")
    print(neighbour_table.round(4).to_string())

# The same check at every strike: predicted IV gap (Gap / average vega)
# against the actual call-put IV gap.
check_rows = []
for K in results.index:
    call = quote_at("Call", K)
    put = quote_at("Put", K)
    vega_avg = 0.5 * (call["Vega"] + put["Vega"])
    check_rows.append({
        "Strike": K,
        "Gap": results.loc[K, "Gap"],
        "Predicted_IV_gap": 100.0 * results.loc[K, "Gap"] / vega_avg,
        "Actual_IV_gap": 100.0 * (call["IV_Mid"] - put["IV_Mid"]),
        "Flag": results.loc[K, "Flag"]
    })

print("\nGap / Vega against the actual call-put IV gap at every strike "
      "(vol points)")
print(pd.DataFrame(check_rows).round(4).to_string(index=False))


# ===========================================================================
# 13. PART (g): THE CALL WING ABOVE 210
# ===========================================================================

WING_START = 210.0

print("\nPart (g): Call Wing Above 210")

wing_rows = []
for K in sorted(iv["Strike"].unique()):

    if K < WING_START:
        continue

    call = quote_at("Call", K)
    put = quote_at("Put", K)

    # Value of the call if the smile simply continued at the put's mid IV
    # at the same strike (by parity, call and put should share one IV).
    fair_call = gbsm_price(True, S, K, T, r, b, put["IV_Mid"])

    # Parity gap converted to vol points with the call's vega.
    gap = results.loc[K, "Gap"]

    wing_rows.append({
        "Strike": K,
        "Bid": call["Bid"],
        "Ask": call["Ask"],
        "IV_Bid": 100 * call["IV_Bid"],
        "IV_Mid": 100 * call["IV_Mid"],
        "IV_Ask": 100 * call["IV_Ask"],
        "IV_range_volpts": 100 * (call["IV_Ask"] - call["IV_Bid"]),
        "dIV_per_cent": call["dIV_per_cent_volpts"],
        "Put_IV_Mid": 100 * put["IV_Mid"],
        "Call_at_put_IV": fair_call,
        "Below_bid": fair_call < call["Bid"],
        "Gap": gap,
        "Band": results.loc[K, "Band"],
        "Gap_volpts": 100.0 * gap / call["Vega"]
    })

wing = pd.DataFrame(wing_rows)

print("\nCall quotes, IVs and sensitivity")
print(wing[["Strike", "Bid", "Ask", "IV_Bid", "IV_Mid", "IV_Ask",
            "IV_range_volpts", "dIV_per_cent"]].round(4).to_string(index=False))

print("\nCall value at the put's IV, compared with the call bid")
print(wing[["Strike", "Put_IV_Mid", "Call_at_put_IV", "Bid",
            "Below_bid"]].round(4).to_string(index=False))

print("\nWhy parity missed it: gap in dollars vs band, and gap in vol points")
print(wing[["Strike", "Gap", "Band", "Gap_volpts"]]
      .round(4).to_string(index=False))

# Identical quotes across strikes: a sign of a price floor.
floor = wing[wing["Strike"] > WING_START]
same_quote = floor[["Bid", "Ask"]].nunique().max() == 1
print(f"\nCalls above {WING_START:.0f} all share one bid/ask quote: {same_quote}")


# ===========================================================================
# 14. PART (h): SHAPE OF THE CLEAN SMILE
# ===========================================================================

# Ignore the two problems: the 175 put (already replaced by the call in
# strike_view) and the floor-priced calls above 210.
clean = strike_view[strike_view.index <= WING_START].copy()
clean["IV_mid_pct"] = clean["IV_mid"] * 100.0

print("\nPart (h): Smile Without the Two Problems "
      "(OTM option per strike; call at 175; strikes up to 210)")
print(clean[["Option_used", "IV_mid_pct"]].round(4).to_string())

# Puts above 210 show where the smile goes once the floor calls are removed.
upper_puts = iv[(iv["Type"] == "Put") & (iv["Strike"] > WING_START)]
print("\nPut mid IV (%) above 210")
print((upper_puts.set_index("Strike")["IV_Mid"] * 100).round(4).to_string())

# Slope of the skew below and above spot, in vol points per dollar of strike.
iv_140 = clean.loc[140.0, "IV_mid_pct"]
iv_180 = clean.loc[180.0, "IV_mid_pct"]
iv_185 = clean.loc[185.0, "IV_mid_pct"]
iv_210 = clean.loc[210.0, "IV_mid_pct"]

downside_slope = (iv_140 - iv_180) / (180.0 - 140.0)
upside_slope = (iv_185 - iv_210) / (210.0 - 185.0)

print(f"\nDownside: IV rises {iv_140 - iv_180:.2f} points from 180 to 140 "
      f"({downside_slope:.3f} points per dollar)")
print(f"Upside:   IV falls {iv_185 - iv_210:.2f} points from 185 to 210 "
      f"({upside_slope:.3f} points per dollar)")

# How expensive downside protection is: price each OTM put at the
# at-the-money IV (strike nearest spot) and compare with its market mid.
atm_strike = min(clean.index, key=lambda k: abs(k - S))
atm_iv = clean.loc[atm_strike, "IV_mid"]

protection_rows = []
for K in [140.0, 150.0, 160.0, 170.0]:
    put = quote_at("Put", K)
    flat_price = gbsm_price(False, S, K, T, r, b, atm_iv)
    protection_rows.append({
        "Strike": K,
        "Put_mid": put["Mid"],
        "Put_IV_pct": 100 * put["IV_Mid"],
        "Price_at_ATM_IV": flat_price,
        "Market_over_flat": put["Mid"] / flat_price
    })

print(f"\nOTM puts priced at the ATM IV ({100 * atm_iv:.2f}% at K={atm_strike:.0f})"
      " vs their market mid")
print(pd.DataFrame(protection_rows).round(4).to_string(index=False))


