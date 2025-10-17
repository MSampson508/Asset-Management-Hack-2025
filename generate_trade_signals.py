"""
Generates a CSV file of monthly long and short stock selections
based on predicted returns from the XGBoost backtest (preds.csv).

For each month (ym), the script:
- Sorts all gvkeys by their predicted return (r_hat).
- Selects the top 100 as "LONG" and bottom 100 as "SHORT".
- Exports a unified CSV file showing the trading positions.
"""

import pandas as pd

# ===== User settings =====
INPUT_FILE = "preds.csv"         # Path to predictions file
OUTPUT_FILE = "trade_signals.csv"  # Path to output CSV
N_LONG = 100
N_SHORT = 100

# ===== Load predictions =====
print(f"Loading predictions from: {INPUT_FILE}")
preds = pd.read_csv(INPUT_FILE)

# Basic sanity check
required_cols = {"ym", "gvkey", "r_hat"}
if not required_cols.issubset(preds.columns):
    raise ValueError(f"Input file must contain columns: {required_cols}")

# Ensure correct sorting and typing
preds["ym"] = preds["ym"].astype(str)
preds = preds.sort_values(["ym", "r_hat"], ascending=[True, False])

# ===== Generate long/short selections =====
signals = []
for ym, group in preds.groupby("ym"):
    group_sorted = group.sort_values("r_hat", ascending=False)

    top = group_sorted.head(N_LONG).copy()
    top["position"] = "LONG"

    bottom = group_sorted.tail(N_SHORT).copy()
    bottom["position"] = "SHORT"

    signals.append(pd.concat([top, bottom]))

trade_signals = pd.concat(signals, ignore_index=True)

# ===== Save to CSV =====
trade_signals.to_csv(OUTPUT_FILE, index=False)
print(f"\n✅ Trade signals saved to: {OUTPUT_FILE}")
print(f"Total months processed: {trade_signals['ym'].nunique()}")
print(f"Total rows: {len(trade_signals)}")
print(f"Columns: {list(trade_signals.columns)}")

# ===== Example output structure =====
# ym, gvkey, r_hat, position
# 2020-03, 12345, 0.045, LONG
# 2020-03, 67890, -0.023, SHORT
# 2020-04, 13579, 0.038, LONG
