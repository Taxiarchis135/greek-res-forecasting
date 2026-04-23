"""
Greek RES Forecasting — Phase 4a: Day-Ahead Price Data Collection
=================================================================
Pulls day-ahead electricity prices for Greece (BZN|GR) from the
ENTSO-E Transparency Platform and merges them with the existing
phase1_combined.csv dataset.

Run this AFTER phase1_data_collection.py and phase1b_wind_data.py.
Output: ./data/phase4_with_prices.csv — full dataset including prices.

Requirements:
    pip install entsoe-py pandas
"""

import os
import pandas as pd
from entsoe import EntsoePandasClient

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

ENTSOE_API_KEY = os.getenv("ENTSOE_API_KEY", "YOUR_API_KEY_HERE")

START_DATE   = "2024-01-01"
END_DATE     = "2025-12-31"
BIDDING_ZONE = "10YGR-HTSO-----Y"

OUTPUT_DIR    = "./data"
COMBINED_FILE = f"{OUTPUT_DIR}/phase1_combined.csv"
OUTPUT_FILE   = f"{OUTPUT_DIR}/phase4_with_prices.csv"
PRICES_FILE   = f"{OUTPUT_DIR}/dayahead_prices.csv"


# ─────────────────────────────────────────────
# FETCH DAY-AHEAD PRICES
# ─────────────────────────────────────────────

def fetch_dayahead_prices(api_key: str, start: str, end: str, zone: str) -> pd.DataFrame:
    """
    Fetch day-ahead electricity prices (EUR/MWh) for Greece from ENTSO-E.
    """
    print(f"[entsoe] Fetching day-ahead prices {start} → {end} for zone {zone}...")

    client = EntsoePandasClient(api_key=api_key)

    start_ts = pd.Timestamp(start, tz="Europe/Athens")
    end_ts   = pd.Timestamp(end,   tz="Europe/Athens") + pd.Timedelta(days=1)

    try:
        prices = client.query_day_ahead_prices(zone, start=start_ts, end=end_ts)
    except Exception as e:
        print(f"[entsoe] ERROR: {e}")
        raise

    # Convert to DataFrame
    df = prices.to_frame(name="price_EURperMWh")
    df.index.name = "datetime_utc"

    # Resample to hourly if needed
    df = df.resample("1h").mean()

    # Convert to UTC
    df.index = df.index.tz_convert("UTC")

    # Basic stats
    print(f"[entsoe] Fetched {len(df):,} hourly price rows")
    print(f"[entsoe] Price range: {df['price_EURperMWh'].min():.2f} → "
          f"{df['price_EURperMWh'].max():.2f} EUR/MWh")
    print(f"[entsoe] Average price: {df['price_EURperMWh'].mean():.2f} EUR/MWh")
    print(f"[entsoe] Negative price hours: "
          f"{(df['price_EURperMWh'] < 0).sum()} "
          f"({(df['price_EURperMWh'] < 0).mean()*100:.1f}%)")

    return df


# ─────────────────────────────────────────────
# MERGE WITH EXISTING DATASET
# ─────────────────────────────────────────────

def merge_with_combined(prices_df: pd.DataFrame) -> pd.DataFrame:
    """
    Loads phase1_combined.csv and merges day-ahead prices on UTC datetime.
    """
    print(f"\n[merge] Loading {COMBINED_FILE}...")
    combined = pd.read_csv(COMBINED_FILE, index_col="datetime_utc", parse_dates=True)

    if combined.index.tz is None:
        combined.index = combined.index.tz_localize("UTC")

    print(f"[merge] Combined dataset: {len(combined):,} rows")
    print(f"[merge] Joining prices on UTC datetime...")

    merged = combined.join(prices_df, how="left")

    # Check coverage
    missing_prices = merged["price_EURperMWh"].isnull().sum()
    print(f"[merge] Missing price values: {missing_prices} "
          f"({missing_prices/len(merged)*100:.1f}%)")

    if missing_prices > 0:
        print("[merge] Forward-filling gaps ≤ 3 hours...")
        merged["price_EURperMWh"] = merged["price_EURperMWh"].ffill(limit=3)

    # Add useful derived price columns
    merged["log_price"]          = merged["price_EURperMWh"].apply(
                                    lambda x: pd.NA if x <= 0 else __import__('numpy').log(x))
    merged["is_negative_price"]  = (merged["price_EURperMWh"] < 0).astype(int)
    merged["price_rolling24h"]   = merged["price_EURperMWh"].rolling(24).mean()

    print(f"[merge] Final dataset shape: {merged.shape}")
    return merged


# ─────────────────────────────────────────────
# QUICK SANITY CHECK
# ─────────────────────────────────────────────

def sanity_check(df: pd.DataFrame):
    """
    Prints key statistics to verify the merge worked correctly.
    """
    solar_col = next((c for c in df.columns if 'solar' in c.lower() and 'mw' in c.lower()), None)
    wind_col  = next((c for c in df.columns if 'windonshore' in c.lower()), None)

    print("\n[check] Sample of merged dataset:")
    print(df[["price_EURperMWh", solar_col, wind_col]].dropna().head(10).round(2).to_string())

    print("\n[check] Annual average prices:")
    for year in [2024, 2025]:
        year_data = df[df.index.year == year]["price_EURperMWh"]
        if len(year_data) > 0:
            print(f"  {year}: avg = {year_data.mean():.2f} EUR/MWh | "
                  f"min = {year_data.min():.2f} | max = {year_data.max():.2f}")

    print("\n[check] Correlation — RES generation vs price:")
    corr_solar = df[["price_EURperMWh", solar_col]].corr().iloc[0, 1]
    corr_wind  = df[["price_EURperMWh", wind_col]].corr().iloc[0, 1]
    corr_total = df[["price_EURperMWh", "TotalRES_MW"]].corr().iloc[0, 1] \
                 if "TotalRES_MW" in df.columns else "N/A"
    print(f"  Solar MW  ↔ Price: {corr_solar:.3f}")
    print(f"  Wind MW   ↔ Price: {corr_wind:.3f}")
    print(f"  Total RES ↔ Price: {corr_total:.3f}")
    print("\n  (Negative correlations confirm merit order effect — more RES = lower prices)")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Greek RES Forecasting — Phase 4a: Day-Ahead Price Data")
    print("=" * 60)

    if ENTSOE_API_KEY == "YOUR_API_KEY_HERE":
        print("\n[!] ENTSO-E API key not set.")
        print("    Edit ENTSOE_API_KEY in this script or set env variable.")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Fetch prices
    prices_df = fetch_dayahead_prices(ENTSOE_API_KEY, START_DATE, END_DATE, BIDDING_ZONE)

    # 2. Save raw prices
    prices_df.to_csv(PRICES_FILE)
    print(f"\n[save] Raw prices → {PRICES_FILE}")

    # 3. Merge with combined dataset
    merged = merge_with_combined(prices_df)

    # 4. Sanity check
    sanity_check(merged)

    # 5. Save final dataset
    merged.to_csv(OUTPUT_FILE)
    print(f"\n[save] Full dataset with prices → {OUTPUT_FILE}")

    print("\n" + "=" * 60)
    print("Phase 4a complete.")
    print("Next step: open phase4_analysis.ipynb")
    print("=" * 60)


if __name__ == "__main__":
    main()
