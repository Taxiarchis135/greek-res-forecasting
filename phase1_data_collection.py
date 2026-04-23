"""
Greek RES Forecasting Project — Phase 1: Data Collection
=========================================================
Pulls two data sources:
  1. ENTSO-E Transparency Platform — hourly wind & solar generation for Greece (BZN|GR)
  2. Open-Meteo Archive API          — hourly weather for Athens (irradiance + wind speed)

Requirements:
    pip install entsoe-py requests pandas

ENTSO-E API key:
    Register for free at https://transparency.entsoe.eu → My Account → Web API Security Token
    Then set the env variable:  export ENTSOE_API_KEY="your_token_here"
    Or paste it directly into the ENTSOE_API_KEY variable below.

Output files (saved to ./data/):
    entsoe_res_generation.csv   — hourly wind + solar generation (MW), Greece
    weather_athens.csv          — hourly irradiance + wind speed, Athens
    phase1_combined.csv         — merged dataset ready for Phase 2 EDA
"""

import os
import requests
import pandas as pd
from datetime import datetime, timedelta
from entsoe import EntsoePandasClient

# ─────────────────────────────────────────────
# CONFIGURATION — edit these
# ─────────────────────────────────────────────

ENTSOE_API_KEY = os.getenv("ENTSOE_API_KEY", "YOUR_API_KEY_HERE")

# Date range: 2 full years of hourly data
START_DATE = "2024-01-01"
END_DATE   = "2025-12-31"

# Greek bidding zone
BIDDING_ZONE = "10YGR-HTSO-----Y"

# Athens coordinates (representative for weather)
LAT = 37.98
LON = 23.73

# Output directory
OUTPUT_DIR = "./data"

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"[setup] Output directory ready: {OUTPUT_DIR}")


def fetch_entsoe_generation(api_key: str, start: str, end: str, zone: str) -> pd.DataFrame:
    """
    Fetch actual generation per production type from ENTSO-E.
    Returns a DataFrame with columns: WindOnshore, WindOffshore, Solar (all in MW).
    """
    print(f"[entsoe] Fetching generation data {start} → {end} for zone {zone}...")

    client = EntsoePandasClient(api_key=api_key)

    start_ts = pd.Timestamp(start, tz="Europe/Athens")
    end_ts   = pd.Timestamp(end,   tz="Europe/Athens") + pd.Timedelta(days=1)

    try:
        raw = client.query_generation(zone, start=start_ts, end=end_ts)
    except Exception as e:
        print(f"[entsoe] ERROR: {e}")
        print("         Check your API key and that your IP is whitelisted on ENTSO-E.")
        raise

    # entsoe-py returns a MultiIndex DataFrame; flatten to what we need
    # Column names vary by zone — print available ones if needed
    print(f"[entsoe] Available generation types: {list(raw.columns)}")

    # Extract RES columns — handle both tuple and string column formats
    cols_map = {}
    for col in raw.columns:
        col_str = str(col).lower()
        if "wind onshore" in col_str:
            cols_map["WindOnshore_MW"] = col
        elif "wind offshore" in col_str:
            cols_map["WindOffshore_MW"] = col
        elif "solar" in col_str:
            cols_map["Solar_MW"] = col

    if not cols_map:
        print("[entsoe] WARNING: No wind/solar columns found. Printing raw columns for debugging:")
        print(raw.columns.tolist())
        raise ValueError("Could not find RES columns in ENTSO-E response.")

    df = raw[[v for v in cols_map.values()]].copy()
    df.columns = list(cols_map.keys())

    # Resample to hourly if data arrives at 15-min or 30-min intervals
    df = df.resample("1h").mean()

    # Flatten timezone to UTC for clean merging
    df.index = df.index.tz_convert("UTC")
    df.index.name = "datetime_utc"

    # Compute total RES
    df["TotalRES_MW"] = df.sum(axis=1)

    print(f"[entsoe] Fetched {len(df)} hourly rows. Columns: {list(df.columns)}")
    return df


def fetch_weather_openmeteo(start: str, end: str, lat: float, lon: float) -> pd.DataFrame:
    """
    Fetch historical hourly weather from the Open-Meteo Archive API (free, no key needed).
    Variables: shortwave_radiation (W/m²), windspeed_10m (km/h), temperature_2m (°C).
    """
    print(f"[weather] Fetching Open-Meteo archive {start} → {end} for ({lat}, {lon})...")

    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude":            lat,
        "longitude":           lon,
        "start_date":          start,
        "end_date":            end,
        "hourly":              "shortwave_radiation,windspeed_10m,temperature_2m",
        "timezone":            "UTC",
        "wind_speed_unit":     "ms",   # metres per second — more useful for energy
    }

    resp = requests.get(url, params=params, timeout=60)
    resp.raise_for_status()
    data = resp.json()

    hourly = data["hourly"]
    df = pd.DataFrame({
        "datetime_utc":        pd.to_datetime(hourly["time"]),
        "irradiance_Wm2":      hourly["shortwave_radiation"],
        "windspeed_ms":        hourly["windspeed_10m"],
        "temperature_C":       hourly["temperature_2m"],
    })
    df.set_index("datetime_utc", inplace=True)
    df.index = df.index.tz_localize("UTC")

    print(f"[weather] Fetched {len(df)} hourly rows.")
    return df


def merge_and_validate(gen_df: pd.DataFrame, weather_df: pd.DataFrame) -> pd.DataFrame:
    """
    Inner-join on UTC datetime, run basic quality checks, report missing values.
    """
    print("[merge] Joining generation + weather on UTC datetime...")
    combined = gen_df.join(weather_df, how="inner")

    total_rows    = len(combined)
    expected_rows = (
        pd.Timestamp(END_DATE, tz="UTC") - pd.Timestamp(START_DATE, tz="UTC")
    ).days * 24

    print(f"[merge] Rows: {total_rows} (expected ~{expected_rows})")

    # Missing value report
    missing = combined.isnull().sum()
    if missing.sum() > 0:
        print("[merge] Missing values per column:")
        print(missing[missing > 0].to_string())
        print("[merge] Forward-filling gaps ≤ 3 hours...")
        combined = combined.ffill(limit=3)
    else:
        print("[merge] No missing values — dataset is clean.")

    # Sanity checks
    for col in combined.columns:
        neg_count = (combined[col] < 0).sum()
        if neg_count > 0:
            print(f"[merge] WARNING: {neg_count} negative values in '{col}' — clipping to 0.")
            combined[col] = combined[col].clip(lower=0)

    # Add local time column for convenience (EDA plots)
    combined["datetime_athens"] = combined.index.tz_convert("Europe/Athens")

    # Add derived features useful in Phase 2
    combined["hour_of_day"]   = combined.index.hour
    combined["month"]         = combined.index.month
    combined["day_of_week"]   = combined.index.dayofweek   # 0=Mon
    combined["is_weekend"]    = combined["day_of_week"].isin([5, 6]).astype(int)

    print(f"[merge] Final dataset shape: {combined.shape}")
    return combined


def save(df: pd.DataFrame, filename: str):
    path = os.path.join(OUTPUT_DIR, filename)
    df.to_csv(path)
    print(f"[save] Saved → {path}  ({len(df)} rows × {len(df.columns)} cols)")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Greek RES Forecasting — Phase 1: Data Collection")
    print("=" * 60)

    if ENTSOE_API_KEY == "YOUR_API_KEY_HERE":
        print("\n[!] You haven't set your ENTSO-E API key.")
        print("    Register at https://transparency.entsoe.eu")
        print("    Then set: export ENTSOE_API_KEY='your_token'")
        print("    Or edit ENTSOE_API_KEY directly in this script.\n")
        return

    ensure_output_dir()

    # 1. ENTSO-E generation data
    gen_df = fetch_entsoe_generation(ENTSOE_API_KEY, START_DATE, END_DATE, BIDDING_ZONE)
    save(gen_df, "entsoe_res_generation.csv")

    # 2. Open-Meteo weather data
    weather_df = fetch_weather_openmeteo(START_DATE, END_DATE, LAT, LON)
    save(weather_df, "weather_athens.csv")

    # 3. Merge + validate
    combined = merge_and_validate(gen_df, weather_df)
    save(combined, "phase1_combined.csv")

    print("\n" + "=" * 60)
    print("Phase 1 complete. Dataset summary:")
    print("=" * 60)
    print(combined.describe().round(2).to_string())
    print("\nNext step: open phase2_eda.ipynb and load phase1_combined.csv")


if __name__ == "__main__":
    main()
