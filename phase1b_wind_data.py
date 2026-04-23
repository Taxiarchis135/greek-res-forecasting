"""
Greek RES Forecasting — Phase 1b: Multi-Location Wind Speed Data
================================================================
Fetches hourly wind speed from 5 representative Greek wind generation
zones via Open-Meteo (free, no API key needed), computes a weighted
average, and updates phase1_combined.csv with the improved variable.

Locations chosen to represent Greece's main wind resource areas:
  - Evia Island       (largest onshore wind cluster)
  - Limnos Island     (northern Aegean, high wind resource)
  - Rhodes Island     (southern Aegean winds)
  - Kozani            (northern mainland wind corridor)
  - Kalamata          (southern Peloponnese corridor)

Run this AFTER phase1_data_collection.py.
Output: updates ./data/phase1_combined.csv in place.

Requirements:
    pip install requests pandas
"""

import requests
import pandas as pd
import time

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

START_DATE = "2024-01-01"
END_DATE   = "2025-12-31"

# Wind generation zones with weights reflecting installed capacity
# Weights are approximate — adjust if you find official IPTO capacity data
WIND_LOCATIONS = [
    {"name": "Evia",      "lat": 38.52, "lon": 23.89, "weight": 0.30},
    {"name": "Limnos",    "lat": 39.91, "lon": 25.35, "weight": 0.25},
    {"name": "Rhodes",    "lat": 36.13, "lon": 27.92, "weight": 0.20},
    {"name": "Kozani",    "lat": 40.30, "lon": 21.79, "weight": 0.15},
    {"name": "Kalamata",  "lat": 37.04, "lon": 22.11, "weight": 0.10},
]

OUTPUT_DIR    = "./data"
COMBINED_FILE = f"{OUTPUT_DIR}/phase1_combined.csv"
WIND_RAW_FILE = f"{OUTPUT_DIR}/wind_multilocation_raw.csv"


# ─────────────────────────────────────────────
# FETCH WIND DATA PER LOCATION
# ─────────────────────────────────────────────

def fetch_wind_for_location(name: str, lat: float, lon: float) -> pd.Series:
    """
    Fetch hourly wind speed (m/s) at 10m for a single location.
    Returns a named Series indexed by UTC datetime.
    """
    print(f"  [open-meteo] Fetching wind data for {name} ({lat}, {lon})...")

    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude":        lat,
        "longitude":       lon,
        "start_date":      START_DATE,
        "end_date":        END_DATE,
        "hourly":          "windspeed_10m,windspeed_100m",
        "timezone":        "UTC",
        "wind_speed_unit": "ms",
    }

    resp = requests.get(url, params=params, timeout=60)
    resp.raise_for_status()
    data = resp.json()

    hourly = data["hourly"]
    df = pd.DataFrame({
        "datetime_utc":       pd.to_datetime(hourly["time"]),
        f"{name}_ws10m_ms":   hourly["windspeed_10m"],
        f"{name}_ws100m_ms":  hourly["windspeed_100m"],  # hub height more relevant for turbines
    })
    df.set_index("datetime_utc", inplace=True)
    df.index = df.index.tz_localize("UTC")

    print(f"  [open-meteo] {name}: {len(df)} rows | "
          f"avg 100m wind = {df[f'{name}_ws100m_ms'].mean():.2f} m/s")

    # Small delay to be polite to the API
    time.sleep(1)

    return df


# ─────────────────────────────────────────────
# COMPUTE WEIGHTED AVERAGE
# ─────────────────────────────────────────────

def compute_weighted_wind(all_dfs: list, locations: list) -> pd.DataFrame:
    """
    Joins all location DataFrames and computes a capacity-weighted
    average wind speed at 100m hub height.
    """
    print("\n[merge] Computing weighted average wind speed...")

    combined = pd.concat(all_dfs, axis=1)

    # Weighted average using 100m wind speed (more representative of turbine hub height)
    weighted_sum = sum(
        combined[f"{loc['name']}_ws100m_ms"] * loc["weight"]
        for loc in locations
    )
    total_weight = sum(loc["weight"] for loc in locations)
    combined["windspeed_greece_weighted_ms"] = weighted_sum / total_weight

    # Also keep simple average for comparison
    ws100m_cols = [f"{loc['name']}_ws100m_ms" for loc in locations]
    combined["windspeed_greece_simple_ms"] = combined[ws100m_cols].mean(axis=1)

    print(f"[merge] Weighted avg wind speed: {combined['windspeed_greece_weighted_ms'].mean():.2f} m/s")
    print(f"[merge] Simple avg wind speed:   {combined['windspeed_greece_simple_ms'].mean():.2f} m/s")
    print(f"[merge] Athens wind speed was:   check phase1_combined.csv → windspeed_ms column")

    return combined


# ─────────────────────────────────────────────
# UPDATE COMBINED DATASET
# ─────────────────────────────────────────────

def update_combined_csv(wind_df: pd.DataFrame):
    """
    Loads phase1_combined.csv, replaces windspeed_ms with the new
    weighted variable, and saves it back in place.
    Also backs up the original file.
    """
    print(f"\n[update] Loading {COMBINED_FILE}...")
    combined = pd.read_csv(COMBINED_FILE, index_col="datetime_utc", parse_dates=True)

    if combined.index.tz is None:
        combined.index = combined.index.tz_localize("UTC")

    # Backup original
    backup_path = COMBINED_FILE.replace(".csv", "_backup.csv")
    combined.to_csv(backup_path)
    print(f"[update] Original backed up → {backup_path}")

    # Drop old Athens-only windspeed column
    if "windspeed_ms" in combined.columns:
        combined.drop(columns=["windspeed_ms"], inplace=True)
        print("[update] Removed old Athens-only windspeed_ms column")

    # Join new wind columns
    new_wind_cols = ["windspeed_greece_weighted_ms", "windspeed_greece_simple_ms"]
    combined = combined.join(wind_df[new_wind_cols], how="left")

    # Forward-fill any small gaps
    combined[new_wind_cols] = combined[new_wind_cols].ffill(limit=3)

    missing = combined[new_wind_cols].isnull().sum()
    if missing.sum() > 0:
        print(f"[update] Remaining missing values: {missing.to_dict()}")
    else:
        print("[update] No missing values in new wind columns.")

    combined.to_csv(COMBINED_FILE)
    print(f"[update] Saved updated dataset → {COMBINED_FILE}")
    print(f"[update] Shape: {combined.shape}")

    return combined


# ─────────────────────────────────────────────
# QUICK CORRELATION CHECK
# ─────────────────────────────────────────────

def check_correlation(combined: pd.DataFrame):
    """
    Prints correlation of old vs new wind speed variable
    against WindOnshore_MW so you can see the improvement.
    """
    print("\n[check] Correlation with WindOnshore_MW:")

    wind_gen_col = next((c for c in combined.columns if "windonshore" in c.lower()), None)
    if wind_gen_col is None:
        print("[check] Could not find WindOnshore_MW column — skipping.")
        return

    cols_to_check = ["windspeed_greece_weighted_ms", "windspeed_greece_simple_ms"]
    cols_to_check = [c for c in cols_to_check if c in combined.columns]

    for col in cols_to_check:
        r = combined[[wind_gen_col, col]].corr().iloc[0, 1]
        print(f"  {col:40s} → r = {r:.3f}")

    print("\n  (Athens-only windspeed_ms was r = ~0.34 — compare improvement)")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Greek RES Forecasting — Phase 1b: Multi-Location Wind Data")
    print("=" * 60)

    # 1. Fetch wind data for each location
    print("\n[step 1] Fetching wind data from 5 Greek locations...")
    all_dfs = []
    for loc in WIND_LOCATIONS:
        df = fetch_wind_for_location(loc["name"], loc["lat"], loc["lon"])
        all_dfs.append(df)

    # 2. Save raw multi-location data
    raw_combined = pd.concat(all_dfs, axis=1)
    raw_combined.to_csv(WIND_RAW_FILE)
    print(f"\n[save] Raw multi-location wind data → {WIND_RAW_FILE}")

    # 3. Compute weighted average
    wind_df = compute_weighted_wind(all_dfs, WIND_LOCATIONS)

    # 4. Update phase1_combined.csv
    updated = update_combined_csv(wind_df)

    # 5. Correlation check
    check_correlation(updated)

    print("\n" + "=" * 60)
    print("Phase 1b complete.")
    print("Your phase1_combined.csv now has improved wind speed data.")
    print("Re-run phase2_eda.ipynb to see the updated correlation matrix.")
    print("=" * 60)


if __name__ == "__main__":
    main()
