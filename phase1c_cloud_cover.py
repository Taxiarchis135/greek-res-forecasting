"""
Greek RES Forecasting — Weather Update with Cloud Cover
=======================================================
Re-pulls weather data from Open-Meteo adding cloud cover variables,
then updates phase1_combined.csv with the new features.

Run this as a standalone script — no need to re-run full Phase 1.

Requirements:
    pip install requests pandas
"""

import requests
import pandas as pd

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

START_DATE  = "2024-01-01"
END_DATE    = "2025-12-31"
LAT         = 37.98
LON         = 23.73

OUTPUT_DIR    = "./data"
COMBINED_FILE = f"{OUTPUT_DIR}/phase1_combined.csv"
WEATHER_FILE  = f"{OUTPUT_DIR}/weather_athens_updated.csv"


# ─────────────────────────────────────────────
# FETCH UPDATED WEATHER DATA
# ─────────────────────────────────────────────

def fetch_weather_with_clouds(start: str, end: str, lat: float, lon: float) -> pd.DataFrame:
    """
    Fetch hourly weather from Open-Meteo including cloud cover variables.
    """
    print(f"[open-meteo] Fetching updated weather data {start} → {end}...")

    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude":        lat,
        "longitude":       lon,
        "start_date":      start,
        "end_date":        end,
        "hourly":          ",".join([
            "shortwave_radiation",
            "windspeed_10m",
            "temperature_2m",
            "cloudcover",
            "cloudcover_low",
            "cloudcover_mid",
            "cloudcover_high",
        ]),
        "timezone":        "UTC",
        "wind_speed_unit": "ms",
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
        "cloudcover_pct":      hourly["cloudcover"],
        "cloudcover_low_pct":  hourly["cloudcover_low"],
        "cloudcover_mid_pct":  hourly["cloudcover_mid"],
        "cloudcover_high_pct": hourly["cloudcover_high"],
    })
    df.set_index("datetime_utc", inplace=True)
    df.index = df.index.tz_localize("UTC")

    print(f"[open-meteo] Fetched {len(df):,} rows")
    print(f"[open-meteo] Avg cloud cover: {df['cloudcover_pct'].mean():.1f}%")
    print(f"[open-meteo] Avg low cloud:   {df['cloudcover_low_pct'].mean():.1f}%")
    return df


# ─────────────────────────────────────────────
# UPDATE COMBINED DATASET
# ─────────────────────────────────────────────

def update_combined(weather_df: pd.DataFrame):
    """
    Loads phase1_combined.csv, drops old weather columns,
    joins updated weather with cloud cover, saves back.
    """
    print(f"\n[update] Loading {COMBINED_FILE}...")
    combined = pd.read_csv(COMBINED_FILE, index_col="datetime_utc", parse_dates=True)

    if combined.index.tz is None:
        combined.index = combined.index.tz_localize("UTC")

    # Backup original
    backup_path = COMBINED_FILE.replace(".csv", "_pre_cloud_backup.csv")
    combined.to_csv(backup_path)
    print(f"[update] Backed up original → {backup_path}")

    # Drop old weather columns that will be replaced
    old_weather_cols = ["irradiance_Wm2", "windspeed_ms", "temperature_C"]
    cols_to_drop = [c for c in old_weather_cols if c in combined.columns]
    combined.drop(columns=cols_to_drop, inplace=True)
    print(f"[update] Dropped old weather columns: {cols_to_drop}")

    # Also drop cloud columns if they already exist (re-run safety)
    cloud_cols = ["cloudcover_pct", "cloudcover_low_pct",
                  "cloudcover_mid_pct", "cloudcover_high_pct"]
    cols_to_drop2 = [c for c in cloud_cols if c in combined.columns]
    if cols_to_drop2:
        combined.drop(columns=cols_to_drop2, inplace=True)
        print(f"[update] Dropped existing cloud columns: {cols_to_drop2}")

    # Join updated weather
    combined = combined.join(weather_df, how="left")

    # Forward fill small gaps
    weather_new_cols = list(weather_df.columns)
    combined[weather_new_cols] = combined[weather_new_cols].ffill(limit=3)

    missing = combined[weather_new_cols].isnull().sum()
    if missing.sum() > 0:
        print(f"[update] Missing values: {missing[missing > 0].to_dict()}")
    else:
        print("[update] No missing values — clean dataset.")

    print(f"[update] Final shape: {combined.shape}")
    print(f"[update] New columns: {list(combined.columns)}")
    return combined


# ─────────────────────────────────────────────
# CORRELATION CHECK
# ─────────────────────────────────────────────

def check_cloud_correlations(df: pd.DataFrame):
    """
    Print correlation of cloud cover variables with solar generation.
    """
    solar_col = next((c for c in df.columns if 'solar' in c.lower()
                      and 'mw' in c.lower()), None)
    if solar_col is None:
        print("[check] Solar column not found.")
        return

    cloud_cols = ["irradiance_Wm2", "cloudcover_pct", "cloudcover_low_pct",
                  "cloudcover_mid_pct", "cloudcover_high_pct", "temperature_C"]
    cloud_cols = [c for c in cloud_cols if c in df.columns]

    print("\n[check] Correlation with Solar Generation (MW):")
    for col in cloud_cols:
        r = df[[solar_col, col]].corr().iloc[0, 1]
        print(f"  {col:30s} → r = {r:.3f}")

    print("\n[check] Expected: cloudcover_pct and cloudcover_low_pct")
    print("         should show NEGATIVE correlation with solar")
    print("         (more clouds = less solar generation)")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Weather Update — Adding Cloud Cover Variables")
    print("=" * 60)

    # 1. Fetch updated weather
    weather_df = fetch_weather_with_clouds(START_DATE, END_DATE, LAT, LON)

    # 2. Save raw updated weather file
    weather_df.to_csv(WEATHER_FILE)
    print(f"\n[save] Updated weather → {WEATHER_FILE}")

    # 3. Update combined dataset
    combined = update_combined(weather_df)

    # 4. Correlation check
    check_cloud_correlations(combined)

    # 5. Save updated combined
    combined.to_csv(COMBINED_FILE)
    print(f"\n[save] Updated combined dataset → {COMBINED_FILE}")

    print("\n" + "=" * 60)
    print("Weather update complete.")
    print("Next step: re-run phase3_forecasting.ipynb to retrain models")
    print("=" * 60)


if __name__ == "__main__":
    main()
