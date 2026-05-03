"""
Greek RES Forecasting — Daily Data Update
==========================================
Checks the last date in existing CSVs and pulls only the missing
days from ENTSO-E and Open-Meteo. Appends new rows to existing files.

Run manually or via GitHub Actions every day at 7:00 AM UTC.

Requirements:
    pip install entsoe-py requests pandas
"""

import os
import requests
import pandas as pd
from entsoe import EntsoePandasClient

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

ENTSOE_API_KEY = os.getenv("ENTSOE_API_KEY", "YOUR_API_KEY_HERE")
BIDDING_ZONE   = "10YGR-HTSO-----Y"
LAT, LON       = 37.98, 23.73

OUTPUT_DIR     = "./data"
COMBINED_FILE  = f"{OUTPUT_DIR}/phase1_combined.csv"
PRICES_FILE    = f"{OUTPUT_DIR}/dayahead_prices.csv"
PHASE4_FILE    = f"{OUTPUT_DIR}/phase4_with_prices.csv"

# How many days back to check for missing data
LOOKBACK_DAYS  = 3  # pulls last 3 days to catch any reporting lags


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def get_last_date(filepath: str) -> pd.Timestamp:
    """Returns the last datetime in an existing CSV file."""
    df = pd.read_csv(filepath, index_col=0, parse_dates=True)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    last = df.index.max()
    print(f"[check] Last date in {filepath}: {last.date()}")
    return last


def fetch_new_generation(api_key: str, start: pd.Timestamp,
                          end: pd.Timestamp, zone: str) -> pd.DataFrame:
    """Fetch new RES generation data from ENTSO-E."""
    print(f"[entsoe] Fetching generation {start.date()} → {end.date()}...")
    client = EntsoePandasClient(api_key=api_key)

    try:
        raw = client.query_generation(zone, start=start, end=end)
    except Exception as e:
        print(f"[entsoe] ERROR fetching generation: {e}")
        return pd.DataFrame()

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
        print("[entsoe] No RES columns found.")
        return pd.DataFrame()

    df = raw[[v for v in cols_map.values()]].copy()
    df.columns = list(cols_map.keys())
    df = df.resample("1h").mean()
    df.index = df.index.tz_convert("UTC")
    df.index.name = "datetime_utc"

    if "Solar_MW" in df.columns and "WindOnshore_MW" in df.columns:
        df["TotalRES_MW"] = df[["Solar_MW", "WindOnshore_MW"]].sum(axis=1)

    df = df.clip(lower=0)
    print(f"[entsoe] Got {len(df)} new generation rows")
    return df


def fetch_new_weather(start: pd.Timestamp, end: pd.Timestamp,
                       lat: float, lon: float) -> pd.DataFrame:
    """Fetch new weather data from Open-Meteo including cloud cover."""
    start_str = start.strftime("%Y-%m-%d")
    end_str   = end.strftime("%Y-%m-%d")
    print(f"[weather] Fetching weather {start_str} → {end_str}...")

    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude":        lat,
        "longitude":       lon,
        "start_date":      start_str,
        "end_date":        end_str,
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

    try:
        resp = requests.get(url, params=params, timeout=60)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"[weather] ERROR: {e}")
        return pd.DataFrame()

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

    print(f"[weather] Got {len(df)} new weather rows")
    return df


def fetch_new_prices(api_key: str, start: pd.Timestamp,
                      end: pd.Timestamp, zone: str) -> pd.DataFrame:
    """Fetch new day-ahead prices from ENTSO-E."""
    print(f"[prices] Fetching prices {start.date()} → {end.date()}...")
    client = EntsoePandasClient(api_key=api_key)

    try:
        prices = client.query_day_ahead_prices(zone, start=start, end=end)
    except Exception as e:
        print(f"[prices] ERROR: {e}")
        return pd.DataFrame()

    df = prices.to_frame(name="price_EURperMWh")
    df.index.name = "datetime_utc"
    df = df.resample("1h").mean()
    df.index = df.index.tz_convert("UTC")

    print(f"[prices] Got {len(df)} new price rows")
    return df


def fetch_multilocation_wind(start: pd.Timestamp,
                              end: pd.Timestamp) -> pd.DataFrame:
    """Fetch wind speed from 5 Greek locations and compute weighted average."""
    locations = [
        {"name": "Evia",     "lat": 38.52, "lon": 23.89, "weight": 0.30},
        {"name": "Limnos",   "lat": 39.91, "lon": 25.35, "weight": 0.25},
        {"name": "Rhodes",   "lat": 36.13, "lon": 27.92, "weight": 0.20},
        {"name": "Kozani",   "lat": 40.30, "lon": 21.79, "weight": 0.15},
        {"name": "Kalamata", "lat": 37.04, "lon": 22.11, "weight": 0.10},
    ]

    start_str = start.strftime("%Y-%m-%d")
    end_str   = end.strftime("%Y-%m-%d")
    all_ws    = []

    for loc in locations:
        url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude":        loc["lat"],
            "longitude":       loc["lon"],
            "start_date":      start_str,
            "end_date":        end_str,
            "hourly":          "windspeed_100m",
            "timezone":        "UTC",
            "wind_speed_unit": "ms",
        }
        try:
            resp = requests.get(url, params=params, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            s = pd.Series(
                data["hourly"]["windspeed_100m"],
                index=pd.to_datetime(data["hourly"]["time"]),
                name=loc["name"]
            )
            s.index = s.index.tz_localize("UTC")
            all_ws.append((s, loc["weight"]))
        except Exception as e:
            print(f"[wind] ERROR for {loc['name']}: {e}")

    if not all_ws:
        return pd.DataFrame()

    weighted = sum(s * w for s, w in all_ws) / sum(w for _, w in all_ws)
    df = weighted.to_frame(name="windspeed_greece_weighted_ms")
    df.index.name = "datetime_utc"
    print(f"[wind] Got {len(df)} weighted wind rows")
    return df


# ─────────────────────────────────────────────
# MAIN UPDATE LOGIC
# ─────────────────────────────────────────────

def update_combined(new_gen: pd.DataFrame, new_weather: pd.DataFrame,
                    new_wind: pd.DataFrame) -> bool:
    """Append new rows to phase1_combined.csv."""
    if new_gen.empty or new_weather.empty:
        print("[update] Skipping combined update — missing data.")
        return False

    # Load existing
    existing = pd.read_csv(COMBINED_FILE, index_col="datetime_utc", parse_dates=True)
    if existing.index.tz is None:
        existing.index = existing.index.tz_localize("UTC")

    # Build new rows by joining generation + weather + wind
    new_rows = new_gen.join(new_weather, how="inner")
    if not new_wind.empty:
        new_rows = new_rows.join(new_wind, how="left")

    # Add time features
    new_rows["hour_of_day"]  = new_rows.index.hour
    new_rows["month"]        = new_rows.index.month
    new_rows["day_of_week"]  = new_rows.index.dayofweek
    new_rows["is_weekend"]   = new_rows.index.dayofweek.isin([5, 6]).astype(int)
    new_rows["datetime_athens"] = new_rows.index.tz_convert("Europe/Athens")

    # Remove overlapping rows then append
    new_rows = new_rows[~new_rows.index.isin(existing.index)]

    if len(new_rows) == 0:
        print("[update] No new rows to add to combined dataset.")
        return False

    updated = pd.concat([existing, new_rows])
    updated = updated.sort_index()
    updated.to_csv(COMBINED_FILE)
    print(f"[update] Added {len(new_rows)} new rows to phase1_combined.csv")
    print(f"[update] Dataset now covers: {updated.index.min().date()} → {updated.index.max().date()}")
    return True


def update_prices(new_prices: pd.DataFrame) -> bool:
    """Append new price rows to dayahead_prices.csv."""
    if new_prices.empty:
        print("[prices] No new price data to append.")
        return False

    existing = pd.read_csv(PRICES_FILE, index_col="datetime_utc", parse_dates=True)
    if existing.index.tz is None:
        existing.index = existing.index.tz_localize("UTC")

    new_prices = new_prices[~new_prices.index.isin(existing.index)]

    if len(new_prices) == 0:
        print("[prices] No new price rows to add.")
        return False

    updated = pd.concat([existing, new_prices]).sort_index()
    updated.to_csv(PRICES_FILE)
    print(f"[prices] Added {len(new_prices)} new price rows")
    return True


def rebuild_phase4():
    """Rebuild phase4_with_prices.csv by merging updated combined + prices."""
    print("[phase4] Rebuilding phase4_with_prices.csv...")

    combined = pd.read_csv(COMBINED_FILE, index_col="datetime_utc", parse_dates=True)
    prices   = pd.read_csv(PRICES_FILE,   index_col="datetime_utc", parse_dates=True)

    if combined.index.tz is None:
        combined.index = combined.index.tz_localize("UTC")
    if prices.index.tz is None:
        prices.index = prices.index.tz_localize("UTC")

    merged = combined.join(prices, how="left")
    merged["price_EURperMWh"]   = merged["price_EURperMWh"].ffill(limit=3)
    merged["is_negative_price"] = (merged["price_EURperMWh"] < 0).astype(int)
    merged["price_rolling24h"]  = merged["price_EURperMWh"].rolling(24).mean()

    merged.to_csv(PHASE4_FILE)
    print(f"[phase4] Saved → {PHASE4_FILE} ({len(merged):,} rows)")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Greek RES Forecasting — Daily Data Update")
    print("=" * 60)

    if ENTSOE_API_KEY == "YOUR_API_KEY_HERE":
        print("[!] ENTSO-E API key not set. Exiting.")
        return

    # Determine update window
    last_date  = get_last_date(COMBINED_FILE)
    start_date = (last_date - pd.Timedelta(days=LOOKBACK_DAYS)).normalize()
    end_date   = pd.Timestamp.utcnow().normalize()

    if start_date >= end_date:
        print("[update] Dataset is already up to date. Nothing to do.")
        return

    print(f"\n[update] Pulling data from {start_date.date()} → {end_date.date()}")

    # Convert for ENTSO-E (needs timezone-aware timestamps)
    entsoe_start = start_date.tz_localize("UTC") if start_date.tz is None else start_date
    entsoe_end   = end_date.tz_localize("UTC") if end_date.tz is None else end_date

    # Fetch all data sources
    new_gen     = fetch_new_generation(ENTSOE_API_KEY, entsoe_start, entsoe_end, BIDDING_ZONE)
    new_weather = fetch_new_weather(start_date, end_date, LAT, LON)
    new_wind    = fetch_multilocation_wind(start_date, end_date)
    new_prices  = fetch_new_prices(ENTSOE_API_KEY, entsoe_start, entsoe_end, BIDDING_ZONE)

    # Update files
    gen_updated    = update_combined(new_gen, new_weather, new_wind)
    price_updated  = update_prices(new_prices)

    if gen_updated or price_updated:
        rebuild_phase4()
        print("\n[done] Data update complete. Dashboard will reflect new data.")
    else:
        print("\n[done] No updates needed.")

    print("=" * 60)


if __name__ == "__main__":
    main()
