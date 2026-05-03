"""
Greek RES Forecasting — Day-Ahead Forecast
===========================================
Fetches tomorrow's weather forecast from Open-Meteo,
runs the trained models, and saves 24 hourly forecasts
for solar and wind generation to data/forecast_tomorrow.csv.

Run daily after update_data.py via GitHub Actions.

Requirements:
    pip install requests pandas numpy scikit-learn
"""

import os
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

OUTPUT_DIR    = "./data"
COMBINED_FILE = f"{OUTPUT_DIR}/phase1_combined.csv"
FORECAST_FILE = f"{OUTPUT_DIR}/forecast_tomorrow.csv"

LAT, LON = 37.98, 23.73

FEATURE_COLS = [
    'irradiance', 'windspeed', 'temperature',
    'cloudcover', 'cloudcover_low', 'cloudcover_high',
    'hour_sin', 'hour_cos', 'month_sin', 'month_cos',
    'is_weekend', 'season',
    'solar_lag24', 'solar_lag168', 'solar_roll24',
    'wind_lag24',  'wind_lag168',  'wind_roll24',
]


# ─────────────────────────────────────────────
# FETCH TOMORROW'S WEATHER FORECAST
# ─────────────────────────────────────────────

def fetch_forecast_weather(lat: float, lon: float) -> pd.DataFrame:
    """
    Fetch 48-hour weather forecast from Open-Meteo.
    Returns tomorrow's 24 hourly rows.
    """
    print("[forecast] Fetching tomorrow's weather forecast...")

    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude":        lat,
        "longitude":       lon,
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
        "forecast_days":   2,
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

    # Keep only tomorrow's 24 hours
    tomorrow = (pd.Timestamp.utcnow() + pd.Timedelta(days=1)).normalize()
    tomorrow_end = tomorrow + pd.Timedelta(hours=23)
    df = df[(df.index >= tomorrow) & (df.index <= tomorrow_end)]

    print(f"[forecast] Tomorrow: {tomorrow.date()} | {len(df)} hours")
    return df


def fetch_forecast_wind(lat: float, lon: float) -> pd.Series:
    """Fetch tomorrow's 100m wind speed forecast for Athens as proxy."""
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude":        lat,
        "longitude":       lon,
        "hourly":          "windspeed_100m",
        "timezone":        "UTC",
        "wind_speed_unit": "ms",
        "forecast_days":   2,
    }
    resp = requests.get(url, params=params, timeout=60)
    resp.raise_for_status()
    data = resp.json()

    s = pd.Series(
        data["hourly"]["windspeed_100m"],
        index=pd.to_datetime(data["hourly"]["time"]),
        name="windspeed_greece_weighted_ms"
    )
    s.index = s.index.tz_localize("UTC")

    tomorrow = (pd.Timestamp.utcnow() + pd.Timedelta(days=1)).normalize()
    tomorrow_end = tomorrow + pd.Timedelta(hours=23)
    return s[(s.index >= tomorrow) & (s.index <= tomorrow_end)]


# ─────────────────────────────────────────────
# TRAIN MODELS ON FULL HISTORICAL DATA
# ─────────────────────────────────────────────

def train_models(df: pd.DataFrame):
    """
    Train Random Forest (solar) and Linear Regression (wind)
    on the full historical dataset.
    """
    print("[train] Training models on full historical dataset...")

    solar_col = next(c for c in df.columns if 'solar' in c.lower() and 'mw' in c.lower())
    wind_col  = next(c for c in df.columns if 'windonshore' in c.lower())
    irrad_col = next(c for c in df.columns if 'irradiance' in c.lower())
    windsp_col = next(c for c in df.columns if 'weighted' in c.lower())
    temp_col  = next(c for c in df.columns if 'temperature' in c.lower())

    # Build features
    feat = pd.DataFrame(index=df.index)
    feat['target_solar'] = df[solar_col]
    feat['target_wind']  = df[wind_col]
    feat['irradiance']   = df[irrad_col]
    feat['windspeed']    = df[windsp_col]
    feat['temperature']  = df[temp_col]

    # Cloud cover
    if 'cloudcover_pct' in df.columns:
        feat['cloudcover']      = df['cloudcover_pct']
        feat['cloudcover_low']  = df['cloudcover_low_pct']
        feat['cloudcover_high'] = df['cloudcover_high_pct']
    else:
        feat['cloudcover']      = 0
        feat['cloudcover_low']  = 0
        feat['cloudcover_high'] = 0

    # Time features
    feat['hour_sin']     = np.sin(2 * np.pi * df.index.hour / 24)
    feat['hour_cos']     = np.cos(2 * np.pi * df.index.hour / 24)
    feat['month_sin']    = np.sin(2 * np.pi * df.index.month / 12)
    feat['month_cos']    = np.cos(2 * np.pi * df.index.month / 12)
    feat['is_weekend']   = df.index.dayofweek.isin([5, 6]).astype(int)
    feat['season']       = df.index.month % 12 // 3

    # Lag features
    feat['solar_lag24']  = df[solar_col].shift(24)
    feat['solar_lag168'] = df[solar_col].shift(168)
    feat['wind_lag24']   = df[wind_col].shift(24)
    feat['wind_lag168']  = df[wind_col].shift(168)
    feat['solar_roll24'] = df[solar_col].shift(24).rolling(24).mean()
    feat['wind_roll24']  = df[wind_col].shift(24).rolling(24).mean()

    feat = feat.dropna()

    available_cols = [c for c in FEATURE_COLS if c in feat.columns]
    X = feat[available_cols]
    y_solar = feat['target_solar']
    y_wind  = feat['target_wind']

    # Train Random Forest for solar
    rf_solar = RandomForestRegressor(
        n_estimators=200, max_depth=12,
        min_samples_leaf=10, random_state=42, n_jobs=-1)
    rf_solar.fit(X, y_solar)

    # Train Linear Regression for wind
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    lr_wind = LinearRegression()
    lr_wind.fit(X_scaled, y_wind)

    print(f"[train] Models trained on {len(feat):,} rows")
    return rf_solar, lr_wind, scaler, available_cols, feat


# ─────────────────────────────────────────────
# BUILD FORECAST FEATURES
# ─────────────────────────────────────────────

def build_forecast_features(weather_df: pd.DataFrame,
                              wind_series: pd.Series,
                              hist_feat: pd.DataFrame,
                              available_cols: list) -> pd.DataFrame:
    """
    Build feature matrix for tomorrow's 24 hours using
    weather forecast + lag values from historical data.
    """
    print("[forecast] Building forecast feature matrix...")

    feat = pd.DataFrame(index=weather_df.index)

    feat['irradiance']   = weather_df['irradiance_Wm2'].clip(lower=0)
    feat['windspeed']    = wind_series.reindex(weather_df.index).fillna(
                           wind_series.mean())
    feat['temperature']  = weather_df['temperature_C']
    feat['cloudcover']   = weather_df.get('cloudcover_pct', pd.Series(0, index=weather_df.index))
    feat['cloudcover_low']  = weather_df.get('cloudcover_low_pct', pd.Series(0, index=weather_df.index))
    feat['cloudcover_high'] = weather_df.get('cloudcover_high_pct', pd.Series(0, index=weather_df.index))

    feat['hour_sin']   = np.sin(2 * np.pi * feat.index.hour / 24)
    feat['hour_cos']   = np.cos(2 * np.pi * feat.index.hour / 24)
    feat['month_sin']  = np.sin(2 * np.pi * feat.index.month / 12)
    feat['month_cos']  = np.cos(2 * np.pi * feat.index.month / 12)
    feat['is_weekend'] = feat.index.dayofweek.isin([5, 6]).astype(int)
    feat['season']     = feat.index.month % 12 // 3

    # Lag features — use last available historical values
    for col in ['solar_lag24', 'solar_lag168', 'solar_roll24',
                'wind_lag24', 'wind_lag168', 'wind_roll24']:
        if col in hist_feat.columns:
            # Use the most recent available value as a proxy
            feat[col] = hist_feat[col].iloc[-1]
        else:
            feat[col] = 0

    # Keep only columns the model was trained on
    feat = feat[[c for c in available_cols if c in feat.columns]]
    for c in available_cols:
        if c not in feat.columns:
            feat[c] = 0

    return feat[available_cols]


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Greek RES Forecasting — Day-Ahead Forecast")
    print(f"Generating forecast for: {(datetime.now(timezone.utc) + pd.Timedelta(days=1)).strftime('%Y-%m-%d')}")
    print("=" * 60)

    # 1. Load historical data
    print("\n[load] Loading historical dataset...")
    df = pd.read_csv(COMBINED_FILE, index_col="datetime_utc", parse_dates=True)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    print(f"[load] {len(df):,} rows | up to {df.index.max().date()}")

    # 2. Train models on full dataset
    rf_solar, lr_wind, scaler, available_cols, hist_feat = train_models(df)

    # 3. Fetch tomorrow's weather forecast
    weather_df  = fetch_forecast_weather(LAT, LON)
    wind_series = fetch_forecast_wind(LAT, LON)

    if weather_df.empty:
        print("[forecast] No weather forecast data available. Exiting.")
        return

    # 4. Build forecast features
    X_tomorrow = build_forecast_features(
        weather_df, wind_series, hist_feat, available_cols)

    # 5. Generate forecasts
    solar_pred = np.clip(rf_solar.predict(X_tomorrow), 0, None)
    wind_pred  = np.clip(lr_wind.predict(scaler.transform(X_tomorrow)), 0, None)
    total_pred = solar_pred + wind_pred

    # 6. Build output DataFrame
    tomorrow = (pd.Timestamp.utcnow() + pd.Timedelta(days=1)).normalize()
    forecast_df = pd.DataFrame({
        "datetime_utc":        weather_df.index,
        "datetime_athens":     weather_df.index.tz_convert("Europe/Athens"),
        "hour_utc":            weather_df.index.hour,
        "forecast_solar_MW":   solar_pred.round(1),
        "forecast_wind_MW":    wind_pred.round(1),
        "forecast_total_MW":   total_pred.round(1),
        "irradiance_Wm2":      weather_df["irradiance_Wm2"].values,
        "cloudcover_pct":      weather_df["cloudcover_pct"].values,
        "temperature_C":       weather_df["temperature_C"].values,
    })
    forecast_df.set_index("datetime_utc", inplace=True)

    # 7. Save forecast
    forecast_df.to_csv(FORECAST_FILE)
    print(f"\n[save] Forecast saved → {FORECAST_FILE}")

    # 8. Print summary
    print(f"\n{'='*60}")
    print(f"Day-Ahead Forecast for {tomorrow.date()} (Athens time)")
    print(f"{'='*60}")
    print(f"Peak solar forecast:  {forecast_df['forecast_solar_MW'].max():.0f} MW "
          f"at {forecast_df['forecast_solar_MW'].idxmax().tz_convert('Europe/Athens').strftime('%H:%M')} Athens")
    print(f"Peak wind forecast:   {forecast_df['forecast_wind_MW'].max():.0f} MW")
    print(f"Peak total RES:       {forecast_df['forecast_total_MW'].max():.0f} MW")
    print(f"Daily avg solar:      {forecast_df['forecast_solar_MW'].mean():.0f} MW")
    print(f"Daily avg wind:       {forecast_df['forecast_wind_MW'].mean():.0f} MW")
    print(f"\nForecast complete.")


if __name__ == "__main__":
    main()
