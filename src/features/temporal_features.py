import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# The time is in 24 Hour Format which creates a big change for the ML model
# on interpreting the change from 23:00 to 00:00(Making it a 23 hr difference)
# Same goes for the 365 days ina year(31 Dec - 1 jan)
# thus we create a cyclical time feature using Sin and Cosin ~~ with gives
# each hour a unique coordinate at equal intervals.
def compute_cyclical_time_features(ts_index: pd.DatetimeIndex) -> pd.DataFrame:

    if not isinstance(ts_index, pd.DatetimeIndex):
        raise TypeError(f"Expected 'ts_index' to be pd.DatetimeIndex, instead recived {type(ts_index)}.")

    hours = ts_index.hour
    day_of_year = ts_index.day_of_year

    hour_sin = np.sin(2.0 * np.pi * hours / 24.0)
    hour_cos = np.cos(2.0 * np.pi * hours / 24.0)
    doy_sin = np.sin(2.0 * np.pi * day_of_year / 365.25)
    doy_cos = np.cos(2.0 * np.pi * day_of_year / 365.25)

    df = pd.DataFrame(
        {
            "hour_sin": hour_sin,
            "hour_cos": hour_cos,
            "doy_sin": doy_sin,
            "doy_cos": doy_cos,
        },
        index=ts_index,
    )

    return df

# The rate of change in cloud volatility and POA.
def compute_atmospheric_dynamics(df: pd.DataFrame) -> pd.DataFrame:
    cloud_3h_mean = df["cloud_cover"].rolling(window=3, min_periods=1).mean()
    cloud_3h_std = df["cloud_cover"].rolling(window=3, min_periods=1).std().fillna(0.0)
    
    cloud_delta_1h = df["cloud_cover"].diff(1).fillna(0.0)

    poa_ramp_1h = df["poa_global"].diff(1).fillna(0.0)
    poa_ramp_3h = df["poa_global"].diff(3).fillna(0.0)

    df = pd.DataFrame(
        {
            "cloud_3h_mean": cloud_3h_mean.values,
            "cloud_3h_std": cloud_3h_std.values,
            "cloud_delta_1h": cloud_delta_1h.values,
            "poa_ramp_1h": poa_ramp_1h.values,
            "poa_ramp_3h": poa_ramp_3h.values,
        },
        index=df.index,
    )

    return df

# Computing the previous days generation to predict the future generation at the exact same moment
# helps monitoring the health of the overall solar plant
def compute_telemetry_lags(
    actual_power: pd.Series,
    lags: tuple[int] = (24, 48, 72),
    ) -> pd.DataFrame:

    lag_24h = actual_power.shift(24)
    lag_48h = actual_power.shift(48)
    lag_72h = actual_power.shift(72)

    same_hour_3d_mean = pd.concat([lag_24h, lag_48h, lag_72h], axis=1).mean(axis=1)

    df = pd.DataFrame(
        {
            "lag_power_24h": lag_24h.values,
            "lag_power_48h": lag_48h.values,
            "lag_power_72h": lag_72h.values,
            "same_hour_3d_mean": same_hour_3d_mean.values,
        },
        index=actual_power.index,
    )

    return df

if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    from src.features.physics_pvlib import add_physics_features
    # test_dates = pd.date_range("2025-01-01 22:00", periods=4, freq="1h", tz="UTC")
    # df_cyclical = compute_cyclical_time_features(test_dates)
    # print(df_cyclical)
    df = pd.read_parquet("data/processed/aligned_telemetry.parquet").iloc[0:120]
    enriched = add_physics_features(df)
    cyclical = compute_cyclical_time_features(enriched.index)
    lags = compute_telemetry_lags(enriched["power_kw"])
    # dyn = compute_atmospheric_dynamics(enriched)
    print(" ---- Telemetry Lags ----")
    sample_view = pd.DataFrame(
        {
            "Actual_kW": enriched["power_kw"].iloc[72:84],
            "Lag_24h": lags["lag_power_24h"].iloc[72:84].round(1),
            "Lag_48h": lags["lag_power_48h"].iloc[72:84].round(1),
            "Lag_72h": lags["lag_power_72h"].iloc[72:84].round(1),
            "3d_mean": lags["same_hour_3d_mean"].iloc[72:84].round(1),
        }
    )
    print(sample_view)

    