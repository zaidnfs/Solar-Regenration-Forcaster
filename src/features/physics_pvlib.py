import logging
import pandas as pd
import numpy as np
import logging
from pvlib import solarposition, atmosphere, location, irradiance

logger = logging.getLogger(__name__)

DEFAULT_LATITUDE = -23.76
DEFAULT_LONGITUDE = 133.87
DEFAULT_ALTITUDE = 558.0
DEFAULT_TILT = 20.0
DEFAULT_AZIMUTH = 0.0

def calculate_effective_solar_timestamps(ts_index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """ 
    We are subtracting 30 min from teh open-meteo weather data cause the hourly radiation is
    backward-integrated hourly average.
    - A timestamp labeled 14:00:00 does not mean "the sun right now at 14:00".
    - It means the average irradiance integrated over the previous hour: 13:00:00 to 14:00:00.
    - The effective physical "center of mass" of that hourly energy is at the window midpoint: 13:30:00 (t - 30 min).
    """
    
    if not isinstance(ts_index, pd.DatetimeIndex):
        raise TypeError(f"Expected 'ts_index' to be pd.DatetimeIndex, but recived {type(ts_index).__name__}")
    if ts_index.tz is None or not str(ts_index.tz) == "UTC":
        raise TypeError(f"Expected 'ts_index' to be timezone-aware UTC, but recived tz={type(ts_index).__name__}.")

    shifted_index = ts_index - pd.Timedelta(minutes=30)

    return shifted_index

def compute_solar_position(
    ts_index: pd.DatetimeIndex,
    latitude: float = DEFAULT_LATITUDE,
    longitude: float = DEFAULT_LONGITUDE,
    altitude: float = DEFAULT_ALTITUDE,
    ) -> pd.DataFrame:

    shifted_ts = calculate_effective_solar_timestamps(ts_index)

    solpos = solarposition.get_solarposition(
        time = shifted_ts,
        latitude = latitude,
        longitude = longitude,
        altitude = altitude,
    )

    airmass = atmosphere.get_relative_airmass(solpos["apparent_zenith"])
    airmass = airmass.fillna(0.0)       #At night no sunlight, Thus value = NaN, Thus NaN switched to 0.0

    is_night = solpos["zenith"] >= 85.0     # If the angel of the sun from the perpendicular line at the location is greater that 85* that means its sunset/night time.

    df = pd.DataFrame(
            {
                "solar_zenith": solpos["zenith"].values,
                "solar_azimuth": solpos["azimuth"].values,
                "solar_elevation": solpos["elevation"].values,
                "airmass": airmass.values,
                "is_night": is_night.values,
            },
            index = ts_index,
        )
    return df


def compute_clearsky_irradiance(
    ts_index: pd.DatetimeIndex,
    latitude: float = DEFAULT_LATITUDE,
    longitude: float = DEFAULT_LONGITUDE,
    altitude: float = DEFAULT_ALTITUDE,
    ) -> pd.DataFrame:

    shifted_ts = calculate_effective_solar_timestamps(ts_index)

    site = location.Location(latitude, longitude, altitude=altitude)
    cs = site.get_clearsky(shifted_ts, model='ineichen')

    df = pd.DataFrame(
        {
            "ghi_clear": cs["ghi"].values,
            "dni_clear": cs["dni"].values,
            "dhi_clear": cs["dhi"].values,
        },
        index=ts_index,
    )

    return df

def compute_clearsky_index(
    ghi_forecast: pd.Series,
    ghi_clear: pd.Series,
    eps: float = 1e-4
    ) -> pd.Series:

    clearsky_index = (ghi_forecast / (ghi_clear + eps)).clip(0.0, 1.5)
    clearsky_index = clearsky_index.where(ghi_clear >= 10.0, 0.0)
    clearsky_index.name = "clearsky_index"

    return clearsky_index

def compute_poa_irradiance(
    surface_tilt: float,
    surface_azimuth: float,
    solar_zenith: pd.Series,
    solar_azimuth: pd.Series,
    dni: pd.Series,
    ghi: pd.Series,
    dhi: pd.Series,
    model: str = "haydavies",
    ) -> pd.DataFrame:
    
    dni_extra = irradiance.get_extra_radiation(solar_zenith.index)

    poa = irradiance.get_total_irradiance(
            surface_tilt=surface_tilt,
            surface_azimuth=surface_azimuth,
            solar_zenith=solar_zenith,
            solar_azimuth=solar_azimuth,
            dni_extra=dni_extra,
            dni=dni,
            ghi=ghi,
            dhi=dhi,
            model=model,   
        )

    df = pd.DataFrame(
        {
            "poa_global": poa["poa_global"].values,
            "poa_direct": poa["poa_direct"].values,
            "poa_diffuse": poa["poa_diffuse"].values,
        },
        index=solar_zenith.index,
    )

    return df

if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    aligned_df = pd.read_parquet("data/processed/aligned_telemetry.parquet")
    sample = aligned_df.iloc[0:12]
    
    test_index = pd.date_range("2025-01-01", periods = 5, freq="1h", tz="UTC")
    solar_df = compute_solar_position(sample.index)
    clearsky_df = compute_clearsky_irradiance(sample.index)
    
    kc = compute_clearsky_index(
        ghi_forecast=sample["shortwave_radiation"],
        ghi_clear=clearsky_df["ghi_clear"]
    )
    
    poa_df = compute_poa_irradiance(
        surface_tilt=DEFAULT_TILT,
        surface_azimuth=DEFAULT_AZIMUTH,
        solar_zenith=solar_df["solar_zenith"],
        solar_azimuth=solar_df["solar_azimuth"],
        dni=sample["direct_normal_irradiance"],
        ghi=sample["shortwave_radiation"],
        dhi=sample["diffuse_radiation"],
    )

    test_results = pd.DataFrame(
        {
            "GHI_forecast":sample["shortwave_radiation"],
            "GHI_clear":clearsky_df["ghi_clear"].round(1),
            "k_c":kc.round(2),
            "POA_global":poa_df["poa_global"].round(1),
            "Actual_kw":sample["power_kw"].round(1),
        }
    )
    # print("--- Solar Position ---")
    # print(solar_df)
    # print("--- Clear-Sky Irradiance ---")
    # print(clearsky_df)
    print("\n--- Physics-Informed Feature validation (Day 1)--- ")
    print(test_results)