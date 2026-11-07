import pandas as pd
import numpy as np
import logging
from pvlib import solarposition, atmosphere

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

if __name__ == "__main__":
    test_index = pd.date_range("2025-01-01", periods = 5, freq="1h", tz="UTC")
    solar_df = compute_solar_position(test_index)
    print(solar_df)