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