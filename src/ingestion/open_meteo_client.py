from urllib3.util import Retry
from pathlib import Path
import requests
from requests.adapters import HTTPAdapter
import pandas as pd
import logging

logger = logging.getLogger(__name__)

class OpenMeteoClient:
    HISTORICAL_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"
    FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

    DEFAULT_VARIABLES = [
        "shortwave_radiation",
        "direct_normal_irradiance",
        "diffuse_radiation",
        "cloud_cover",
        "temperature_2m",
        "wind_speed_10m",
        "relative_humidity_2m",
    ]

    def __init__(self, cache_dir: Path | str = "data/raw", timeout: float = 10.0, max_retries: int = 3, backoff_factor: float = 0.5):
        cache_path = Path(cache_dir)
        cache_path.mkdir(parents = True, exist_ok = True)
        self.cache_dir = cache_path
        self.timeout = timeout
        self.session = requests.Session()
        retry = Retry(
            total = max_retries,
            backoff_factor = backoff_factor,
            status_forcelist = [429, 500, 502, 503, 504],
            allowed_methods = ["GET"],
            raise_on_status = False,
        )
        adapter = HTTPAdapter(max_retries = retry)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

    def _to_dataframe(self, data:dict) -> pd.DataFrame:
        if "hourly" not in data:
            raise ValueError(f"Open-Meteo response did not contain 'hourly' data: {data}")
        df = pd.DataFrame(data["hourly"])
        df["time"] = pd.to_datetime(df["time"], utc = True)
        df.set_index("time", inplace = True)

        return df

    def _get(self, url: str, params: dict) -> dict:
        response = self.session.get(url, params = params, timeout = self.timeout)
        response.raise_for_status()

        return response.json()

    def get_historical_forecast(
        self,
        latitude: float,
        longitude: float,
        start_date: str,
        end_date: str,
        force_refresh: bool = False,
        ) -> pd.DataFrame:

        params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": self.DEFAULT_VARIABLES,
        "timezone": "UTC",
        "wind_speed_unit": "ms",
        }

        cache_file = self.cache_dir / f"weather_forecasts_{latitude}_{longitude}_{start_date}_{end_date}.parquet"
        if cache_file.exists() and not force_refresh:
            logger.info(f"Loading cached weather forecast from {cache_file}")
            return pd.read_parquet(cache_file)

        data = self._get(self.HISTORICAL_URL, params)
        df = self._to_dataframe(data)
        df.to_parquet(cache_file)
        logger.info(f"Cached weather forecast to {cache_file}")

        return df

if __name__ == "__main__":

    logging.basicConfig(level = logging.INFO)

    client = OpenMeteoClient()
    df = client.get_historical_forecast(
            latitude = 26.850000,
            longitude = 80.949997,
            start_date = "2026-01-01",
            end_date = "2026-01-07"
        )
    print(df.head)
