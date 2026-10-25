from pathlib import Path
import requests
import pandas as pd

# Fetching Weather Data From Open-Meteo API
def fetch_historical_forecasts(
    latitude: float,        # Required Field for Latitude
    longitude: float,       # Required Field for Longitude
    start_date: str,
    end_date: str,
    cache_dir: str = "data/raw"

) -> pd.DataFrame:
    base_url = ("https://historical-forecast-api.open-meteo.com/v1/forecast")

    hourly_variables = [
        "shortwave_radiation",
        "direct_normal_irradiance",
        "diffuse_radiation",
        "cloud_cover",
        "temperature_2m",
        "wind_speed_10m",
        "relative_humidity_2m",
    ]

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": hourly_variables,
        "timezone": "UTC",
        "wind_speed_unit": "ms",
        
    }
    cache_path = Path(cache_dir)                #Need to conver the 'str' to path object
    cache_path.mkdir(parents=True, exist_ok= True)       #Check if dir exist if not Create

    #Cache data to avoide repetative API calls

    cache_file = cache_path / f"weather_forcasts_{latitude}_{longitude}_{start_date}_{end_date}.parquet"

    #If data already exist in cache return it in dataframe

    if cache_file.exists():
        print(f"Loading cached weather forecast from {cache_file}")
        return pd.read_parquet(cache_file)


    response = requests.get(base_url, params=params, timeout=10)
    response.raise_for_status()

    data = response.json()          #Responses without the Hourly data raises Error
    if "hourly" not in data:
        raise ValueError(f"Open-Meteo response did not contain 'hourly' data: {data}")
    
    df = pd.DataFrame(data["hourly"])           #Convert the hourly data into a Dataframe

    df["time"] = pd.to_datetime(df["time"], utc= True)      #Having the time column in UTC is critical for uniquely identify the rows
    df.set_index("time", inplace= True)

    #Save fetched data locally for later use and prevent identical API call

    df.to_parquet(cache_file)
    print(f"Cached weather forecast to {cache_file}") 

    return df


# fetch_open_meteo_solar(12.9716, 77.5946)

weather_data = fetch_historical_forecasts(12.9716, 77.5946, "2024-01-01", "2024-01-07")

print(weather_data)
