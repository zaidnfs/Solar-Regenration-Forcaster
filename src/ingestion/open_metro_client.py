from requests import request
import requests
import pandas as pd

# Fetching Weather Data From Open-Meteo API
def fetch_open_meteo_solar(
    latitude: float,        # Required Field for Latitude
    longitude: float        # Required Field for Longitude
) -> pd.DataFrame:
    base_url = ("https://api.open-meteo.com/v1/forecast")
    params = {
        "latitude": latitude,
        "longitude": longitude,    
    }
    
    response = requests.get(base_url, params=params, timeout=10)
    return response.json()


# fetch_open_meteo_solar(12.9716, 77.5946)

weather_data = fetch_open_meteo_solar(12.9716, 77.5946)
print(weather_data)
