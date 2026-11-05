from open_meteo_client import OpenMeteoClient
from pvdaq_loader import PVDAQLoader
import pandas as pd
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def align_weather_and_actuals(weather_df: pd.DataFrame, actuals_df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(weather_df.index, pd.DatetimeIndex) or not isinstance(actuals_df.index, pd.DatetimeIndex):
        raise ValueError("Both DataFrames must be indexed by pd.DatetimeIndex.")
    if not (str(weather_df.index.tz) == "UTC") or not (str(actuals_df.index.tz) == "UTC"):
        raise ValueError("Both DataFrames indices must be 'UTC' format.")

    aligned_df = weather_df.join(actuals_df, how = "inner")

    if aligned_df.empty:
        raise ValueError("Alignment resulted in an empty DataFrame. Ensure data ranges and timezones overlap.")
    if not aligned_df.index.is_monotonic_increasing:
        raise ValueError("Aligned Dataframe index is not monotonically oncreasing in time")
    if aligned_df.index.has_duplicates:
        raise ValueError("Aligned DataFrame contains duplicate timestamps.")
    if aligned_df.isna().any().any():
        aligned_df = aligned_df.dropna()

    return aligned_df

def build_aligned_dataset(output_path: Path | str = "data/processed/aligned_telemetry.parquet", force_refresh: bool = False) -> pd.DataFrame:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents = True, exist_ok = True)

    if output_path.exists() and not force_refresh:
        logger.info(f"Loading cached aligned data from {output_path}")
        return pd.read_parquet(output_path)
    
    # Fetch Actual solar telemetry
    loader = PVDAQLoader()
    actuals_df = loader.load()

    # Fetch hystorical weather data
    weather_client = OpenMeteoClient()
    weather_df = weather_client.get_historical_forecast(
        latitude = -23.76,
        longitude = 133.87,
        start_date = "2025-01-01",
        end_date="2025-12-31"
    )

    # Align the data fetched earlier
    aligned_df = align_weather_and_actuals(weather_df, actuals_df)
    aligned_df.to_parquet(output_path)      # Saved the Processed Data
    logger.info(f"Successfully aligned and saved {len(aligned_df)} hourly records to {output_path}")

    return aligned_df


if __name__ == "__main__":
    logging.basicConfig(level = logging.INFO)
    df = build_aligned_dataset()
    print(df.head())
    print(df.info())
