import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


class PVDAQLoader:
    def __init__(
        self,
        raw_data_path: Path | str = "data/raw/Alice_Springs_2025.csv",
        output_path: Path | str = "data/raw/actual_generation.parquet",
        plant_capacity_kw: float = 250.0,
        target_column: str = "241_DKA_Totals_PV_Active_Power",
        source_timezone: str = "Australia/Darwin",
    ):
        self.raw_data_path = Path(raw_data_path)  # Make sure the path is a Path object
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.plant_capacity_kw = plant_capacity_kw
        self.target_column = target_column
        self.source_timezone = source_timezone

    def _clean_resample(self, df: pd.DataFrame) -> pd.DataFrame:
        # Converting the local time to UTC for alignment witht he weather data
        parsed_time = (
            pd.to_datetime(df["timestamp"])
            .dt.tz_localize(self.source_timezone)
            .dt.tz_convert("UTC")
        )
        df["time"] = parsed_time
        df.set_index("time")

        df.rename(columns={self.target_column: "power_kw"})
        # converting 5 min data to 1 hr for weather data alignment
        df = df[["power_kw"]].resample("1h").mean()
        # Clipping of negative power at night time
        df["power_kw"] = df["power_kw"].clip(lower=0.0)

        df["capacity_factor"] = (df["power_kw"] / self.plant_capacity_kw).clip(0.0, 1.2)

        return df

    def load(self, force_refresh: bool = False) -> pd.DataFrame:
        if self.output_path.exists() and not force_refresh:
            logger.info(f"Reading data from cache...{self.output_path}")
            data = pd.read_parquet(self.output_path)
            return data
        if not self.raw_data_path.exists():
            raise FileNotFoundError(
                f"Raw telemetry file not found at:{self.raw_data_path}"
            )
        df = pd.read_csv(self.raw_data_path, usecols=["timestamp", self.target_column])
        df = self._clean_resample(df)
        df.to_parquet(self.output_path)
        logger.info(
            f"Successfully processed and cached generation data to {self.output_path}"
        )

        return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    loader = PVDAQLoader()
    df = loader.load()
    print(df.head())
    print(df.info())
    print(df.describe())
    print(df[df["power_kw"] > 0].head())
