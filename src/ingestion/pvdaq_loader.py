from pathlib import Path
import pandas as pd
class PVDAQLoader:

    def __init__(
        self,
        raw_data_path: Path | str = "data/raw/Alice_Springs_2025.csv",
        output_path: Path | str = "data/raw/actual_generation.parquet",
        plant_capacity_kw: float = 250.0,
        target_column: str = "241_DKA_Totals_PV_Active_Power",
        source_timezone: str = "Australia/Darwin"
    ):
        self.raw_data_path = Path(raw_data_path)            # Make sure the path is a Path object
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents = True, exist_ok = True)
        self.plant_capacity_kw = plant_capacity_kw
        self.target_column = target_column
        self.source_timezone = source_timezone

    def _clean_resample(self, df: pd.DataFrame) -> pd.DataFrame:
        # Converting the local time to UTC for alignment witht he weather data
        parsed_time = pd.to_datetime(df["timestamp"]).dt.tz_localize(self.source_timezone).dt.tz_convert("UTC")
        df["time"] = parsed_time
        df.set_index("time", inplace = True)

        df.rename(columns= {self.target_column: "power_kw"}, inplace = True)
        # converting 5 min data to 1 hr for weather data alignment
        df = df[["power_kw"]].resample("1h").mean()
        # Clipping of negative power at night time
        df["power_kw"] = df["power_kw"].clip(lower=0.0)

        df["capacity_factor"] = (df["power_kw"] / self.plant_capacity_kw).clip(0.0, 1.2)

        return df