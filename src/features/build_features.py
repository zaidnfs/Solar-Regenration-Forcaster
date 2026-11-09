from pathlib import Path

import pandas as pd

from src.features.physics_pvlib import add_physics_features
from src.features.temporal_features import (
    compute_atmospheric_dynamics,
    compute_cyclical_time_features,
    compute_telemetry_lags,
)


def build_master_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    physics_df = add_physics_features(df)
    cyclical_df = compute_cyclical_time_features(df.index)
    dynamics_df = compute_atmospheric_dynamics(physics_df)
    lags_df = compute_telemetry_lags(df["power_kw"])

    master_df = pd.concat([physics_df, cyclical_df, dynamics_df, lags_df], axis=1)
    master_df = master_df.dropna()

    return master_df

def chronological_split(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    # Slice the Dataset
    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]

    if not train_df.index.max() < val_df.index.min():
        raise ValueError("Leakage detected: Train overlaps with validation!")
    if not val_df.index.max() < test_df.index.min():
        raise ValueError("Leakage detected: Validation overlaps with test!")
    if len(train_df) + len(val_df) + len(test_df) != n:
        raise ValueError("Total row count mismatch!")

    return train_df, val_df, test_df

def run_feature_pipeline(
    input_path: str = "data/processed/aligned_telemetry.parquet",
    output_dir: str = "data/processed",
    ) -> None:

    aligned_df = pd.read_parquet(input_path)
    master_df = build_master_feature_matrix(aligned_df)
    train_df, val_df, test_df = chronological_split(master_df)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    train_df.to_parquet(output_path / "train.parquet")
    val_df.to_parquet(output_path / "val.parquet")
    test_df.to_parquet(output_path / "test.parquet")

    print(f"Train_df: {len(train_df)} rows | {train_df.index.min()} -> {train_df.index.max()}")
    print(f"Val_df: {len(val_df)} rows | {val_df.index.min()} -> {val_df.index.max()}")
    print(f"Test_df: {len(test_df)} rows | {test_df.index.min()} -> {test_df.index.max()}")


if __name__ == "__main__":
    run_feature_pipeline()




