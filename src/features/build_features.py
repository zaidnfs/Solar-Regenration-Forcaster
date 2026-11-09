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