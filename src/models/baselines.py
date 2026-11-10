import logging

import pandas as pd

logger = logging.getLogger(__name__)

class PersistenceModel:
    def __init__(self, nameplate_capacity_kw: float = 250.0):
        self.nameplate_capacity_kw = nameplate_capacity_kw

    def predict(self, df: pd.DataFrame) -> pd.Series:
        preds = df["lag_power_24h"].copy().fillna(df["lag_power_48h"]).fillna(0.0)
        preds = preds.mask(df["is_night"], 0.0)
        preds = preds.clip(lower = 0.0, upper = 1.1 * self.nameplate_capacity_kw)

        preds.name = "persistence_pred"

        return preds

class PhysicalClearSkyBaseline:
    def __init__(
        self,
        nameplate_capacity_kw: float = 250.0,
        system_degrade: float = 0.80,
        ):
        self.nameplate_capacity_kw = nameplate_capacity_kw
        self.system_degrade = system_degrade

    def predict(self, df: pd.DataFrame) -> pd.Series:
        power = self.nameplate_capacity_kw * (df["poa_global"] / 1000.0 * df["thermal_derating"] * self.system_degrade)

        preds = power.mask(df["is_night"], 0.0)
        preds = preds.clip(lower = 0.0, upper = 1.1 * self.nameplate_capacity_kw)

        preds.name = "physical_preds"

        return preds

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    test_df = pd.read_parquet("data/processed/test.parquet")

    persistence = PersistenceModel()
    physical = PhysicalClearSkyBaseline()

    p_preds = persistence.predict(test_df)
    c_preds = physical.predict(test_df)

    sample_comparison = pd.DataFrame({
        "Actual_kW": test_df["power_kw"].iloc[10:22].round(1),
        "Persistence_kW": p_preds.iloc[10:22].round(1),
        "Physical_kW": c_preds.iloc[10:22].round(1),
    })

    print("\n --- Baseline Prediction vs. Actual (Test Set Sample) ---")
    print(sample_comparison)