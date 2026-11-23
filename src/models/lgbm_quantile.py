import pickle
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd


class LGBMQuantileForecaster:
    def __init__(
        self,
        quantiles: tuple[float, float, float] = (0.10, 0.50, 0.90),
        capacity_kw: float = 250.0,
        n_estimators: int = 1000,
        learning_rate: float = 0.03,
        num_leaves: int = 31,
        random_state: int = 42,
    ):

        self.quantiles = quantiles
        self.capacity_kw = capacity_kw
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.random_state = random_state

        self.models: dict[float, lgb.LGBMRegressor] = {}
        self.feature_names: list[str] = []

    def fit(
        self,
        x_train: pd.DataFrame,
        y_train: pd.Series,
        x_val: pd.DataFrame,
        y_val: pd.Series,
        early_stopping_rounds: int = 50,
        verbose: bool = False,
    ) -> "LGBMQuantileForecaster":
        """
        Train independent LightGBM quantile regressors for each specified alpha.
        """
        self.feature_names = list(x_train.columns)

        for alpha in self.quantiles:
            if verbose:
                print(f"Training LightGBM model for quantile alpha={alpha:.2f}...")

            model = lgb.LGBMRegressor(
                objective="quantile",
                alpha=alpha,
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                num_leaves=self.num_leaves,
                random_state=self.random_state,
                importance_type="gain",
                n_jobs=-1,
                verbose=-1,
            )

            callbacks = [
                lgb.early_stopping(
                    stopping_rounds=early_stopping_rounds, verbose=verbose
                ),
                lgb.log_evaluation(period=0),
            ]

            model.fit(
                x_train,
                y_train,
                eval_set=[(x_val, y_val)],
                callbacks=callbacks,
            )

            self.models[alpha] = model
        return self

    def predict(
        self,
        df: pd.DataFrame,
        post_process: bool = True,
    ) -> pd.DataFrame:
        """
        Generate P10, P50, P90 predictions with optional monotonnicity sorting
        and physical boundary/night clamping.
        """
        if not self.models:
            raise RuntimeError("Models have not been fitted yet. Call fit() first.")

        x = df[self.feature_names]

        preds = {}
        for alpha in self.quantiles:
            tag = f"p{int(alpha * 100):02d}"
            preds[tag] = self.models[alpha].predict(x)

        preds_df = pd.DataFrame(preds, index=df.index)

        if not post_process:
            return preds_df

        sorted_matrix = np.sort(preds_df[["p10", "p50", "p90"]].values, axis=1)
        preds_df["p10"] = sorted_matrix[:, 0]
        preds_df["p50"] = sorted_matrix[:, 1]
        preds_df["p90"] = sorted_matrix[:, 2]

        max_capacity = self.capacity_kw * 1.10
        for col in ["p10", "p50", "p90"]:
            preds_df[col] = preds_df[col].clip(lower=0.0, upper=max_capacity)

        if "is_night" in df.columns:
            night_mask = df["is_night"].astype(bool)
        elif "solar_zenith" in df.columns:
            night_mask = df["solar_zenith"] >= 85.0
        else:
            night_mask = pd.Series(False, index=df.index)

        preds_df.loc[night_mask, ["p10", "p50", "p90"]] = 0.0

        return preds_df

    def save(self, output_dir: str = "models") -> None:
        """
        Serialize trained models and metadata to disk.
        """
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)

        bundle = {
            "quantiles": self.quantiles,
            "capacity_kw": self.capacity_kw,
            "feature_name": self.feature_names,
            "models": self.models,
        }
        with open(path / "lgbm_quantile_bundle.pkl", "wb") as f:
            pickle.dump(bundle, f)
        print(f"Model bundle saved to: {path / 'lgbm_quantile_bundle.pkl'}")

    @classmethod
    def load(cls, filepath: str = "models/lgbm_quantile_bundle.pkl") -> "LGBMQuantileForecaster":
        """
        Load serialized model bundle frfom disk.
        """
        with open(filepath, "rb") as f:
            bundle = pickle.load(f)

        instance = cls(
            quantiles=bundle["quantiles"],
            capacity_kw=bundle["capacity_kw"],
        )
        instance.feature_names = bundle["feature_name"]
        instance.models = bundle["models"]
        return instance

if __name__ == "__main__":
    from src.evaluation.metrics import (
        calculate_mpiw,
        calculate_nmae,
        calculate_nrmse,
        calculate_picp,
        calculate_pinball_loss,
        calculate_skill_score,
    )
    from src.models.baselines import PersistenceModel

    print("--- Loading datasets ---")
    train_df = pd.read_parquet("data/processed/train.parquet")
    val_df = pd.read_parquet("data/processed/val.parquet")
    test_df = pd.read_parquet("data/processed/test.parquet")

    target_col = "power_kw"
    excluded_cols = [target_col, "capacity_factor"]
    feature_cols = [c for c in train_df.columns if c not in excluded_cols]

    print(f"Features count: {len(feature_cols)}")
    print(f"Train sample : {len(train_df)} | val: {len(val_df)} | Test: {len(test_df)}")

    x_train, y_train = train_df[feature_cols], train_df[target_col]
    x_val, y_val = val_df[feature_cols], val_df[target_col]

    forecaster = LGBMQuantileForecaster()
    print("\n--- Training LightGBM Quantile Models (P10, P50, P90) ---")
    forecaster.fit(x_train, y_train, x_val, y_val, early_stopping_rounds=50, verbose=True)

    forecaster.save("models")

    preds = forecaster.predict(test_df, post_process=True)
    actuals = test_df[target_col]

    p_preds = PersistenceModel().predict(test_df)
    p_nrmse = calculate_nrmse(actuals, p_preds)

    p50_nmae = calculate_nmae(actuals, preds["p50"])
    p50_nrmse = calculate_nrmse(actuals, preds["p50"])
    skill_score = calculate_skill_score(p50_nrmse, p_nrmse)

    loss_p10 = calculate_pinball_loss(actuals, preds["p10"], alpha=0.10)
    loss_p50 = calculate_pinball_loss(actuals, preds["p50"], alpha=0.50)
    loss_p90 = calculate_pinball_loss(actuals, preds["p90"], alpha=0.90)
    picp = calculate_picp(actuals, preds["p10"], preds["p90"])
    mpiw = calculate_mpiw(preds["p10"], preds["p90"])

    print("\n--- TEST SET PERFORMANCE EVALUATION ---")
    print(f"LightGBM P50 (Median)   : nMAE = {p50_nmae:.2f}% | nRMSE = {p50_nrmse:.2f}%")
    print(f"Skill Score vs. Persist : {skill_score:.2f}%")
    print(f"Pinball Losses          : P10 = {loss_p10:.3f} | P50 = {loss_p50:.3f} | P90 = {loss_p90:.3f}")
    print(f"Prediction Interval     : PICP = {picp:.1f}% (Nominal Target: 80.0%) | MPIW = {mpiw:.2f}%")

    print("-------------------------------------------------------------")

    sample = pd.DataFrame({
        "Actual_KW": actuals,
        "P10": preds["p10"].round(1),
        "P50": preds["p50"].round(1),
        "P90": preds["p90"].round(1),
    }).iloc[12:24]
    print("\n--- Predictions Sample (Daylight Test Slice) ---")
    print(sample)





