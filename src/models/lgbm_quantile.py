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
        self.feature_name: list[str] = []

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
        self.feature_name = list(x_train.columns)

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

            self.modedls[alpha] = model
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
