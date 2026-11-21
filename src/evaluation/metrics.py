import numpy as np
import pandas as pd


def calculate_nmae(
    actual: pd.Series,
    predicted: pd.Series,
    capacity_kw: float = 250.0,
) -> float:
    """
    Normalized Mean Absolute Error (nMAE) as a percentage of plant capacity.
    """
    mae = np.mean(np.abs(actual - predicted))

    return float((mae / capacity_kw) * 100.0)


def calculate_nrmse(
    actual: pd.Series,
    predicted: pd.Series,
    capacity_kw: float = 250.0,
) -> float:
    """
    Normalized Root Mean Squared Error (nRMSE) as a percentage of plant capacity.
    """
    rmse = np.sqrt(np.mean(np.square(actual - predicted)))

    return float((rmse / capacity_kw) * 100.0)


def calculate_skill_score(
    rmse_model: float,
    rsme_baseline: float,
) -> float:
    """
    Forecase Skill Score(%) relative to a baseline (e.g. Persistence).
    """
    return float((1.0 - (rmse_model / rsme_baseline)) * 100.0)


def calculate_pinball_loss(
    actual: pd.Series,
    predicted: pd.Series,
    alpha: float,
) -> float:
    """Pinball loss for quantile alpha"""
    diff = actual - predicted
    loss = np.maximum(alpha * diff, (alpha - 1.0) * diff)

    return float(np.mean(loss))


def calculate_picp(
    actual: pd.Series,
    lower_p10: pd.Series,
    upper_p90: pd.Series,
) -> float:
    """
    Pridiction Interval Coverage Probability (PICP) for nominal 80% interval [P10, P90].
    Returns coverage as a percentage (0 to 100%)
    """
    inside = (actual >= lower_p10) & (actual <= upper_p90)

    return float(np.mean(inside) * 100.0)


def calculate_mpiw(
    lower_p10: pd.Series,
    upper_p90: pd.Series,
    capacity_kw: float = 250.0,
) -> float:
    """
    Normalized Mean prediction Interval Width (MPIW) as a percentage of plant capacity.
    """
    width = upper_p90 - lower_p10

    return float((np.mean(width) / capacity_kw) * 100.0)


if __name__ == "__main__":
    from src.models.baselines import PersistenceModel, PhysicalClearSkyBaseline

    test_df = pd.read_parquet("data/processed/test.parquet")
    actuals = test_df["power_kw"]

    # Evaluate Persistence
    p_model = PersistenceModel()
    p_preds = p_model.predict(test_df)
    p_nmae = calculate_nmae(actuals, p_preds)
    p_nrmse = calculate_nrmse(actuals, p_preds)

    # Evaluate Physical baseline
    c_model = PhysicalClearSkyBaseline()
    c_preds = c_model.predict(test_df)
    c_nmae = calculate_nmae(actuals, c_preds)
    c_nrmse = calculate_nrmse(actuals, c_preds)

    skill_score = calculate_skill_score(c_nrmse, p_nrmse)

    print(f"Persistence baseline : nMAE = {p_nmae:.2f}% | nRMSE = {p_nrmse:.2f}%")
    print(f"Physical baseline : nMAE = {c_nmae:.2f}% | nRMSE = {c_nrmse:.2f}%")
    print(f"Physical Skill Score : {skill_score:+.2f}% vs. Persistence")
