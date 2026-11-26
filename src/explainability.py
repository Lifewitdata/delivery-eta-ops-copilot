"""Compute global, zone, and hour-band SHAP summaries for the ETA winner."""

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import shap

from src.features import make_model_features


ROOT = Path(__file__).resolve().parents[1]
HOUR_BANDS = ["late_night", "morning", "afternoon", "evening"]


def rank_drivers(values: np.ndarray, feature_cols: list[str], limit: int = 10) -> list[dict]:
    importance = np.abs(values).mean(axis=0)
    ranked = sorted(
        (
            {"feature": col, "mean_abs_shap_min": round(float(value), 3)}
            for col, value in zip(feature_cols, importance)
        ),
        key=lambda item: item["mean_abs_shap_min"],
        reverse=True,
    )
    return ranked[:limit]


def build_summary(metrics_data: dict, test_df: pd.DataFrame, sample_size: int = 2000) -> dict:
    feature_cols = metrics_data["feature_cols"]
    winner = metrics_data["winner"]
    if winner == "lightgbm":
        model = lgb.Booster(model_file=str(ROOT / "models" / metrics_data["model_files"]["lightgbm"]))
    else:
        import xgboost as xgb

        model = xgb.XGBRegressor()
        model.load_model(ROOT / "models" / metrics_data["model_files"]["xgboost"])

    sample_df = test_df.sample(n=min(sample_size, len(test_df)), random_state=42).sort_index()
    x_sample = make_model_features(sample_df)[feature_cols]
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(x_sample)
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    shap_values = np.asarray(shap_values)

    zone_summary = (
        test_df.assign(
            missed_late=((test_df["actual_sla_miss"] == 1) & (test_df["predicted_sla_miss"] == 0)).astype(int)
        )
        .groupby("zone")
        .agg(
            orders=("order_id", "count"),
            avg_actual_eta=("delivery_time_min", "mean"),
            mae_min=("abs_error_min", "mean"),
            sla_miss_rate=("actual_sla_miss", "mean"),
            missed_lates=("missed_late", "sum"),
        )
        .round({"avg_actual_eta": 2, "mae_min": 2, "sla_miss_rate": 4})
        .reset_index()
        .to_dict(orient="records")
    )

    bands = sample_df["hour_band"].astype(str)
    hour_band_summary = []
    for band in HOUR_BANDS:
        mask = bands.eq(band).to_numpy()
        if not mask.any():
            continue
        band_df = sample_df.loc[mask]
        band_values = shap_values[mask]
        hour_band_summary.append(
            {
                "hour_band": band,
                "orders": int(mask.sum()),
                "avg_actual_eta": round(float(band_df["delivery_time_min"].mean()), 2),
                "mae_min": round(float(band_df["abs_error_min"].mean()), 2),
                "sla_miss_rate": round(float(band_df["actual_sla_miss"].mean()), 4),
                "top_shap_drivers": rank_drivers(band_values, feature_cols, limit=5),
            }
        )

    late_night = test_df[test_df["hour_band"].astype(str) == "late_night"]
    return {
        "model": winner,
        "metrics": metrics_data[winner],
        "baseline": metrics_data["baseline"],
        "sample_size": int(len(sample_df)),
        "top_shap_drivers": rank_drivers(shap_values, feature_cols),
        "zone_summary": zone_summary,
        "hour_band_summary": hour_band_summary,
        "late_night_sla_miss_rate": round(float(late_night["actual_sla_miss"].mean()), 4) if len(late_night) else 0.0,
    }


def main() -> None:
    models_dir = ROOT / "models"
    metrics_data = json.loads((models_dir / "metrics.json").read_text(encoding="utf-8"))
    test_df = pd.read_parquet(ROOT / "data" / "processed" / "test_predictions.parquet")
    summary = build_summary(metrics_data, test_df)
    out_path = models_dir / "shap_summary.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved SHAP summary for {summary['model']} -> {out_path}")


if __name__ == "__main__":
    main()
