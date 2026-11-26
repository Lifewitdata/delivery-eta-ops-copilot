"""Train chronological ETA baselines, XGBoost, and LightGBM."""

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.features import CATEGORICAL_FEATURES, NUMERIC_FEATURES, TARGET_COL, engineer_features, make_model_features


ROOT = Path(__file__).resolve().parents[1]
SPLIT_DATE = pd.Timestamp("2022-03-22")
SLA_THRESHOLD_MIN = 30.0


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray, label: str) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    actual_late = y_true > SLA_THRESHOLD_MIN
    predicted_late = y_pred > SLA_THRESHOLD_MIN
    return {
        "model": label,
        "mae_min": round(float(mean_absolute_error(y_true, y_pred)), 2),
        "rmse_min": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 2),
        "r2": round(float(r2_score(y_true, y_pred)), 3),
        "actual_sla_miss_rate": round(float(actual_late.mean()), 4),
        "predicted_sla_miss_rate": round(float(predicted_late.mean()), 4),
        "sla_accuracy": round(float((actual_late == predicted_late).mean()), 4),
        "missed_late_count": int((actual_late & ~predicted_late).sum()),
        "false_alarm_count": int((~actual_late & predicted_late).sum()),
        "mean_error_min": round(float((y_true - y_pred).mean()), 2),
    }


def build_defaults(train_df: pd.DataFrame, x_train: pd.DataFrame) -> dict:
    defaults = {}
    for col in NUMERIC_FEATURES:
        median = x_train[col].median()
        defaults[col] = float(median) if pd.notna(median) else 0.0
    for col in x_train.columns:
        if col not in NUMERIC_FEATURES:
            defaults[col] = 0
    for col in CATEGORICAL_FEATURES:
        mode = train_df[col].mode(dropna=True)
        if not mode.empty:
            dummy = f"{col}_{mode.iloc[0]}"
            if dummy in defaults:
                defaults[dummy] = 1
    return defaults


def train_models(df: pd.DataFrame, root: Path = ROOT) -> tuple[dict, pd.DataFrame]:
    df = df.sort_values(["order_dt", "order_id"], kind="mergesort").reset_index(drop=True)
    features = make_model_features(df)
    feature_cols = list(features.columns)
    train_mask = df["order_dt"] < SPLIT_DATE

    train_df = df.loc[train_mask].copy()
    test_df = df.loc[~train_mask].copy()
    x_train = features.loc[train_mask].copy()
    x_test = features.loc[~train_mask].copy()
    y_train = train_df[TARGET_COL].astype(float)
    y_test = test_df[TARGET_COL].astype(float)

    overall_median = float(y_train.median())
    zone_medians = train_df.groupby("zone")[TARGET_COL].median()
    baseline_overall_pred = np.repeat(overall_median, len(test_df))
    baseline_zone_pred = test_df["zone"].map(zone_medians).fillna(overall_median).to_numpy()
    baseline_overall = evaluate_predictions(y_test, baseline_overall_pred, "Naive overall median")
    baseline_zone = evaluate_predictions(y_test, baseline_zone_pred, "Naive zone median")

    xgb_model = xgb.XGBRegressor(
        n_estimators=350,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        min_child_weight=3,
        reg_lambda=1.0,
        n_jobs=-1,
        random_state=42,
    )
    xgb_model.fit(x_train, y_train)
    xgb_pred = xgb_model.predict(x_test)

    lgb_model = lgb.LGBMRegressor(
        n_estimators=450,
        num_leaves=31,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        min_child_samples=20,
        reg_lambda=1.0,
        n_jobs=-1,
        random_state=42,
        verbosity=-1,
    )
    lgb_model.fit(x_train, y_train)
    lgb_pred = lgb_model.predict(x_test)

    xgb_metrics = evaluate_predictions(y_test, xgb_pred, "XGBoost")
    lgb_metrics = evaluate_predictions(y_test, lgb_pred, "LightGBM")
    winner = "lightgbm" if mean_absolute_error(y_test, lgb_pred) <= mean_absolute_error(y_test, xgb_pred) else "xgboost"

    models_dir = root / "models"
    processed_dir = root / "data" / "processed"
    models_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    xgb_model.save_model(models_dir / "eta_xgb_model.json")
    lgb_model.booster_.save_model(str(models_dir / "eta_lgbm_model.txt"))

    winner_pred = lgb_pred if winner == "lightgbm" else xgb_pred
    test_out = test_df.copy()
    test_out["predicted_xgb_eta_min"] = np.round(xgb_pred, 2)
    test_out["predicted_lgbm_eta_min"] = np.round(lgb_pred, 2)
    test_out["predicted_eta_min"] = np.round(winner_pred, 2)
    test_out["abs_error_min"] = np.abs(test_out[TARGET_COL] - test_out["predicted_eta_min"]).round(2)
    test_out["actual_sla_miss"] = (test_out[TARGET_COL] > SLA_THRESHOLD_MIN).astype(int)
    test_out["predicted_sla_miss"] = (winner_pred > SLA_THRESHOLD_MIN).astype(int)
    test_out.to_parquet(processed_dir / "test_predictions.parquet", index=False)
    test_out.loc[test_out["actual_sla_miss"] != test_out["predicted_sla_miss"]].to_csv(
        processed_dir / "sla_miss_log.csv", index=False
    )

    metrics = {
        "split": {
            "method": "chronological",
            "split_date": SPLIT_DATE.date().isoformat(),
            "train_rows": int(train_mask.sum()),
            "test_rows": int((~train_mask).sum()),
            "train_start": train_df["order_dt"].min().isoformat(),
            "train_end": train_df["order_dt"].max().isoformat(),
            "test_start": test_df["order_dt"].min().isoformat(),
            "test_end": test_df["order_dt"].max().isoformat(),
        },
        "sla_threshold_min": SLA_THRESHOLD_MIN,
        "baseline": baseline_overall,
        "baseline_zone": baseline_zone,
        "xgboost": xgb_metrics,
        "lightgbm": lgb_metrics,
        "winner": winner,
        "feature_cols": feature_cols,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "category_levels": {
            col: sorted(train_df[col].dropna().astype(str).unique().tolist()) for col in CATEGORICAL_FEATURES
        },
        "defaults": build_defaults(train_df, x_train),
        "model_files": {
            "xgboost": "eta_xgb_model.json",
            "lightgbm": "eta_lgbm_model.txt",
        },
    }
    (models_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics, test_out


def main() -> None:
    features_path = ROOT / "data" / "processed" / "features_eta.parquet"
    if features_path.exists():
        df = pd.read_parquet(features_path)
    else:
        df = engineer_features(pd.read_csv(ROOT / "data" / "raw" / "delivery_orders.csv"))
    metrics, _ = train_models(df)
    print(json.dumps({k: metrics[k] for k in ["split", "baseline", "xgboost", "lightgbm", "winner"]}, indent=2))


if __name__ == "__main__":
    main()
