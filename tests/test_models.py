import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.features import make_model_features
from src.portable_model import PortableLightGBM


ROOT = Path(__file__).resolve().parents[1]


def test_chronological_metrics_and_winner() -> None:
    metrics = json.loads((ROOT / "models" / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["split"]["method"] == "chronological"
    assert metrics["split"]["train_rows"] == 25_153
    assert metrics["split"]["test_rows"] == 13_811
    assert metrics["winner"] == "lightgbm"
    assert metrics["lightgbm"]["mae_min"] <= metrics["xgboost"]["mae_min"]


def test_saved_lightgbm_model_predicts() -> None:
    metrics = json.loads((ROOT / "models" / "metrics.json").read_text(encoding="utf-8"))
    test_df = pd.read_parquet(ROOT / "data" / "processed" / "test_predictions.parquet")
    model = lgb.Booster(model_file=str(ROOT / "models" / "eta_lgbm_model.txt"))
    x_sample = make_model_features(test_df.head(25)).reindex(columns=metrics["feature_cols"], fill_value=0)
    predictions = model.predict(x_sample)
    assert len(predictions) == 25
    assert predictions.min() > 0
    assert "eta_xgb_model.json" in metrics["model_files"].values()


def test_portable_lightgbm_matches_native_predictions() -> None:
    metrics = json.loads((ROOT / "models" / "metrics.json").read_text(encoding="utf-8"))
    test_df = pd.read_parquet(ROOT / "data" / "processed" / "test_predictions.parquet")
    x_sample = make_model_features(test_df.head(100)).reindex(columns=metrics["feature_cols"], fill_value=0)
    native_model = lgb.Booster(model_file=str(ROOT / "models" / metrics["model_files"]["lightgbm"]))
    portable_model = PortableLightGBM(ROOT / "models" / metrics["model_files"]["lightgbm_portable"])

    np.testing.assert_allclose(portable_model.predict(x_sample), native_model.predict(x_sample), rtol=0, atol=1e-12)

    x_missing = x_sample.copy()
    x_missing["distance_km"] = np.nan
    x_missing["courier_age"] = np.nan
    np.testing.assert_allclose(portable_model.predict(x_missing), native_model.predict(x_missing), rtol=0, atol=1e-12)
