import csv
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from src.features import assign_hour_band, engineer_features, haversine_km, make_model_features


ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_SHA256 = "e33e182ebb9c1d7e0f54c83c0cef6ddc5aa98d203e8888d33ab68cca3a079cba"


def test_original_kaggle_file_is_verbatim() -> None:
    path = ROOT / "data" / "raw" / "Zomato Dataset.csv"
    assert path.stat().st_size == 6_371_825
    assert hashlib.sha256(path.read_bytes()).hexdigest() == ORIGINAL_SHA256
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        rows = sum(1 for _ in reader)
    assert len(header) == 20
    assert rows == 45_584


def test_adapted_data_and_features() -> None:
    raw = pd.read_csv(ROOT / "data" / "raw" / "delivery_orders.csv")
    assert len(raw) == 38_964
    features = engineer_features(raw)
    model_features = make_model_features(features)
    assert len(features) == 38_964
    assert features["order_dt"].notna().all()
    assert features["distance_gap_km"].max() < 0.01
    assert "delivery_time_min" not in model_features.columns
    assert "sla_miss" not in model_features.columns
    assert "order_id" not in model_features.columns


def test_haversine_known_distance() -> None:
    distance = haversine_km(
        pd.Series([0.0]), pd.Series([0.0]), pd.Series([1.0]), pd.Series([0.0])
    )
    assert np.isclose(distance.iloc[0], 111.195, atol=0.01)


def test_hour_band_boundaries() -> None:
    bands = assign_hour_band(pd.Series([0, 5, 6, 11, 12, 17, 18, 23])).astype(str)
    assert bands.tolist() == [
        "late_night",
        "late_night",
        "morning",
        "morning",
        "afternoon",
        "afternoon",
        "evening",
        "evening",
    ]
