"""Feature engineering for delivery ETA prediction."""

from pathlib import Path

import numpy as np
import pandas as pd


NUMERIC_FEATURES = [
    "courier_age",
    "courier_rating",
    "vehicle_condition",
    "multiple_deliveries",
    "distance_km_original",
    "prep_wait_min",
    "hour",
    "weekday",
    "is_weekend",
    "is_peak",
    "is_late_night",
    "is_morning",
    "distance_km",
    "distance_gap_km",
    "zone_demand_3h",
    "zone_orders_that_day",
]
CATEGORICAL_FEATURES = [
    "zone",
    "city_type",
    "order_category",
    "type_of_order",
    "vehicle_type",
    "weather",
    "traffic",
    "festival",
]
TARGET_COL = "delivery_time_min"


def haversine_km(lat1: pd.Series, lon1: pd.Series, lat2: pd.Series, lon2: pd.Series) -> pd.Series:
    earth_radius_km = 6371.0
    lat1_r, lon1_r = np.radians(lat1), np.radians(lon1)
    lat2_r, lon2_r = np.radians(lat2), np.radians(lon2)
    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon / 2.0) ** 2
    return 2.0 * earth_radius_km * np.arcsin(np.sqrt(a.clip(0.0, 1.0)))


def assign_hour_band(hour: pd.Series) -> pd.Series:
    return pd.cut(
        hour,
        bins=[-1, 5, 11, 17, 23],
        labels=["late_night", "morning", "afternoon", "evening"],
        ordered=True,
    )


def normalize_order_time(values: pd.Series) -> pd.Series:
    text = values.astype("string").str.strip()
    parsed = pd.to_datetime(text, format="%H:%M", errors="coerce")
    numeric = pd.to_numeric(values, errors="coerce")
    minutes = (numeric.mod(1) * 24 * 60).round().astype("Int64")
    hours = (minutes // 60).mod(24)
    mins = minutes.mod(60)
    fractional_time = (
        hours.astype("string").str.zfill(2)
        + ":"
        + mins.astype("string").str.zfill(2)
    )
    return parsed.dt.strftime("%H:%M").fillna(fractional_time).fillna("00:00")


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["order_date"] = pd.to_datetime(out["order_date"], errors="coerce")
    order_time = normalize_order_time(out["time_ordered"])
    out["order_dt"] = pd.to_datetime(
        out["order_date"].dt.strftime("%Y-%m-%d") + " " + order_time,
        errors="coerce",
    )
    out["hour"] = out["order_dt"].dt.hour
    out["weekday"] = out["order_dt"].dt.dayofweek
    out["is_weekend"] = out["weekday"].isin([4, 5]).astype(int)
    out["is_peak"] = (out["hour"].between(12, 14) | out["hour"].between(19, 22)).astype(int)
    out["is_late_night"] = (out["hour"] <= 5).astype(int)
    out["is_morning"] = out["hour"].between(6, 11).astype(int)
    out["distance_km"] = haversine_km(
        out["store_lat"], out["store_lng"], out["drop_lat"], out["drop_lng"]
    ).round(3)
    out["distance_gap_km"] = (out["distance_km"] - out["distance_km_original"]).abs().round(3)
    out = out.sort_values(["order_dt", "order_id"], kind="mergesort").reset_index(drop=True)

    demand = np.zeros(len(out), dtype=int)
    for _, indices in out.groupby("zone", sort=False).groups.items():
        idx = np.fromiter(indices, dtype=int)
        times = out.loc[idx, "order_dt"].to_numpy()
        window_start = np.searchsorted(times, times - np.timedelta64(3, "h"), side="left")
        demand[idx] = np.arange(1, len(idx) + 1) - window_start
    out["zone_demand_3h"] = demand
    out["zone_orders_that_day"] = out.groupby(["zone", "order_date"], sort=False).cumcount() + 1
    out["hour_band"] = assign_hour_band(out["hour"])
    return out


def make_model_features(df: pd.DataFrame) -> pd.DataFrame:
    numeric = df[NUMERIC_FEATURES].apply(pd.to_numeric, errors="coerce").astype(float)
    categorical = pd.get_dummies(
        df[CATEGORICAL_FEATURES].astype("string"),
        prefix=CATEGORICAL_FEATURES,
        dtype=int,
    )
    features = pd.concat([numeric, categorical], axis=1)
    return features.replace([np.inf, -np.inf], np.nan)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    raw_path = root / "data" / "raw" / "delivery_orders.csv"
    out_dir = root / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_df = pd.read_csv(raw_path)
    features = engineer_features(raw_df)
    out_path = out_dir / "features_eta.parquet"
    features.to_parquet(out_path, index=False)
    print(f"Engineered {features.shape[1]} columns for {len(features)} rows -> {out_path}")


if __name__ == "__main__":
    main()
