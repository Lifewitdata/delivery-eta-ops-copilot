"""Run exported LightGBM trees without the native LightGBM package."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def compact_tree(node: dict[str, Any]) -> dict[str, Any]:
    if "leaf_value" in node:
        return {"leaf_value": node["leaf_value"]}
    return {
        "split_feature": node["split_feature"],
        "threshold": node["threshold"],
        "decision_type": node.get("decision_type", "<="),
        "default_left": node.get("default_left", True),
        "left": compact_tree(node["left_child"]),
        "right": compact_tree(node["right_child"]),
    }


def export_booster(booster: Any, output_path: str | Path, feature_names: list[str] | None = None) -> None:
    model = booster.dump_model()
    payload = {
        "version": model["version"],
        "feature_names": list(feature_names) if feature_names is not None else model["feature_names"],
        "average_output": bool(model.get("average_output", False)),
        "trees": [compact_tree(tree["tree_structure"]) for tree in model["tree_info"]],
    }
    Path(output_path).write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")


class PortableLightGBM:
    def __init__(self, model_path: str | Path) -> None:
        payload = json.loads(Path(model_path).read_text(encoding="utf-8"))
        self.feature_names = payload["feature_names"]
        self.average_output = payload["average_output"]
        self.trees = payload["trees"]

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        values = frame.loc[:, self.feature_names].to_numpy(dtype=float)
        predictions = np.zeros(len(values), dtype=float)
        for row_index, row in enumerate(values):
            total = 0.0
            for tree in self.trees:
                node = tree
                while "split_feature" in node:
                    value = row[node["split_feature"]]
                    if np.isnan(value):
                        go_left = node["default_left"]
                    elif node["decision_type"] == "==":
                        go_left = value == node["threshold"]
                    elif node["decision_type"] == "<=":
                        go_left = value <= node["threshold"]
                    else:
                        go_left = value < node["threshold"]
                    node = node["left"] if go_left else node["right"]
                total += node["leaf_value"]
            predictions[row_index] = total / len(self.trees) if self.average_output else total
        return predictions
