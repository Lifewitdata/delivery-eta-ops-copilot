import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_shap_summary_has_real_hour_bands() -> None:
    summary = json.loads((ROOT / "models" / "shap_summary.json").read_text(encoding="utf-8"))
    assert summary["model"] == "lightgbm"
    assert summary["sample_size"] == 2_000
    assert {item["hour_band"] for item in summary["hour_band_summary"]} == {
        "late_night",
        "morning",
        "afternoon",
        "evening",
    }
    assert all(item["top_shap_drivers"] for item in summary["hour_band_summary"])
    assert all(
        driver["mean_abs_shap_min"] >= 0
        for item in summary["hour_band_summary"]
        for driver in item["top_shap_drivers"]
    )


def test_no_shap_placeholder_remains() -> None:
    source = (ROOT / "src" / "explainability.py").read_text(encoding="utf-8").lower()
    assert "todo" not in source
    assert "placeholder" not in source
    assert "notimplemented" not in source
