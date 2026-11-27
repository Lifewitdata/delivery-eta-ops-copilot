import builtins
from pathlib import Path

from streamlit.testing.v1 import AppTest


ROOT = Path(__file__).resolve().parents[1]


def test_streamlit_app_runs_without_lightgbm(monkeypatch) -> None:
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "lightgbm" or name.startswith("lightgbm."):
            raise ModuleNotFoundError("No module named 'lightgbm'")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    app = AppTest.from_file(str(ROOT / "app" / "streamlit_app.py"), default_timeout=30)
    app.run()
    assert not app.exception
    assert app.title[0].value == "ML Delivery ETA Predictor & AI Operations Copilot"
    labels = [metric.label for metric in app.metric]
    assert "LightGBM MAE (min)" in labels
    assert "Predicted Delivery ETA" in labels
