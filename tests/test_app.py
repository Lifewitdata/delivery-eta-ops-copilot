from pathlib import Path

from streamlit.testing.v1 import AppTest


ROOT = Path(__file__).resolve().parents[1]


def test_streamlit_app_runs_with_lightgbm() -> None:
    app = AppTest.from_file(str(ROOT / "app" / "streamlit_app.py"), default_timeout=30)
    app.run()
    assert not app.exception
    assert app.title[0].value == "Delivery ETA Predictor & AI Operations Copilot"
    labels = [metric.label for metric in app.metric]
    assert "LightGBM MAE (min)" in labels
    assert "Predicted Delivery ETA" in labels
