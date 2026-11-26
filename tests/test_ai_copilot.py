import pytest

from src import ai_copilot


SUMMARY = {
    "metrics": {"missed_late_count": 12, "sla_accuracy": 0.93},
    "top_shap_drivers": [
        {"feature": "distance_km", "mean_abs_shap_min": 2.2},
        {"feature": "courier_age", "mean_abs_shap_min": 2.0},
    ],
    "zone_summary": [
        {"zone": "Riyadh", "sla_miss_rate": 0.35, "mae_min": 3.1},
    ],
}


def test_rule_based_fallback_without_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    brief = ai_copilot.generate_ops_brief(SUMMARY, api_key="")
    assert "Riyadh" in brief
    assert "distance_km" in brief


def test_gemini_rewrite_is_used(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    def fake_rewrite(summary: dict, api_key: str, model: str) -> str:
        calls.append((summary, api_key, model))
        return "Gemini briefing"

    monkeypatch.setattr(ai_copilot, "rewrite_with_gemini", fake_rewrite)
    assert ai_copilot.generate_ops_brief(SUMMARY, api_key="test-key") == "Gemini briefing"
    assert calls[0][1] == "test-key"


def test_gemini_failure_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(summary: dict, api_key: str, model: str) -> str:
        raise RuntimeError("simulated API failure")

    monkeypatch.setattr(ai_copilot, "rewrite_with_gemini", fail)
    assert "Riyadh" in ai_copilot.generate_ops_brief(SUMMARY, api_key="test-key")
