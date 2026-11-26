"""Generate deterministic or Gemini-powered delivery operations briefings."""

import json
import os
from pathlib import Path

from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[1]
SYSTEM_PROMPT = (
    "You are a delivery operations analyst. Using only the supplied aggregate ETA metrics, "
    "SHAP drivers, zone risks, and hour-band results, write a concise briefing with three bullets: "
    "highest-risk zone, likely delay drivers, and one concrete dispatch action. "
    "Do not invent numbers or claim that correlation proves causation."
)


def generate_rule_based_brief(summary: dict) -> str:
    zones = summary.get("zone_summary", [])
    drivers = summary.get("top_shap_drivers", [])
    metrics = summary.get("metrics", {})
    if not zones:
        return "No zone summary is available."

    worst_zone = max(zones, key=lambda item: item.get("sla_miss_rate", 0.0))
    driver_text = ", ".join(
        f"{item['feature']} ({item['mean_abs_shap_min']:.2f} min mean |SHAP|)" for item in drivers[:3]
    ) or "No SHAP drivers available"
    missed = metrics.get("missed_late_count", "n/a")
    accuracy = metrics.get("sla_accuracy")
    accuracy_text = f"{accuracy * 100:.1f}%" if accuracy is not None else "n/a"
    return (
        "### Delivery ETA Operations Brief\n"
        f"- **Highest-risk zone:** {worst_zone['zone']} has a "
        f"{worst_zone['sla_miss_rate'] * 100:.1f}% SLA miss rate and "
        f"{worst_zone['mae_min']:.2f} min MAE.\n"
        f"- **Delay drivers:** {driver_text}. The model missed {missed} late orders; "
        f"overall SLA-call accuracy is {accuracy_text}.\n"
        f"- **Dispatch action:** Review courier coverage and ETA buffers in {worst_zone['zone']} first, "
        "then inspect the hour bands with the highest miss rates before changing platform-wide policy."
    )


def rewrite_with_gemini(summary: dict, api_key: str, model: str = "gemini-2.5-flash") -> str:
    from google import genai

    client = genai.Client(api_key=api_key)
    payload = json.dumps(summary, indent=2)
    response = client.models.generate_content(
        model=model,
        contents=f"{SYSTEM_PROMPT}\n\nAggregate project results:\n{payload}",
    )
    text = getattr(response, "text", None)
    return text.strip() if text else generate_rule_based_brief(summary)


def generate_ops_brief(summary: dict, api_key: str | None = None) -> str:
    load_dotenv(ROOT / ".env")
    key = api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not key:
        return generate_rule_based_brief(summary)

    try:
        return rewrite_with_gemini(summary, key, os.getenv("GEMINI_MODEL", "gemini-2.5-flash"))
    except Exception:
        return generate_rule_based_brief(summary)


if __name__ == "__main__":
    summary = json.loads((ROOT / "models" / "shap_summary.json").read_text(encoding="utf-8"))
    print(generate_ops_brief(summary))
