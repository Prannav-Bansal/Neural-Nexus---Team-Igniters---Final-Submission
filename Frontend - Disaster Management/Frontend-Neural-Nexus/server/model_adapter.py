from pathlib import Path
from typing import Any, Dict


def predict_disaster(image_path: str, claim_id: str = "", location: str = "", incident_type: str = "") -> Dict[str, Any]:
    """
    Replace only the inside of this function with a call to your existing model.
    Treat the model as a black-box inference API and keep its internals untouched.

    Expected return shape:
    {
        "disasterType": "Flood",
        "damageSeverity": "high",
        "confidenceScore": 0.94,
        "metadata": {"topLabel": "flood", "secondaryConfidence": 0.81}
    }
    """
    _ = (claim_id, location, incident_type)
    image_name = Path(image_path).name.lower()

    if "fire" in image_name:
        return {
            "disasterType": "Fire",
            "damageSeverity": "high",
            "confidenceScore": 0.91,
            "metadata": {"topLabel": "fire", "secondaryConfidence": 0.74, "damageSpread": "broad"},
        }
    if "quake" in image_name:
        return {
            "disasterType": "Earthquake",
            "damageSeverity": "medium",
            "confidenceScore": 0.83,
            "metadata": {"topLabel": "earthquake", "secondaryConfidence": 0.68, "structureCracks": True},
        }

    return {
        "disasterType": incident_type or "Flood",
        "damageSeverity": "medium",
        "confidenceScore": 0.87,
        "metadata": {"topLabel": "flood", "secondaryConfidence": 0.72, "damageSpread": "localized"},
    }
