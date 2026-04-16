from pathlib import Path
import os
from tempfile import NamedTemporaryFile

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from model_adapter import predict_disaster


app = FastAPI(title="DDIS API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health_check() -> dict:
    return {"status": "ok"}


@app.post("/api/analyze")
async def analyze_damage(
    image: UploadFile = File(...),
    claimId: str = Form(""),
    location: str = Form(""),
    incidentType: str = Form(""),
) -> dict:
    suffix = Path(image.filename or "upload.jpg").suffix or ".jpg"
    with NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_file.write(await image.read())
        temp_path = temp_file.name

    try:
        result = predict_disaster(
            image_path=temp_path,
            claim_id=claimId,
            location=location,
            incident_type=incidentType,
        )
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

    normalized = {
        "disasterType": result.get("disasterType") or result.get("disaster_type") or "Unknown",
        "damageSeverity": (result.get("damageSeverity") or result.get("damage_severity") or "medium").title(),
        "confidenceScore": float(result.get("confidenceScore") or result.get("confidence_score") or 0.0),
        "metadata": result.get("metadata") or {},
    }
    return normalized
