# Disaster Damage Intelligence System

Hackathon-ready UI for an AI-powered disaster insurance verification workflow. This project keeps your model untouched and wraps it with a premium product dashboard.

## Stack

- Frontend: React + Vite
- UI: Custom premium dashboard styling
- Backend: FastAPI wrapper around your existing inference function

## Folder Structure

```text
.
|-- server/
|   |-- app.py
|   |-- model_adapter.py
|   `-- requirements.txt
|-- src/
|   |-- data/mockClaims.js
|   |-- utils/report.js
|   |-- App.jsx
|   |-- main.jsx
|   `-- styles.css
|-- index.html
|-- package.json
|-- vite.config.js
`-- README.md
```

## Run Locally

### 1. Frontend

```bash
npm install
npm run dev
```

The frontend starts on `http://localhost:5173`.

### 2. Backend

```bash
cd server
pip install -r requirements.txt
uvicorn app:app --reload --port 8000
```

The API starts on `http://127.0.0.1:8000`.

## Connect Your Existing Model

Open `server/model_adapter.py` and replace the inside of `predict_disaster(...)` with a direct call to your existing model inference function.

Example pattern:

```python
from your_model_script import model


def predict_disaster(image_path: str, claim_id: str = "", location: str = "", incident_type: str = ""):
    result = model.predict(image_path)
    return {
        "disasterType": result["disaster_type"],
        "damageSeverity": result["damage_severity"],
        "confidenceScore": result["confidence"],
        "metadata": result.get("metadata", {}),
    }
```

Important:

- Do not change model internals.
- Only adapt the returned fields to the UI contract.
- Keep the model as a black-box function/API.

## Included Product Features

- Premium landing dashboard
- Claim evidence upload
- AI analysis view with preview
- Damage insights panel
- Claim authenticity check
- Report generation and download
- Historical claims sidebar
- Loading-state button copy for demo polish

## Notes

- The backend currently includes a filename-based fallback stub so the UI can be demoed immediately.
- Replace that stub with your real model call when your script is available in this workspace.
