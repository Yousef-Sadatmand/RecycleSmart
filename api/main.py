"""
main.py — RecycleSmart FastAPI server

Endpoints:
  GET  /health    — liveness check (used by deployment platforms)
  POST /predict   — accepts an image file, returns classification result

Run locally:
  cd "Y:/Data Science/Develop/RecycleSmart"
  uvicorn api.main:app --reload
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from api.model import predict

# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="RecycleSmart API",
    description="Classify waste items into 9 categories and get bin instructions.",
    version="1.0.0",
)

# ── CORS ──────────────────────────────────────────────────────────────────────
# CORS = Cross-Origin Resource Sharing.
# Browsers block requests from one domain to another by default.
# This allows our React frontend (localhost:5173) to call this API (localhost:8000).
# allow_origins=["*"] means any origin — fine for local dev, tighten before production.

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    """Liveness check — returns ok if the server is running."""
    return {"status": "ok"}


@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)):
    """
    Accept an image upload and return the waste classification.

    Expected: multipart/form-data with field name "file"
    Returns:
      {
        "class":           "plastic",
        "confidence":      0.94,
        "bin_instruction": "♻️ Recycling bin...",
        "low_confidence":  false,
        "all_scores":      { "battery": 0.01, ... }
      }
    """
    # Validate file type
    if file.content_type not in ("image/jpeg", "image/png", "image/webp"):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file.content_type}. Send JPEG or PNG."
        )

    image_bytes = await file.read()
    result = predict(image_bytes)
    return result
