"""ClinSense FastAPI application - Week 2."""

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Add project root
ROOT = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(ROOT))

from src.services.predictor import SpecialtyPredictor

app = FastAPI(
    title="ClinSense API",
    description="Medical specialty classification from clinical notes",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Global predictor (loaded on first request)
_predictor: SpecialtyPredictor | None = None


def get_predictor() -> SpecialtyPredictor:
    global _predictor
    if _predictor is None:
        model_path = os.getenv("CLINSENSE_MODEL_PATH", str(ROOT / "models" / "bert_finetuned"))
        _predictor = SpecialtyPredictor(model_path, max_length=512)
        _predictor.load()
    return _predictor


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=50000, description="Clinical note text")


class PredictResponse(BaseModel):
    specialty: str
    confidence: float
    probabilities: dict[str, float]
    model: str = "bert-base-uncased"


@app.get("/")
def root():
    return {"service": "ClinSense", "status": "ok", "version": "1.0.0"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    """Classify clinical note into medical specialty with confidence scores."""
    try:
        predictor = get_predictor()
        specialty, prob_dict = predictor.predict_proba(req.text)
        
        # Ensure we get the correct confidence even if there's a minor label mismatch
        confidence = prob_dict.get(specialty, max(prob_dict.values()) if prob_dict else 0.0)
        
        return PredictResponse(
            specialty=specialty,
            confidence=float(confidence),
            probabilities=prob_dict
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class DriftRequest(BaseModel):
    reference_texts: list[str] = Field(..., min_length=1, description="Reference text samples")
    current_texts: list[str] = Field(..., min_length=1, description="Current/production text samples")


@app.post("/monitor/drift")
def monitor_drift(req: DriftRequest):
    """Evidently AI data drift: compare current vs reference text distribution."""
    try:
        from src.monitoring.drift import compute_drift_report
        result = compute_drift_report(req.reference_texts, req.current_texts)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.on_event("startup")
async def startup():
    """Preload BERT model on startup if CLINSENSE_PRELOAD=true (reduces first-request latency)."""
    if os.getenv("CLINSENSE_PRELOAD", "false").lower() == "true":
        get_predictor()
