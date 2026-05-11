import io
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from pydantic import BaseModel
from torchvision import models, transforms

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [classifier] %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────
_HERE = Path(__file__).parent

CLASS_NAMES   = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
MODEL_PATH    = Path(os.getenv("MODEL_PATH", str(_HERE / "skin_cancer_model.pth")))
MEL_THRESHOLD = float(os.getenv("MEL_THRESHOLD", "0.35"))
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"

_MEL_IDX = CLASS_NAMES.index("mel")   # 4 — derived, never hardcoded elsewhere

_MEAN = [0.7630, 0.5456, 0.5700]      # HAM10000 channel means
_STD  = [0.1409, 0.1526, 0.1695]      # HAM10000 channel stds

_preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(_MEAN, _STD),
])


# ── Schema ────────────────────────────────────────────────────────────
class PredictResponse(BaseModel):
    """
    ML inference result. Contains no business rules (no risk level, no disclaimer).
    The threshold override is model logic — tuned on validation set alongside the model.
    """
    lesion:                  str
    confidence:              float
    mel_prob:                float
    mel_threshold_triggered: bool
    all_probs:               dict[str, float]


# ── Model lifecycle ───────────────────────────────────────────────────
def _build_model() -> nn.Module:
    m = models.efficientnet_v2_s(weights=None)
    in_features = m.classifier[1].in_features
    m.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, len(CLASS_NAMES)),
    )
    return m


_model: nn.Module | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model
    m = _build_model()
    m.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    m.to(DEVICE).eval()
    _model = m
    log.info(
        "Model ready — path=%s device=%s threshold=%.2f",
        MODEL_PATH, DEVICE, MEL_THRESHOLD,
    )
    yield
    log.info("Classifier shutdown")


# ── Application ───────────────────────────────────────────────────────
app = FastAPI(
    title="Skin Cancer Classifier",
    description=(
        "EfficientNetV2-S inference service. "
        "Applies MEL_THRESHOLD override and returns prediction semantics. "
        "Owns no business rules — RISK_MAP lives in the api layer."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    return {
        "status":        "ok" if _model is not None else "loading",
        "device":        DEVICE,
        "classes":       CLASS_NAMES,
        "mel_threshold": MEL_THRESHOLD,
    }


@app.post("/predict", response_model=PredictResponse)
async def predict(
    file: Annotated[UploadFile, File(description="Dermatoscopic image (JPG/PNG/WEBP)")],
):
    if _model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded yet")

    try:
        raw = await file.read()
        img = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        raise HTTPException(
            status_code=422,
            detail="Cannot decode image — file must be a valid JPG, PNG, or WEBP",
        )

    tensor = _preprocess(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs: list[float] = F.softmax(_model(tensor), dim=1)[0].cpu().tolist()

    mel_prob  = probs[_MEL_IDX]
    triggered = mel_prob >= MEL_THRESHOLD

    if triggered:
        # Override: mel probability alone exceeds threshold regardless of argmax
        lesion     = "mel"
        confidence = mel_prob
    else:
        idx        = int(max(range(len(probs)), key=lambda i: probs[i]))
        lesion     = CLASS_NAMES[idx]
        confidence = probs[idx]

    return PredictResponse(
        lesion=lesion,
        confidence=round(confidence, 6),
        mel_prob=round(mel_prob, 6),
        mel_threshold_triggered=triggered,
        all_probs={CLASS_NAMES[i]: round(probs[i], 6) for i in range(len(CLASS_NAMES))},
    )