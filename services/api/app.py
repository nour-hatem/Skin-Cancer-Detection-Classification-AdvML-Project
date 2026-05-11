import asyncio
import json
import logging
import os
from collections import defaultdict
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated

import httpx
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [api] %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────
_HERE = Path(__file__).parent

CLASSIFIER_URL = os.getenv("CLASSIFIER_URL", "http://classifier:8001")
LOG_PATH       = Path(os.getenv("LOG_PATH",  str(_HERE / "logs/predictions.jsonl")))
DATA_PATH      = Path(os.getenv("DATA_PATH", str(_HERE / "data/lesions.json")))

DISCLAIMER = (
    "This tool is for screening assistance only and does not constitute a medical diagnosis. "
    "Please consult a licensed dermatologist for any skin concerns."
)

# Business rule: risk classification by lesion code.
# Lives here — not in classifier — because it is domain policy, not model semantics.
RISK_MAP: dict[str, str] = {
    "mel":   "HIGH",
    "bcc":   "HIGH",
    "akiec": "MEDIUM",
    "bkl":   "LOW",
    "df":    "LOW",
    "nv":    "LOW",
    "vasc":  "LOW",
}


# ── Schemas ───────────────────────────────────────────────────────────
class PredictResponse(BaseModel):
    lesion:                  str
    lesion_name:             str
    risk:                    str
    confidence:              float
    mel_prob:                float
    mel_threshold_triggered: bool
    all_probs:               dict[str, float]
    urgency_message:         str
    disclaimer:              str


class LogEntry(BaseModel):
    timestamp:               str
    lesion:                  str
    lesion_name:             str
    risk:                    str
    confidence:              float
    mel_prob:                float
    mel_threshold_triggered: bool
    image_filename:          str | None


# ── Startup ───────────────────────────────────────────────────────────
_lesions: dict = {}
_client:  httpx.AsyncClient


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _lesions, _client

    with DATA_PATH.open() as f:
        _lesions = json.load(f)
    log.info("Loaded %d lesion profiles from %s", len(_lesions), DATA_PATH)

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    _client = httpx.AsyncClient(timeout=httpx.Timeout(30.0))
    log.info("API ready — classifier=%s", CLASSIFIER_URL)
    yield
    await _client.aclose()


# ── Application ───────────────────────────────────────────────────────
app = FastAPI(
    title="SkinCancerDetector API",
    description=(
        "Orchestration layer. "
        "Calls classifier → attaches risk (RISK_MAP) → enriches with clinical data "
        "(loaded at startup from lesions.json) → writes audit log → returns to client."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ── Logging middleware (not a service — an async function) ────────────
LOGGER_URL = os.getenv("LOGGER_URL", "http://logger:8003")


async def _fire_log(payload: dict) -> None:
    try:
        await _client.post(f"{LOGGER_URL}/log", json=payload)
    except Exception as exc:
        log.error("Logger unreachable: %s", exc)


# ── Routes ────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    clf_status = "unreachable"
    try:
        r = await _client.get(f"{CLASSIFIER_URL}/health", timeout=3.0)
        clf_status = "ok" if r.status_code == 200 else "degraded"
    except Exception:
        pass
    return {
        "status":   "ok" if clf_status == "ok" else "degraded",
        "services": {"classifier": clf_status},
    }


@app.post("/predict", response_model=PredictResponse)
async def predict(
    file: Annotated[UploadFile, File(description="Dermatoscopic lesion image (JPG/PNG/WEBP)")],
):
    raw = await file.read()

    # 1 — call classifier for inference + threshold decision
    try:
        clf_r = await _client.post(
            f"{CLASSIFIER_URL}/predict",
            files={"file": (file.filename or "image.jpg", raw, file.content_type or "image/jpeg")},
        )
        clf_r.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise HTTPException(status_code=502, detail=f"Classifier error: {exc.response.text}")
    except httpx.RequestError as exc:
        raise HTTPException(status_code=503, detail=f"Classifier unreachable: {exc}")

    clf    = clf_r.json()
    lesion = clf["lesion"]

    # 2 — api layer attaches risk: business rule, classifier knows nothing about it
    risk = RISK_MAP.get(lesion, "UNKNOWN")

    # 3 — enrich with clinical data (in-process dict lookup, zero network hop)
    info            = _lesions.get(lesion, {})
    lesion_name     = info.get("name",            lesion)
    urgency_message = info.get("urgency_message", "Consult a dermatologist.")

    response = PredictResponse(
        lesion=lesion,
        lesion_name=lesion_name,
        risk=risk,
        confidence=round(clf["confidence"], 4),
        mel_prob=round(clf["mel_prob"], 4),
        mel_threshold_triggered=clf["mel_threshold_triggered"],
        all_probs={k: round(v, 4) for k, v in clf["all_probs"].items()},
        urgency_message=urgency_message,
        disclaimer=DISCLAIMER,
    )

    # 4 — async audit log: fire-and-forget, never blocks response
    asyncio.create_task(_fire_log({
        "lesion":                  lesion,
        "lesion_name":             lesion_name,
        "risk":                    risk,
        "confidence":              response.confidence,
        "mel_prob":                response.mel_prob,
        "mel_threshold_triggered": clf["mel_threshold_triggered"],
        "image_filename":          file.filename,
    }))

    log.info(
        "Prediction: %s (%s) conf=%.4f mel=%.4f triggered=%s",
        lesion, risk, clf["confidence"], clf["mel_prob"], clf["mel_threshold_triggered"],
    )
    return response


@app.get("/logs", response_model=list[LogEntry])
def read_logs(
    limit:  int        = Query(50, ge=1, le=500, description="Max entries to return (most recent first)"),
    lesion: str | None = Query(None, description="Filter by lesion code"),
    risk:   str | None = Query(None, description="Filter by risk level (HIGH/MEDIUM/LOW)"),
):
    if not LOG_PATH.exists():
        return []
    entries: list[LogEntry] = []
    with LOG_PATH.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            e = LogEntry.model_validate_json(line)
            if lesion and e.lesion != lesion:
                continue
            if risk and e.risk != risk.upper():
                continue
            entries.append(e)
    return entries[-limit:]


@app.get("/stats")
def stats():
    if not LOG_PATH.exists():
        return {"total": 0, "by_lesion": {}, "by_risk": {}}
    by_lesion: dict[str, int] = defaultdict(int)
    by_risk:   dict[str, int] = defaultdict(int)
    total = 0
    with LOG_PATH.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            e = json.loads(line)
            by_lesion[e["lesion"]] += 1
            by_risk[e["risk"]]     += 1
            total += 1
    return {
        "total":     total,
        "by_lesion": dict(sorted(by_lesion.items(), key=lambda x: -x[1])),
        "by_risk":   dict(by_risk),
    }


# Static files mounted after all routes — API routes always take precedence
app.mount("/", StaticFiles(directory=_HERE / "static", html=True), name="static")