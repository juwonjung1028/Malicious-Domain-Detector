from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse, JSONResponse
from pydantic import BaseModel, Field

from .model_runtime import DGAModel
from .storage import JsonlStore


ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
LOGS_DIR = ROOT / "logs"
LOGS_DIR.mkdir(exist_ok=True)
PRED_LOG = LOGS_DIR / "predictions.jsonl"

app = FastAPI(title="Realtime Malicious Domain Detector (DGA/LSTM)")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

# 정적 UI는 /ui 로 서빙, 루트는 /ui/로 리다이렉트
static_dir = ROOT / "api" / "static"
app.mount("/ui", StaticFiles(directory=static_dir, html=True), name="static")

@app.get("/")
def root():
    return RedirectResponse(url="/ui/")

# 전역 싱글톤들
_model: Optional[DGAModel] = None
_store = JsonlStore(PRED_LOG)


class PredictIn(BaseModel):
    domain: str = Field(..., description="Domain (without TLD), e.g., google")


class PredictOut(BaseModel):
    domain: str
    probability: float
    label: str
    model_loaded: bool


@app.get("/health")
def health():
    return {"status": "ok", "ts": int(time.time())}


@app.get("/version")
def version():
    return {
        "app": "dga-lstm-web-demo",
        "model_exists": (MODELS_DIR / "model.h5").exists(),
        "meta_exists": (MODELS_DIR / "meta.json").exists(),
    }


def _get_model() -> DGAModel:
    global _model
    if _model is None:
        _model = DGAModel(MODELS_DIR)
    return _model


@app.post("/predict", response_model=PredictOut)
def predict(inp: PredictIn):
    # 모델 없을 때는 503으로 안내
    try:
        m = _get_model()
    except FileNotFoundError:
        return JSONResponse(
            status_code=503,
            content={
                "detail": (
                    "Model not found. Please run training: "
                    "python train/train_and_export.py --epochs 1"
                )
            },
        )

    if not inp.domain or len(inp.domain.strip()) == 0:
        raise HTTPException(400, "Empty domain")

    prob = m.predict_proba(inp.domain)
    label = "malicious" if prob >= m.threshold else "benign"

    # 로그에 기록 (id 자동 부여)
    _store.append(
        {
            "ts": int(time.time()),
            "domain": inp.domain,
            "probability": prob,
            "label": label,
        }
    )
    return PredictOut(
        domain=inp.domain, probability=prob, label=label, model_loaded=True
    )


@app.get("/logs")
def logs(limit: int = 20):
    return _store.tail(limit=limit)


@app.delete("/logs")
def delete_logs(
    all: bool = Query(default=False, description="true이면 전체 삭제"),
    id: Optional[str] = Query(default=None, description="특정 로그 id"),
):
    if all:
        _store.clear()
        return {"cleared": True}
    if id:
        ok = _store.delete_by_id(id)
        if not ok:
            raise HTTPException(404, detail="Log item not found")
        return {"deleted": True, "id": id}
    raise HTTPException(400, detail="Specify ?all=true or ?id=<log_id>")
