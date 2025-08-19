from __future__ import annotations
import time
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field
from .model_runtime import DGAModel
from .storage import JsonlStore

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / 'models'
LOGS_DIR = ROOT / 'logs'
LOGS_DIR.mkdir(exist_ok=True)
PRED_LOG = LOGS_DIR / 'predictions.jsonl'

app = FastAPI(title='Realtime Malicious Domain Detector (DGA/LSTM)')
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_methods=['*'], allow_headers=['*'])

static_dir = ROOT / 'api' / 'static'
app.mount("/ui", StaticFiles(directory=static_dir, html=True), name="static")

# 루트 접근 시 /ui/로 보내기 (브라우저 편의)
@app.get("/")
def root():
    return RedirectResponse(url="/ui/")

_model = None
_store = JsonlStore(PRED_LOG)

class PredictIn(BaseModel):
    domain: str = Field(..., description='Domain (without TLD), e.g., google')

class PredictOut(BaseModel):
    domain: str
    probability: float
    label: str
    model_loaded: bool

@app.get('/health')
def health():
    return {'status': 'ok', 'ts': int(time.time())}

@app.get('/version')
def version():
    return {'app': 'dga-lstm-web-demo',
            'model_exists': (MODELS_DIR / 'model.h5').exists(),
            'meta_exists': (MODELS_DIR / 'meta.json').exists()}

def _get_model() -> DGAModel:
    global _model
    if _model is None:
        _model = DGAModel(MODELS_DIR)
    return _model

@app.post('/predict', response_model=PredictOut)
def predict(inp: PredictIn):
    m = _get_model()
    if not inp.domain or len(inp.domain.strip()) == 0:
        raise HTTPException(400, 'Empty domain')
    prob = m.predict_proba(inp.domain)
    label = 'malicious' if prob >= m.threshold else 'benign'
    _store.append({'ts': int(time.time()), 'domain': inp.domain, 'probability': prob, 'label': label})
    return PredictOut(domain=inp.domain, probability=prob, label=label, model_loaded=True)

@app.get('/logs')
def logs(limit: int = 50):
    return _store.tail(limit=limit)