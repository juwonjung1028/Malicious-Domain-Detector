from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import numpy as np
from tensorflow.keras.models import load_model
from keras.utils import pad_sequences  # Keras 3 / NumPy 2 호환


class DGAModel:
    def __init__(self, models_dir):
        models_dir = Path(models_dir)
        self.model_path = models_dir / "model.h5"
        self.meta_path = models_dir / "meta.json"

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        if not self.meta_path.exists():
            raise FileNotFoundError(f"Meta not found: {self.meta_path}")

        self.model = load_model(self.model_path.as_posix())
        with open(self.meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        self.valid_chars: Dict[str, int] = meta["valid_chars"]
        self.maxlen: int = int(meta["maxlen"])
        self.threshold: float = float(meta.get("threshold", 0.5))

    def _vectorize(self, domain: str):
        domain = domain.strip().lower()
        seq = [self.valid_chars.get(ch, 0) for ch in domain]
        return pad_sequences([seq], maxlen=self.maxlen)

    def predict_proba(self, domain: str) -> float:
        x = self._vectorize(domain)
        return float(self.model.predict(x, verbose=0)[0][0])

    def predict_label(self, domain: str) -> str:
        p = self.predict_proba(domain)
        return "malicious" if p >= self.threshold else "benign"
