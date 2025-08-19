from __future__ import annotations
import os
import argparse, json
from pathlib import Path
import numpy as np
from keras.utils import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import data as data
import lstm as lstm

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / 'models'
MODELS_DIR.mkdir(exist_ok=True)

# 상대경로(./top-1m.csv, traindata.pkl 등)를 train/ 기준으로 고정
os.chdir(Path(__file__).resolve().parent)

def build_char_mapping(domains):
    valid_chars = {x: idx + 1 for idx, x in enumerate(set(''.join(domains)))}
    maxlen = max(len(x) for x in domains)
    return valid_chars, maxlen

def vectorize(domains, valid_chars, maxlen):
    X = [[valid_chars.get(ch, 0) for ch in s] for s in domains]
    return pad_sequences(X, maxlen=maxlen)

def main(epochs=5, batch_size=128, top1m_csv=None):
    if top1m_csv:
        top1m_csv = Path(top1m_csv)
        if not top1m_csv.exists():
            raise FileNotFoundError(f'top-1m.csv not found at {top1m_csv}')
    indata = data.get_data(force=False)

    X_domains = [x[1].lower() for x in indata]
    y_labels = [x[0] for x in indata]
    y = np.array([0 if lbl == 'benign' else 1 for lbl in y_labels], dtype='int32')

    valid_chars, maxlen = build_char_mapping(X_domains)
    X = vectorize(X_domains, valid_chars, maxlen)

    X_train, X_holdout, y_train, y_holdout = train_test_split(
        X, y, test_size=0.05, random_state=42, stratify=y
    )

    model = lstm.build_model(max_features=len(valid_chars)+1, maxlen=maxlen)

    best_auc = -1.0; best_weights = None
    for ep in range(epochs):
        model.fit(X_train, y_train, batch_size=batch_size, epochs=1, verbose=1)
        probs = model.predict(X_holdout, verbose=0).ravel()  # ← 1D로
        auc = roc_auc_score(y_holdout, probs)
        print(f'[Epoch {ep+1}] holdout AUC={auc:.6f}')
        if auc > best_auc:
            best_auc, best_weights = auc, model.get_weights()

    if best_weights is not None:
        model.set_weights(best_weights)

    model_path = MODELS_DIR / 'model.h5'
    meta_path = MODELS_DIR / 'meta.json'
    model.save(model_path.as_posix())
    meta = {'valid_chars': valid_chars, 'maxlen': int(maxlen), 'threshold': 0.5}
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f'Saved model to {model_path}')
    print(f'Saved meta to {meta_path}')

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=5)
    ap.add_argument('--batch_size', type=int, default=128)
    ap.add_argument('--top1m_csv', type=str, default=None)
    args = ap.parse_args()
    main(epochs=args.epochs, batch_size=args.batch_size, top1m_csv=args.top1m_csv)
