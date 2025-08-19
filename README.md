# DGA/LSTM Realtime Malicious Domain Detector (Web Demo)

FastAPI + Keras LSTM으로 **도메인 본문**(TLD 제외)을 입력받아 악성(DGA 등) 여부를 예측하는 데모입니다.  
웹 UI(`/ui/`)와 REST API(`/predict`, `/logs` 등)를 제공합니다.

---

## 1. 요구사항

- Python 3.10 권장
- Windows/PowerShell 기준 가이드
- 주요 라이브러리
  - TensorFlow 2.20.x, Keras 3.10.x
  - FastAPI 0.116.x, Uvicorn 0.35.x
  - NumPy 1.26.x 이상
  > 같은 파이썬에 설치되도록 항상 `python -m pip ...` 형식을 권장합니다


## 2. 프로그램 구조
dga-lstm-web-demo/ 
├─ api/ 
│  ├─ app.py              # FastAPI 앱: /health, /version, /predict, /logs + 정적 UI(/ui) 
│  ├─ model_runtime.py    # 모델 로더/전처리 (meta.json 기반), keras.utils.pad_sequences 사용 
│  ├─ storage.py          # 예측 결과 JSONL 로깅 
│  └─ static/ 
│     └─ index.html       # 웹 UI (입력창/결과/최근 로그 표) 
├─ train/ 
│  ├─ train_and_export.py # 학습→검증→최적 가중치 저장→models/ 내보내기 
│  ├─ lstm.py             # Keras 3 호환 LSTM 모델 정의 (Embedding→LSTM→Dropout→Dense→Sigmoid) 
│  ├─ data.py             # 데이터 생성/로딩(악성 DGA + Alexa 정상), 경로 고정(DATA_DIR) 
│  ├─ top-1m.csv          # 정상 도메인 소스(csv) (해당 위치 자동 사용) 
│  ├─ traindata.pkl       # 캐시(자동 생성, DATA_DIR 기준) 
│  ├─ banjori.py, corebot.py, ...   # DGA Generator 모듈 및 리소스 일체 
│  └─ ... (matsnu_dict*, suppobox_words*, gozi_*.txt, set2_seeds.json, words.json 등) 
├─ models/ 
│  ├─ model.h5            # 학습 결과 (생성 후 존재) 
│  └─ meta.json           # {valid_chars, maxlen, threshold} (생성 후 존재) 
├─ Dockerfile 
├─ requirements.txt 
└─ README.md 


## 3. 프로그램 실행 흐름 (Data → Train → Export → Serve → UI)
**(1) 데이터 구성 (train/data.py)**
- DGA 모듈들(banjori, kraken, matsnu, suppobox, gozi 등)로 악성 도메인 생성.
- top-1m.csv(train 폴더)에서 정상 도메인을 같은 수만큼 추출.
- 합쳐서 [('label', 'domain'), ...] → traindata.pkl로 캐시. 
**(2) 모델 학습/내보내기 (train/train_and_export.py)**
- data.get_data() 로 캐시/생성 데이터 로드.
- 문자 단위 인덱싱/패딩 → lstm.build_model() 생성.
- 에폭마다 홀드아웃 AUC 측정 → 최고 성능 가중치 저장.
- 출력: models/model.h5, models/meta.json. 
**(3) 서빙/추론 (api/app.py, api/model_runtime.py)**
- 서버 기동 시 /version에서 모델/메타 존재 여부 확인.
- POST /predict 입력(도메인 본문) → DGAModel이 meta.json 기반 인코딩/패딩 → model.h5 추론 → 확률/라벨 반환.
- GET /logs 최근 결과를 JSONL에서 반환.
- GET /ui/ 간단한 웹 UI 제공. 
(참고)데이터 인코딩 규칙(학습/추론 일치)
- 소문자화 → 문자별 인덱싱(valid_chars) → maxlen으로 좌측 패딩 → LSTM 입력.
- meta.json 예:
{
  "valid_chars": {"a":1,"b":2,...},
  "maxlen": 23,
  "threshold": 0.5
}
- 추론 라벨링: probability >= threshold → "malicious" else "benign". 
**내부 동작 요약**
- 학습: 문자 인덱스(valid_chars) 생성 → pad(maxlen) → LSTM 분류 → 홀드아웃 AUC 최고 에폭 선택.
- 내보내기: models/model.h5 + meta.json 저장.
- 추론: meta.json 기준 동일 전처리로 예측(threshold 기본 0.5).


## 4. 빠른 실행 예시
**(1) (선택) 가상환경** 
python -3.10 -m venv .venv 
.\.venv\Scripts\Activate.ps1 
python -m pip install --upgrade pip 

**(2) 의존성 설치** 
python -m pip install -r requirements.txt 

**(3) 학습 및 내보내기** 
python train/train_and_export.py --epochs 1 
  > 완료 후 아래 목록 생성 확인 
  models/model.h5 
  models/meta.json 

**(4) 서버 실행**
python -m uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload 
  > 확인 
  curl http://localhost:8000/version 
  => model_exists:true, meta_exists:true 

**(5) UI 접속** 
http://localhost:8000/ui/ 접속 후 도메인 본문 입력 → Check 
  > 예: google, facebook, xqzvtnpl, lzxvtdqprmna 입력 후 Check 버튼 클릭 
  

## 5. 자주 묻는 질문(FAQ) / 트러블슈팅
  > Model not found (models/model.h5) 
  → 먼저 학습을 수행하세요: python train/train_and_export.py --epochs 1

  > /predict 405 또는 /logs 404 
  → 정적 라우팅 충돌 이슈입니다. UI는 /ui로 마운트되어야 합니다. 
  app.py에서 app.mount("/ui", StaticFiles(...))와 루트 리다이렉트(/ -> /ui/)가 적용되어 있어야 합니다. 

  > top-1m.csv FileNotFoundError 
  → train/top-1m.csv가 존재해야 합니다(기본 사용). 다른 위치라면 data.py의 get_alexa()에 절대경로를 넘기거나 파일을 이 위치로 옮기세요. 

  > ModuleNotFoundError: tensorflow 
  → pip와 python 인터프리터가 다를 수 있습니다. python -m pip install ...을 사용하세요. 

  > AttributeError: np.unicode_ was removed 
  → pad_sequences를 keras.utils에서 임포트해야 합니다. 
  from keras.utils import pad_sequences 


## 6. API 사용 예시
**Health & Version** 
curl http://localhost:8000/health 
curl http://localhost:8000/version 

**예측** 
curl -X POST http://localhost:8000/predict ` 
     -H "Content-Type: application/json" ` 
     -d "{\"domain\":\"xqzvtnpl\"}" 
  
  > 응답 예시 
  { 
    "domain": "xqzvtnpl", 
    "probability": 0.97, 
    "label": "malicious", 
    "model_loaded": true 
  } 

**최근 로그** 
curl "http://localhost:8000/logs?limit=20" 


## 7. Docker (선택) 
docker build -t dga-lstm-web-demo . 
docker run --rm -p 8000:8000 dga-lstm-web-demo 
// http://localhost:8000/ui/ 
