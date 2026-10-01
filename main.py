
import io
import os
import sys
from pathlib import Path
from typing import List

import joblib
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field, create_model

try:
    from sklearn._loss import _loss as _sk_loss

    sys.modules.setdefault("_loss", _sk_loss)
except ImportError:
    pass

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = os.getenv("MODEL_PATH", str(BASE_DIR / "finscam" / "ml_model" / "fraud_model.joblib"))
MAX_BATCH = int(os.getenv("MAX_BATCH", "10000"))  # max rows per request / макс. строк за запрос

if not os.path.exists(MODEL_PATH):
    raise RuntimeError(f"Model file not found / Файл модели не найден: {MODEL_PATH}")

bundle = joblib.load(MODEL_PATH)
MODEL = bundle["model"]
THRESHOLD = float(bundle["threshold"])
FEATURES = list(bundle["features"])

Transaction = create_model(
    "Transaction",
    **{name: (float, Field(..., allow_inf_nan=False)) for name in FEATURES},
)


class PredictionResult(BaseModel):
    probability: float
    is_fraud: bool
    threshold: float


class BatchResult(BaseModel):
    count: int
    fraud_count: int
    results: List[PredictionResult]


app = FastAPI(title="Fraud Detection API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def run_model(df: pd.DataFrame) -> BatchResult:
    df = df[FEATURES]
    proba = MODEL.predict_proba(df)[:, 1]
    results = [
        PredictionResult(probability=round(float(p), 6), is_fraud=bool(p >= THRESHOLD), threshold=THRESHOLD)
        for p in proba
    ]
    return BatchResult(count=len(results), fraud_count=sum(r.is_fraud for r in results), results=results)

@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/model-info")
def model_info():
    return {
        "model": type(MODEL).__name__,
        "threshold": THRESHOLD,
        "n_features": len(FEATURES),
        "features": FEATURES,
        "max_batch": MAX_BATCH,
    }


@app.post("/predict", response_model=PredictionResult)
def predict(tx: Transaction):
    df = pd.DataFrame([tx.model_dump()])
    return run_model(df).results[0]


@app.post("/predict/batch", response_model=BatchResult)
def predict_batch(items: List[Transaction]):
    if not items:
        raise HTTPException(status_code=422, detail="Empty list / Пустой список")
    if len(items) > MAX_BATCH:
        raise HTTPException(status_code=413, detail=f"Too many rows (max {MAX_BATCH}) / Слишком много строк")
    df = pd.DataFrame([i.model_dump() for i in items])
    return run_model(df)


@app.post("/predict/csv", response_model=BatchResult)
async def predict_csv(file: UploadFile = File(...)):
    raw = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(raw))
    except Exception:
        raise HTTPException(status_code=422, detail="Cannot read CSV / Не удалось прочитать CSV")

    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise HTTPException(status_code=422, detail=f"Missing columns / Нет столбцов: {missing}")
    if len(df) == 0:
        raise HTTPException(status_code=422, detail="CSV is empty / CSV пустой")
    if len(df) > MAX_BATCH:
        raise HTTPException(status_code=413, detail=f"Too many rows (max {MAX_BATCH}) / Слишком много строк")

    df = df[FEATURES].apply(pd.to_numeric, errors="coerce")
    if df.isna().any().any():
        raise HTTPException(status_code=422, detail="NaN or non-numeric values found / Есть NaN или нечисловые значения")
    return run_model(df)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)