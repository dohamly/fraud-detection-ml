"""
FraudGuard AI — FastAPI backend
--------------------------------
Serves the same XGBoost/Random Forest/Logistic Regression model your
Streamlit app (app/app.py) uses, over a REST API, so a plain HTML/JS
frontend (index.html) can call it instead of running inside Streamlit.

Expects this project layout (main.py lives at the repo root):
    fraud-detection-ml/
      app/app.py
      models/best_model.txt, <model>.joblib, scaler.joblib, feature_names.joblib
      reports/metrics.json, reports/figures/*.png
      src/preprocessing.py   (FEATURE_LABELS, prepare_model_frame, single_transaction_to_frame)
      main.py                <-- this file
      index.html

Run from the repo root:
    pip install -r requirements.txt
    uvicorn main:app --reload --port 8000

Docs:
    http://localhost:8000/docs
"""

import json
import os
import sys

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, "models")
FIG_DIR = os.path.join(BASE_DIR, "reports", "figures")
METRICS_PATH = os.path.join(BASE_DIR, "reports", "metrics.json")
TREE_MODELS = ("Random Forest", "XGBoost")

sys.path.append(os.path.join(BASE_DIR, "src"))
try:
    from preprocessing import FEATURE_LABELS, prepare_model_frame, single_transaction_to_frame  # noqa: E402
except ImportError as e:  # pragma: no cover
    raise RuntimeError(
        "Could not import src/preprocessing.py. Make sure this backend sits next to your "
        "existing project (models/, src/, reports/) exactly like app.py does."
    ) from e

app = FastAPI(title="FraudGuard AI API", version="1.0")

# Dev-friendly CORS. Tighten allow_origins to your real frontend origin(s) before deploying.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

if os.path.isdir(FIG_DIR):
    app.mount("/figures", StaticFiles(directory=FIG_DIR), name="figures")

_artifacts: dict = {}


def get_artifacts() -> dict:
    """Load model + scaler + feature names + SHAP explainer once, then cache in memory."""
    if not _artifacts:
        with open(os.path.join(MODEL_DIR, "best_model.txt")) as f:
            best_model_name = f.read().strip()
        model_file = best_model_name.lower().replace(" ", "_") + ".joblib"
        model = joblib.load(os.path.join(MODEL_DIR, model_file))
        scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.joblib"))
        feature_names = joblib.load(os.path.join(MODEL_DIR, "feature_names.joblib"))

        explainer = None
        if best_model_name in TREE_MODELS:
            import shap  # imported lazily: only needed for tree models

            explainer = shap.TreeExplainer(model)

        _artifacts.update(
            model=model,
            scaler=scaler,
            feature_names=feature_names,
            best_model_name=best_model_name,
            explainer=explainer,
        )
    return _artifacts


class TransactionIn(BaseModel):
    step: int
    tx_type: str
    amount: float
    oldbalanceOrg: float
    newbalanceOrig: float
    oldbalanceDest: float
    newbalanceDest: float
    isFlaggedFraud: bool = False


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/model-info")
def model_info():
    a = get_artifacts()
    roc_auc = None
    try:
        with open(METRICS_PATH) as f:
            metrics = json.load(f)
        roc_auc = metrics[a["best_model_name"]]["roc_auc"]
    except Exception:
        pass
    figures = []
    if os.path.isdir(FIG_DIR):
        figures = sorted(f for f in os.listdir(FIG_DIR) if f.lower().endswith((".png", ".jpg", ".jpeg")))
    return {
        "model": a["best_model_name"],
        "task": "Binary classification",
        "metric": "F1-score + ROC-AUC",
        "roc_auc": roc_auc,
        "dataset": "PaySim-schema synthetic",
        "figures": figures,
    }


@app.post("/predict")
def predict(tx: TransactionIn):
    a = get_artifacts()
    model, scaler, feature_names = a["model"], a["scaler"], a["feature_names"]
    explainer, best_model_name = a["explainer"], a["best_model_name"]

    try:
        raw = single_transaction_to_frame(
            step=tx.step,
            tx_type=tx.tx_type,
            amount=tx.amount,
            oldbalanceOrg=tx.oldbalanceOrg,
            newbalanceOrig=tx.newbalanceOrig,
            oldbalanceDest=tx.oldbalanceDest,
            newbalanceDest=tx.newbalanceDest,
            isFlaggedFraud=int(tx.isFlaggedFraud),
        )
        X = prepare_model_frame(raw).reindex(columns=feature_names, fill_value=0)
        X_scaled = scaler.transform(X)
        proba_fraud = float(model.predict_proba(X_scaled)[0, 1])
        prediction = int(proba_fraud >= 0.5)

        if explainer is not None:
            shap_raw = explainer.shap_values(X_scaled)
            values = shap_raw[0] if isinstance(shap_raw, list) else shap_raw[0]
            values = np.asarray(values).reshape(-1)
        else:
            values = (model.coef_[0] * X_scaled[0]).reshape(-1)

        order = np.argsort(-np.abs(values))[:4]
        top_factors = [
            {
                "feature": FEATURE_LABELS.get(feature_names[i], feature_names[i]),
                "value": float(values[i]),
            }
            for i in order
        ]
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {
        "probability": proba_fraud,
        "prediction": "FRAUD" if prediction == 1 else "LEGITIMATE",
        "model": best_model_name,
        "top_factors": top_factors,
        "method": "SHAP values (TreeExplainer)" if best_model_name in TREE_MODELS else "Linear contribution",
    }
