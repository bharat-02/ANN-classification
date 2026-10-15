"""Shared inference pipeline — guarantees train/serve consistency.

Loads model.h5 + pickles + feature_order.json, applies the EXACT same
transformations as train.py (LabelEncoder Gender, OneHot Geography
ignore-unknown, fixed column order, StandardScaler).
"""
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from config import CHURN_THRESHOLD  # single source of truth (defined in config.py)

BASE = Path(__file__).parent

_MODEL = None
_LE = None
_OHE = None
_SCALER = None
_FEATURES = None


def _load():
    global _MODEL, _LE, _OHE, _SCALER, _FEATURES
    if _MODEL is None:
        _MODEL = tf.keras.models.load_model(str(BASE / "model.h5"))
        with open(BASE / "label_encoder_gender.pkl", "rb") as f:
            _LE = pickle.load(f)
        with open(BASE / "onehot_encoder_geo.pkl", "rb") as f:
            _OHE = pickle.load(f)
        with open(BASE / "scaler.pkl", "rb") as f:
            _SCALER = pickle.load(f)
        with open(BASE / "feature_order.json") as f:
            _FEATURES = json.load(f)
    return _MODEL, _LE, _OHE, _SCALER, _FEATURES


def predict_churn(payload: dict) -> tuple[float, int]:
    """payload keys: CreditScore, Geography, Gender, Age, Tenure, Balance,
    NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary.
    Returns (probability, label) with label = int(proba >= CHURN_THRESHOLD).
    The threshold is fixed (config.py) and intentionally NOT a parameter:
    the UI must not let users change model behavior. Raises ValueError
    on invalid input."""
    model, le, ohe, scaler, features = _load()
    required = ["CreditScore", "Geography", "Gender", "Age", "Tenure",
                "Balance", "NumOfProducts", "HasCrCard", "IsActiveMember",
                "EstimatedSalary"]
    missing = [k for k in required if k not in payload]
    if missing:
        raise ValueError(f"Missing fields: {missing}")
    geo, gender = payload["Geography"], payload["Gender"]
    if geo not in list(ohe.categories_[0]):
        raise ValueError(f"Unknown Geography '{geo}'. Expected {list(ohe.categories_[0])}")
    if gender not in list(le.classes_):
        raise ValueError(f"Unknown Gender '{gender}'. Expected {list(le.classes_)}")

    row = pd.DataFrame([payload])
    row["Gender"] = le.transform(row["Gender"])
    geo_enc = ohe.transform(row[["Geography"]])
    geo_df = pd.DataFrame(geo_enc, columns=ohe.get_feature_names_out(["Geography"]))
    row = pd.concat([row.drop(columns=["Geography"]), geo_df], axis=1)[features]
    x = scaler.transform(row)
    proba = float(model.predict(x, verbose=0)[0][0])
    # Single-neuron sigmoid output: model was trained with y = Exited
    # (1 = churned), so output[0][0] IS P(churn). No column selection needed.
    label = int(proba >= CHURN_THRESHOLD)
    return proba, label
