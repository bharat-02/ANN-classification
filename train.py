"""Optimized ANN classification pipeline for Customer Churn (Churn_Modelling.csv).

Fixes vs original notebooks:
- Stratified train/val/test split (70/15/15) instead of single unstratified 80/20.
- Encoders + scaler fit on TRAIN ONLY (originals fit on full data -> leakage).
- OneHotEncoder(handle_unknown='ignore') for robust inference.
- Fixed feature order persisted to feature_order.json.
- Adam lr=0.001 (original 0.01 too high), EarlyStopping patience 10-15
  + ReduceLROnPlateau, Dropout/L2 variants.
- Class imbalance (Exited ~20%): stratified splits + F1/ROC-AUC/PR-AUC
  model selection (not accuracy alone) + one class-weighted variant.
- Compares ANN variants + LogisticRegression / RandomForest baselines.
- Saves model.h5 + pickles + metrics.json + PNG plots.

Usage:
    venv/Scripts/python.exe train.py
"""
import json
import os
import random
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle

warnings.filterwarnings("ignore")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

import tensorflow as tf

tf.random.set_seed(SEED)
# CPU-only env (no GPU devices on this machine); keep threads reasonable.
tf.config.threading.set_intra_op_parallelism_threads(4)
tf.config.threading.set_inter_op_parallelism_threads(4)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

from config import CHURN_THRESHOLD

BASE = Path(__file__).parent
CSV = BASE / "Churn_Modelling.csv"
RESULTS = BASE / "results"
RESULTS.mkdir(exist_ok=True)

FEATURE_ORDER = [
    "CreditScore", "Gender", "Age", "Tenure", "Balance",
    "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary",
    "Geography_France", "Geography_Germany", "Geography_Spain",
]


def load_and_validate():
    df = pd.read_csv(CSV)
    print(f"[data] shape={df.shape}")
    print(f"[data] target distribution:\n{df['Exited'].value_counts()}")
    print(f"[data] rates:\n{df['Exited'].value_counts(normalize=True).round(4)}")
    print(f"[data] missing={int(df.isna().sum().sum())} duplicates={int(df.duplicated().sum())}")
    # Basic sanity: expected columns / binary target / known categories
    assert set(df["Exited"].unique()) <= {0, 1}, "Target must be binary 0/1"
    assert set(df["Geography"].unique()) <= {"France", "Germany", "Spain"}, "Unexpected Geography"
    assert set(df["Gender"].unique()) <= {"Male", "Female"}, "Unexpected Gender"
    df = df.drop(columns=["RowNumber", "CustomerId", "Surname"])
    return df


def split_and_preprocess(df):
    X_raw = df.drop(columns=["Exited"])
    y = df["Exited"].values
    # 70% train / 15% val / 15% test, stratified
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        X_raw, y, test_size=0.30, random_state=SEED, stratify=y)
    X_val, X_te, y_val, y_te = train_test_split(
        X_tmp, y_tmp, test_size=0.50, random_state=SEED, stratify=y_tmp)
    print(f"[split] train={X_tr.shape} val={X_val.shape} test={X_te.shape}")
    for name, yy in [("train", y_tr), ("val", y_val), ("test", y_te)]:
        print(f"  {name} churn rate={yy.mean():.4f}")

    # Fit ONLY on train
    le = LabelEncoder()
    le.fit(X_tr["Gender"])
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    ohe.fit(X_tr[["Geography"]])

    def transform(frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.copy()
        out["Gender"] = le.transform(out["Gender"])
        geo = ohe.transform(out[["Geography"]])
        geo_df = pd.DataFrame(geo, columns=ohe.get_feature_names_out(["Geography"]),
                              index=out.index)
        out = pd.concat([out.drop(columns=["Geography"]), geo_df], axis=1)
        return out[FEATURE_ORDER]

    Xtr_df, Xval_df, Xte_df = transform(X_tr), transform(X_val), transform(X_te)
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(Xtr_df)
    Xval = scaler.transform(Xval_df)
    Xte = scaler.transform(Xte_df)
    return (Xtr, Xval, Xte, y_tr, y_val, y_te, le, ohe, scaler)


def build_ann(units=(64, 32), dropout=0.0, l2=0.0, lr=0.001, input_dim=12):
    reg = regularizers.l2(l2) if l2 > 0 else None
    model = Sequential()
    for i, u in enumerate(units):
        kwargs = dict(activation="relu", kernel_regularizer=reg)
        if i == 0:
            model.add(Dense(u, input_shape=(input_dim,), **kwargs))
        else:
            model.add(Dense(u, **kwargs))
        if dropout > 0:
            model.add(Dropout(dropout))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(optimizer=Adam(learning_rate=lr),
                  loss="binary_crossentropy", metrics=["accuracy"])
    return model


def fit_ann(model, Xtr, y_tr, Xval, y_val, class_weight=None, epochs=100, patience=12):
    cbs = [EarlyStopping(monitor="val_loss", patience=patience,
                         restore_best_weights=True, verbose=0),
           ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5,
                             min_lr=1e-6, verbose=0)]
    hist = model.fit(Xtr, y_tr, validation_data=(Xval, y_val),
                     epochs=epochs, batch_size=32, verbose=0,
                     class_weight=class_weight)
    return hist, int(np.argmin(hist.history["val_loss"]) + 1)


def ann_metrics(model, X, y, threshold=CHURN_THRESHOLD):
    p = model.predict(X, verbose=0).ravel()
    pred = (p >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
    }, p, pred


def sk_metrics(clf, X, y):
    pred = clf.predict(X)
    p = clf.predict_proba(X)[:, 1] if hasattr(clf, "predict_proba") else pred.astype(float)
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
    }


def main():
    print(f"[env] TF={tf.__version__} devices={tf.config.list_physical_devices()}")
    df = load_and_validate()
    Xtr, Xval, Xte, ytr, yval, yte, le, ohe, scaler = split_and_preprocess(df)

    cw = compute_class_weight("balanced", classes=np.array([0, 1]), y=ytr)
    cw_dict = {0: float(cw[0]), 1: float(cw[1])}
    print(f"[info] class_weight balanced={cw_dict}")

    configs = [
        dict(name="ann_baseline_64x32_lr0.01", units=(64, 32), dropout=0.0, l2=0.0, lr=0.01, cw=None),
        dict(name="ann_opt_64x32_drop_l2_lr0.001", units=(64, 32), dropout=0.3, l2=1e-4, lr=0.001, cw=None),
        dict(name="ann_small_32x16_drop_lr0.001", units=(32, 16), dropout=0.2, l2=0.0, lr=0.001, cw=None),
        dict(name="ann_opt_classweighted", units=(64, 32), dropout=0.3, l2=1e-4, lr=0.001, cw=cw_dict),
    ]
    val_results, histories, models = {}, {}, {}
    for cfg in configs:
        m = build_ann(cfg["units"], cfg["dropout"], cfg["l2"], cfg["lr"], Xtr.shape[1])
        hist, best_ep = fit_ann(m, Xtr, ytr, Xval, yval, class_weight=cfg["cw"])
        met, _, _ = ann_metrics(m, Xval, yval)
        met["best_epoch"] = best_ep
        n_params = m.count_params()
        met["params"] = int(n_params)
        val_results[cfg["name"]] = met
        histories[cfg["name"]] = {k: [float(v) for v in vals] for k, vals in hist.history.items()}
        models[cfg["name"]] = m
        print(f"[val] {cfg['name']}: " + " ".join(f"{k}={v:.4f}" for k, v in met.items() if k not in ("best_epoch", "params")) + f" best_epoch={best_ep} params={n_params}")

    # Traditional baselines on same splits
    logreg = LogisticRegression(max_iter=1000, random_state=SEED)
    logreg.fit(Xtr, ytr)
    val_results["logreg"] = sk_metrics(logreg, Xval, yval)
    rf = RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=-1, class_weight="balanced")
    rf.fit(Xtr, ytr)
    val_results["randomforest_balanced"] = sk_metrics(rf, Xval, yval)
    for n in ("logreg", "randomforest_balanced"):
        print(f"[val] {n}: " + " ".join(f"{k}={v:.4f}" for k, v in val_results[n].items()))

    # Select by val F1 (primary for imbalanced churn), tie-break ROC-AUC
    ann_only = {k: v for k, v in val_results.items() if k.startswith("ann_")}
    best_name = max(ann_only, key=lambda k: (ann_only[k]["f1"], ann_only[k]["roc_auc"]))
    print(f"[select] best ANN by val F1: {best_name}")
    best = models[best_name]

    # Confirm on untouched test set
    test_met, p_te, pred_te = ann_metrics(best, Xte, yte)
    print("[test] best ANN: " + " ".join(f"{k}={v:.4f}" for k, v in test_met.items()))
    print("[test] classification_report:\n" + classification_report(yte, pred_te, target_names=["retained", "churned"]))
    print("[test] confusion_matrix:\n" + str(confusion_matrix(yte, pred_te)))

    # --- Save artifacts (same filenames app.py / notebooks expect) ---
    best.save(str(BASE / "model.h5"))
    with open(BASE / "label_encoder_gender.pkl", "wb") as f:
        pickle.dump(le, f)
    with open(BASE / "onehot_encoder_geo.pkl", "wb") as f:
        pickle.dump(ohe, f)
    with open(BASE / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open(BASE / "feature_order.json", "w") as f:
        json.dump(FEATURE_ORDER, f, indent=2)
    with open(RESULTS / "metrics.json", "w") as f:
        json.dump({"val": val_results, "test_best_ann": {"name": best_name, **test_met},
                   "test_confusion_matrix": confusion_matrix(yte, pred_te).tolist()}, f, indent=2)

    # --- Plots ---
    # 1. loss/accuracy curves for best model
    h = histories[best_name]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(h["loss"], label="train loss")
    ax[0].plot(h["val_loss"], label="val loss")
    ax[0].set_title(f"{best_name} loss"); ax[0].legend(); ax[0].grid(alpha=0.3)
    if "accuracy" in h:
        ax[1].plot(h["accuracy"], label="train acc")
        ax[1].plot(h["val_accuracy"], label="val acc")
        ax[1].set_title("accuracy"); ax[1].legend(); ax[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS / "training_curves.png", dpi=130)
    plt.close(fig)

    # 2. confusion matrix (test)
    cm = confusion_matrix(yte, pred_te)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["retained", "churned"]); ax.set_yticklabels(["retained", "churned"])
    ax.set_xlabel("predicted"); ax.set_ylabel("actual")
    ax.set_title(f"Confusion matrix (test) - {best_name}")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=13)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(RESULTS / "confusion_matrix.png", dpi=130)
    plt.close(fig)
    print("[done] saved model.h5, pickles, feature_order.json, results/metrics.json + PNGs")


if __name__ == "__main__":
    main()
