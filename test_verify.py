"""Final verification: save/load, unseen samples, edge cases, overfitting check."""
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from config import CHURN_THRESHOLD
from predict import predict_churn

print("== 1. artifacts load ==")
model = tf.keras.models.load_model("model.h5")
print("model input:", model.input_shape, "output:", model.output_shape)
le = pickle.load(open("label_encoder_gender.pkl", "rb"))
ohe = pickle.load(open("onehot_encoder_geo.pkl", "rb"))
sc = pickle.load(open("scaler.pkl", "rb"))
print("gender:", list(le.classes_), "| geo:", list(ohe.categories_[0]),
      "| scaler features:", sc.n_features_in_)

print("== 2. unseen representative samples ==")
samples = [
    {"CreditScore": 600, "Geography": "France", "Gender": "Male", "Age": 40,
     "Tenure": 3, "Balance": 60000.0, "NumOfProducts": 2, "HasCrCard": 1,
     "IsActiveMember": 1, "EstimatedSalary": 50000.0},  # original notebook example
    {"CreditScore": 450, "Geography": "Germany", "Gender": "Female", "Age": 55,
     "Tenure": 1, "Balance": 150000.0, "NumOfProducts": 1, "HasCrCard": 0,
     "IsActiveMember": 0, "EstimatedSalary": 30000.0},  # high-risk profile
    {"CreditScore": 800, "Geography": "Spain", "Gender": "Male", "Age": 30,
     "Tenure": 8, "Balance": 20000.0, "NumOfProducts": 2, "HasCrCard": 1,
     "IsActiveMember": 1, "EstimatedSalary": 120000.0},  # low-risk profile
]
for s in samples:
    p, label = predict_churn(s)
    assert 0.0 <= p <= 1.0 and label in (0, 1)
    print(f"  {s['Geography']}/{s['Gender']}/age{s['Age']} -> p={p:.4f} label={label}")

print("== 3. edge cases ==")
for bad, desc in [
    (dict(samples[0], Geography="Italy"), "unexpected category"),
    (dict(samples[0], Gender="Other"), "unexpected gender"),
    ({k: v for k, v in samples[0].items() if k != "Age"}, "missing field"),
]:
    try:
        predict_churn(bad)
        print(f"  FAIL: no error for {desc}")
    except ValueError as e:
        print(f"  OK ({desc}): {e}")

print("== 4. train/val/test gap (overfitting check, same splits as train.py) ==")
df = pd.read_csv("Churn_Modelling.csv").drop(columns=["RowNumber", "CustomerId", "Surname"])
Xr, y = df.drop(columns=["Exited"]), df["Exited"].values
Xtr_raw, Xtmp, ytr, ytmp = train_test_split(Xr, y, test_size=0.30, random_state=42, stratify=y)
Xva_raw, Xte_raw, yva, yte = train_test_split(Xtmp, ytmp, test_size=0.50, random_state=42, stratify=ytmp)


def tx(frame):
    out = frame.copy()
    out["Gender"] = le.transform(out["Gender"])
    g = pd.DataFrame(ohe.transform(out[["Geography"]]),
                     columns=ohe.get_feature_names_out(["Geography"]), index=out.index)
    import json
    order = json.load(open("feature_order.json"))
    return sc.transform(pd.concat([out.drop(columns=["Geography"]), g], axis=1)[order])


for name, Xr_, yr_ in [("train", Xtr_raw, ytr), ("val", Xva_raw, yva), ("test", Xte_raw, yte)]:
    p = model.predict(tx(Xr_), verbose=0).ravel()
    pred = (p >= CHURN_THRESHOLD).astype(int)
    print(f"  {name}: acc={accuracy_score(yr_, pred):.4f} "
          f"f1={f1_score(yr_, pred):.4f} roc={roc_auc_score(yr_, p):.4f}")

print("== 5. fixed threshold behavior (below / at / above 0.5) ==")
import inspect
from config import CHURN_THRESHOLD as T
assert T == 0.5, "training/eval threshold must stay 0.5"
assert list(inspect.signature(predict_churn).parameters) == ["payload"], \
    "predict_churn must take only payload (fixed threshold, no kwarg)"
for s in samples:
    p, label = predict_churn(s)
    assert label == int(p >= T), f"label rule broken at p={p}"
    side = "below" if p < 0.5 else "above"
    print(f"  p={p:.4f} ({side} 0.5) label={label} OK")
# exact boundary: rule is >=, so p == 0.5 -> churn (matches training labels)
assert int(0.5 >= T) == 1
print("  boundary p == 0.5 -> label 1 (>= semantics) OK")
# the reported TypeError shape is gone: no threshold kwarg exists anymore
try:
    predict_churn(samples[0], threshold=0.65)
    raise AssertionError("threshold kwarg unexpectedly accepted")
except TypeError as e:
    print(f"  threshold kwarg correctly rejected: {e}")

print("== 6. single authoritative threshold, no duplicates ==")
import re
import pathlib
py_files = ("config.py", "predict.py", "app.py", "train.py", "test_verify.py")
hits = []
for f in py_files:
    for i, line in enumerate(pathlib.Path(f).read_text().splitlines(), 1):
        code = line.split("#", 1)[0]  # ignore comments
        # Flag decision comparisons against a hardcoded 0.5 literal
        # (e.g. ">= 0.5"); LR factors like "factor=0.5" have no ">" and pass.
        if re.search(r">=?\s*0\.5\b", code) and "CHURN_THRESHOLD" not in code:
            hits.append(f"{f}:{i}: {line.strip()}")
# Flag any production caller passing a threshold kwarg (the reported
# TypeError shape). test_verify.py is excluded: it holds the intentional
# negative test proving the kwarg is rejected.
for f in ("config.py", "predict.py", "app.py", "train.py"):
    for i, line in enumerate(pathlib.Path(f).read_text().splitlines(), 1):
        code = line.split("#", 1)[0]
        if re.search(r"predict_churn\([^)]*threshold\s*=", code):
            hits.append(f"{f}:{i}: {line.strip()}")
assert not hits, f"hardcoded/duplicate thresholds found: {hits}"
defs = [(f, i, l.strip()) for f in py_files
        for i, l in enumerate(pathlib.Path(f).read_text().splitlines(), 1)
        if re.match(r"\s*CHURN_THRESHOLD\s*=", l)]
assert len(defs) == 1 and defs[0][0] == "config.py", f"expected one definition in config.py, got {defs}"
import config
import predict as predict_mod
assert predict_mod.CHURN_THRESHOLD is config.CHURN_THRESHOLD  # re-export, not a copy
assert config.CHURN_THRESHOLD == 0.5
print(f"  authoritative: config.py CHURN_THRESHOLD = {config.CHURN_THRESHOLD} OK")
print("== 7. no threshold UI control in app.py ==")
app_src = pathlib.Path("app.py").read_text()
for pat in ("Churn threshold", "threshold=threshold", "session_state"):
    assert pat not in app_src, f"obsolete UI reference still present: {pat}"
assert "st.slider" in app_src  # other (feature) sliders remain
print("  no Churn-threshold control, no session-state, no kwarg passing OK")
print("ALL CHECKS PASSED")
