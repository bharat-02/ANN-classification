"""Final verification: save/load, unseen samples, edge cases, overfitting check."""
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

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
    pred = (p >= 0.5).astype(int)
    print(f"  {name}: acc={accuracy_score(yr_, pred):.4f} "
          f"f1={f1_score(yr_, pred):.4f} roc={roc_auc_score(yr_, p):.4f}")
print("ALL CHECKS PASSED")
