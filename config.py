"""Shared project configuration — single source of truth.

CHURN_THRESHOLD is the ONE authoritative churn decision threshold:
    label = 1 iff predicted churn probability >= CHURN_THRESHOLD.

It is defined EXACTLY ONCE, here. Every other module (predict.py,
train.py, test_verify.py, app.py) must import it from this file —
never redefine it, never hardcode 0.5 as a decision rule.

Value rationale: the ANN was trained and all reported metrics
(accuracy/precision/recall/F1/ROC-AUC) were evaluated with labels
computed at 0.5. Do NOT change this value to "fix" a display;
it encodes intended model behavior.
"""

CHURN_THRESHOLD = 0.5
