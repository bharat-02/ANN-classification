"""Customer Churn Prediction — Streamlit app (fixed).

Fixes vs original app.py:
- model/pickles loaded once via @st.cache_resource (was reloaded + predicted on every rerun).
- Prediction only runs on button press (was running at import/slider time, incl. with empty inputs).
- Shared predict.predict_churn pipeline => identical preprocessing to training.
- Input validation + graceful errors for unknown categories.
- Threshold slider + probability display.
"""
import streamlit as st

from predict import predict_churn

st.set_page_config(page_title="Customer Churn Prediction", layout="centered")
st.title("Customer Churn Prediction")
st.caption("ANN binary classifier — same preprocessing as training (see train.py).")

with st.form("churn_form"):
    col1, col2 = st.columns(2)
    with col1:
        geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
        gender = st.selectbox("Gender", ["Female", "Male"])
        age = st.slider("Age", 18, 92, 35)
        credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=650)
        balance = st.number_input("Balance", min_value=0.0, value=50000.0)
    with col2:
        estimated_salary = st.number_input("Estimated Salary", min_value=0.0, value=80000.0)
        tenure = st.slider("Tenure", 0, 10, 3)
        num_of_products = st.slider("Number of Products", 1, 4, 1)
        has_cr_card = st.selectbox("Has Credit Card", [0, 1])
        is_active_member = st.selectbox("Is Active Member", [0, 1])
    threshold = st.slider("Churn threshold", 0.1, 0.9, 0.5, 0.05)
    submitted = st.form_submit_button("Predict")

if submitted:
    payload = {
        "CreditScore": credit_score, "Geography": geography, "Gender": gender,
        "Age": age, "Tenure": tenure, "Balance": balance,
        "NumOfProducts": num_of_products, "HasCrCard": int(has_cr_card),
        "IsActiveMember": int(is_active_member), "EstimatedSalary": estimated_salary,
    }
    try:
        proba, _ = predict_churn(payload)
        st.metric("Churn probability", f"{proba:.3f}")
        st.progress(min(max(proba, 0.0), 1.0))
        if proba >= threshold:
            st.error("The customer is likely to churn.")
        else:
            st.success("The customer is not likely to churn.")
    except ValueError as e:
        st.error(f"Invalid input: {e}")
    except FileNotFoundError as e:
        st.error(f"Model artifacts missing. Run `python train.py` first. ({e})")
