"""
Customer Churn - Streamlit Prediction App
=========================================
Loads the model trained by `train.py` and exposes a simple form for
predicting churn risk on a single customer.

Run locally:
    streamlit run streamlit_app.py
"""
import joblib
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Telco Customer Churn Predictor",
    page_icon="📊",
    layout="centered",
)

st.title("📊 Telco Customer Churn Predictor")
st.write(
    "Predicts whether a telecom customer is likely to churn, based on a "
    "Random Forest model trained on the IBM Telco Customer Churn dataset. "
    "Adjust the inputs below and click **Predict**."
)
st.write(
    "**Model:** Random Forest + SMOTEENN resampling on training data &nbsp;|&nbsp; "
    "**Test recall on churn class:** 81% &nbsp;|&nbsp; "
    "**Test accuracy:** 76%"
)
st.divider()


# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model():
    model = joblib.load("churn_model.pkl")
    columns = joblib.load("feature_columns.pkl")
    return model, columns


model, FEATURE_COLUMNS = load_model()


# ---------------------------------------------------------------------------
# Input form (focuses on the top features by importance)
# ---------------------------------------------------------------------------
st.subheader("Customer details")

col1, col2 = st.columns(2)

with col1:
    contract = st.selectbox(
        "Contract type",
        ["Month-to-month", "One year", "Two year"],
        help="Top predictor: month-to-month customers churn most.",
    )
    tenure = st.slider("Tenure (months with the company)", 0, 72, 12)
    internet_service = st.selectbox(
        "Internet service",
        ["DSL", "Fiber optic", "No"],
    )
    payment_method = st.selectbox(
        "Payment method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        ],
    )

with col2:
    monthly_charges = st.slider("Monthly charges (USD)", 0.0, 120.0, 70.0, step=0.5)
    total_charges = st.slider(
        "Total charges (USD)",
        0.0,
        9000.0,
        float(monthly_charges * max(tenure, 1)),
        step=10.0,
    )
    tech_support = st.selectbox("Tech support add-on", ["No", "Yes"])
    online_security = st.selectbox("Online security add-on", ["No", "Yes"])

with st.expander("Additional details (optional, defaults are typical values)"):
    col3, col4 = st.columns(2)
    with col3:
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior = st.selectbox("Senior citizen", ["No", "Yes"])
        partner = st.selectbox("Has partner", ["No", "Yes"])
        dependents = st.selectbox("Has dependents", ["No", "Yes"])
        paperless = st.selectbox("Paperless billing", ["Yes", "No"])
    with col4:
        phone_service = st.selectbox("Phone service", ["Yes", "No"])
        multiple_lines = st.selectbox("Multiple lines", ["No", "Yes"])
        online_backup = st.selectbox("Online backup add-on", ["No", "Yes"])
        device_protection = st.selectbox("Device protection add-on", ["No", "Yes"])
        streaming_tv = st.selectbox("Streaming TV", ["No", "Yes"])
        streaming_movies = st.selectbox("Streaming movies", ["No", "Yes"])

st.divider()


# ---------------------------------------------------------------------------
# Build the feature row that matches the training preprocessing
# ---------------------------------------------------------------------------
def yes_no(value: str) -> int:
    return 1 if value == "Yes" else 0


def build_feature_row() -> pd.DataFrame:
    row = {col: 0 for col in FEATURE_COLUMNS}

    # Binary columns
    row["gender"] = 1 if gender == "Male" else 0
    row["SeniorCitizen"] = yes_no(senior)
    row["Partner"] = yes_no(partner)
    row["Dependents"] = yes_no(dependents)
    row["tenure"] = tenure
    row["PhoneService"] = yes_no(phone_service)
    row["MultipleLines"] = yes_no(multiple_lines)
    row["OnlineSecurity"] = yes_no(online_security)
    row["OnlineBackup"] = yes_no(online_backup)
    row["DeviceProtection"] = yes_no(device_protection)
    row["TechSupport"] = yes_no(tech_support)
    row["StreamingTV"] = yes_no(streaming_tv)
    row["StreamingMovies"] = yes_no(streaming_movies)
    row["PaperlessBilling"] = yes_no(paperless)
    row["MonthlyCharges"] = monthly_charges
    row["TotalCharges"] = total_charges

    # One-hot columns: set the matching value to 1
    row[f"InternetService_{internet_service}"] = 1
    row[f"Contract_{contract}"] = 1
    row[f"PaymentMethod_{payment_method}"] = 1

    return pd.DataFrame([row])[FEATURE_COLUMNS]


# ---------------------------------------------------------------------------
# Predict
# ---------------------------------------------------------------------------
if st.button("Predict churn risk", type="primary", use_container_width=True):
    features = build_feature_row()
    probabilities = model.predict_proba(features)[0]
    churn_prob = probabilities[1]
    prediction = int(churn_prob >= 0.5)

    st.subheader("Prediction")

    if prediction == 1:
        st.error(f"⚠️ **Likely to churn** (probability {churn_prob:.1%})")
        st.write(
            "Suggested retention actions: longer-term contract, "
            "tech support add-on, or automatic payment method."
        )
    else:
        st.success(f"✅ **Likely to stay** (churn probability {churn_prob:.1%})")
        st.write("No immediate retention action needed.")

    with st.expander("Probability breakdown"):
        prob_df = pd.DataFrame(
            {"Outcome": ["Stay", "Churn"], "Probability": probabilities}
        )
        st.bar_chart(prob_df.set_index("Outcome"))

st.divider()
st.caption("Trained on the IBM Telco Customer Churn dataset (Kaggle).")
