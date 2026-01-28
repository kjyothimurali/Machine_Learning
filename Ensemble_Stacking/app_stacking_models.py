# ===============================
# app.py
# ===============================

import streamlit as st
import numpy as np
import pandas as pd
import joblib

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(
    page_title="Smart Loan Approval System",
    layout="wide"
)

# -------------------------------
# Load Models & Feature Names
# -------------------------------
@st.cache_resource
def load_models():
    lr = joblib.load("lr_model.pkl")
    dt = joblib.load("dt_model.pkl")
    rf = joblib.load("rf_model.pkl")
    stack = joblib.load("stacking_model.pkl")
    features = joblib.load("feature_names.pkl")
    return lr, dt, rf, stack, features

lr_model, dt_model, rf_model, stacking_model, feature_names = load_models()

# -------------------------------
# Title & Description
# -------------------------------
st.title("🎯 Smart Loan Approval System – Stacking Model")

st.markdown("""
This system predicts **loan approval** using a **Stacking Ensemble Machine Learning model**
by combining multiple base models for better decision making.
""")

# -------------------------------
# Sidebar Inputs
# -------------------------------
st.sidebar.header("📋 Applicant Details")

ApplicantIncome = st.sidebar.number_input("Applicant Income", min_value=0)
CoapplicantIncome = st.sidebar.number_input("Co-Applicant Income", min_value=0.0)
LoanAmount = st.sidebar.number_input("Loan Amount", min_value=0.0)
Loan_Amount_Term = st.sidebar.number_input("Loan Amount Term (Months)", min_value=0.0)

Credit_History = st.sidebar.radio("Credit History", ["Yes", "No"])
Self_Employed = st.sidebar.selectbox("Employment Status", ["Salaried", "Self-Employed"])
Property_Area = st.sidebar.selectbox("Property Area", ["Urban", "Semi-Urban", "Rural"])

# -------------------------------
# Encoding Inputs
# -------------------------------
Credit_History = 1 if Credit_History == "Yes" else 0
Self_Employed = 1 if Self_Employed == "Self-Employed" else 0

# Property Area Encoding
prop_urban = 1 if Property_Area == "Urban" else 0
prop_semiurban = 1 if Property_Area == "Semi-Urban" else 0

# -------------------------------
# Create Input DataFrame
# -------------------------------
input_dict = {
    'ApplicantIncome': ApplicantIncome,
    'CoapplicantIncome': CoapplicantIncome,
    'LoanAmount': LoanAmount,
    'Loan_Amount_Term': Loan_Amount_Term,
    'Credit_History': Credit_History,
    'Self_Employed': Self_Employed,
    'Property_Area_Semiurban': prop_semiurban,
    'Property_Area_Urban': prop_urban
}

input_df = pd.DataFrame([input_dict])

# Align with training features
input_df = input_df.reindex(columns=feature_names, fill_value=0)

# -------------------------------
# Model Architecture Display
# -------------------------------
st.subheader("🧩 Model Architecture")

st.markdown("""
**Base Models**
- Logistic Regression  
- Decision Tree  
- Random Forest  

**Meta Model**
- Logistic Regression  
""")

# -------------------------------
# Prediction Button
# -------------------------------
if st.button("🔘 Check Loan Eligibility (Stacking Model)"):

    lr_pred = lr_model.predict(input_df)[0]
    dt_pred = dt_model.predict(input_df)[0]
    rf_pred = rf_model.predict(input_df)[0]

    final_pred = stacking_model.predict(input_df)[0]
    confidence = stacking_model.predict_proba(input_df)[0][1] * 100

    # ---------------------------
    # Results
    # ---------------------------
    st.subheader("📊 Prediction Results")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Base Model Predictions")
        st.write("Logistic Regression →", "Approved" if lr_pred else "Rejected")
        st.write("Decision Tree →", "Approved" if dt_pred else "Rejected")
        st.write("Random Forest →", "Approved" if rf_pred else "Rejected")

    with col2:
        st.markdown("### Final Stacking Decision")
        if final_pred == 1:
            st.success("✅ Loan Approved")
        else:
            st.error("❌ Loan Rejected")

        st.metric("Confidence Score", f"{confidence:.2f}%")

    # ---------------------------
    # Business Explanation
    # ---------------------------
    st.subheader("💼 Business Explanation")

    if final_pred == 1:
        st.write("""
        Based on the applicant’s income, credit history, and combined predictions
        from multiple machine learning models, the applicant is likely to repay the loan.
        Therefore, the system **approves the loan**.
        """)
    else:
        st.write("""
        Considering the applicant’s financial details and the combined assessment
        from multiple models, the risk of repayment is high.
        Therefore, the system **rejects the loan**.
        """)
