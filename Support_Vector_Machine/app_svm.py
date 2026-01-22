import streamlit as st
import pandas as pd
import numpy as np

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# ------------------------------------------------
# Page Configuration
# ------------------------------------------------
st.set_page_config(
    page_title="Smart Loan Approval System",
    page_icon="💳",
    layout="centered"
)

# ------------------------------------------------
# Title & Description
# ------------------------------------------------
st.title("💳 Smart Loan Approval System")

st.write(
    """
    This system uses **Support Vector Machine (SVM)** algorithms  
    to predict whether a loan will be **Approved or Rejected**
    based on applicant financial details.
    """
)

# ------------------------------------------------
# Load Dataset
# ------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("train_u6lujuX_CVtuZ9i.csv")

df = load_data()

# ------------------------------------------------
# Handle Missing Values
# ------------------------------------------------
num_cols = df.select_dtypes(include=['int64', 'float64']).columns
cat_cols = df.select_dtypes(include=['object']).columns

for col in num_cols:
    df[col].fillna(df[col].mean(), inplace=True)

for col in cat_cols:
    df[col].fillna(df[col].mode()[0], inplace=True)

# ------------------------------------------------
# Encode Target
# ------------------------------------------------
y = df['Loan_Status'].replace({'Y': 1, 'N': 0}).astype(int)
X = df.drop('Loan_Status', axis=1)

# One-hot encoding
X = pd.get_dummies(X, drop_first=True)

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ------------------------------------------------
# Sidebar – Input Section
# ------------------------------------------------
st.sidebar.header("📋 Applicant Details")

app_income = st.sidebar.number_input("Applicant Income", min_value=0)
loan_amount = st.sidebar.number_input("Loan Amount", min_value=0)
credit_history = st.sidebar.selectbox("Credit History", ["Yes", "No"])
employment = st.sidebar.selectbox("Employment Status", ["Employed", "Self Employed"])
property_area = st.sidebar.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

# Convert inputs to dataframe
input_data = {
    "ApplicantIncome": app_income,
    "LoanAmount": loan_amount,
    "Credit_History": 1.0 if credit_history == "Yes" else 0.0,
    "Self_Employed_Yes": 1 if employment == "Self Employed" else 0,
    "Property_Area_Semiurban": 1 if property_area == "Semiurban" else 0,
    "Property_Area_Urban": 1 if property_area == "Urban" else 0
}

input_df = pd.DataFrame([input_data])
input_df = input_df.reindex(columns=X.columns, fill_value=0)
input_scaled = scaler.transform(input_df)

# ------------------------------------------------
# Model Selection
# ------------------------------------------------
st.subheader("🧠 Select SVM Kernel")

kernel_choice = st.radio(
    "Choose the kernel type:",
    ["Linear SVM", "Polynomial SVM", "RBF SVM"]
)

if kernel_choice == "Linear SVM":
    kernel = "linear"
elif kernel_choice == "Polynomial SVM":
    kernel = "poly"
else:
    kernel = "rbf"

# ------------------------------------------------
# Train Model
# ------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

model = SVC(kernel=kernel,degree=5, probability=True,class_weight='balanced')
model.fit(X_train, y_train)

# ------------------------------------------------
# Prediction Button
# ------------------------------------------------
st.markdown("---")

if st.button("🔍 Check Loan Eligibility"):
    
    prob = model.predict_proba(input_scaled)[0][1]
    prediction = 1 if prob >= 0.6 else 0
    confidence = model.predict_proba(input_scaled)[0].max() * 100

    # Output Section
    if prediction == 1:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Rejected")

    st.info(f"🔧 Kernel Used: **{kernel_choice}**")
    st.info(f"📊 Confidence Score: **{confidence:.2f}%**")

    # Business Explanation
    st.markdown("### 📌 Business Explanation")
    if prediction == 1:
        st.write(
            "Based on the applicant’s **credit history and income pattern**, "
            "the system predicts that the applicant is **likely to repay the loan**."
        )
    else:
        st.write(
            "Based on the applicant’s **credit history and income pattern**, "
            "the system predicts that the applicant is **unlikely to repay the loan**."
        )

# ------------------------------------------------
# Footer
# ------------------------------------------------
st.markdown("---")
st.caption("Model: Support Vector Machine (SVM)")

