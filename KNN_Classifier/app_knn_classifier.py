import streamlit as st
import numpy as np
import pandas as pd
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# -------------------------------------------------
# Page Config
# -------------------------------------------------
st.set_page_config(
    page_title="Customer Risk Prediction (KNN)",
    layout="wide"
)

# -------------------------------------------------
# Simple, Safe CSS (NO label breaking)
# -------------------------------------------------
st.markdown("""
<style>
.stApp {
    background-color: #f6f8fc;
}

.sidebar-title {
    font-size: 22px;
    font-weight: 700;
    color: #ffffff;
    margin-bottom: 10px;
}

.card {
    background-color: #ffffff;
    padding: 20px;
    border-radius: 12px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    margin-bottom: 20px;
}

.result-high {
    color: #d90429;
    font-size: 28px;
    font-weight: 700;
}

.result-low {
    color: #2b9348;
    font-size: 28px;
    font-weight: 700;
}

div.stButton > button {
    background-color: #4f46e5;
    color: white;
    font-size: 18px;
    font-weight: 600;
    padding: 10px 24px;
    border-radius: 8px;
}
div.stButton > button:hover {
    background-color: #4338ca;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# App Header
# -------------------------------------------------
st.markdown("## 📊 Customer Risk Prediction System (KNN)")
st.markdown(
    "This system predicts customer risk by comparing a customer with similar customers "
    "using the **K-Nearest Neighbors (KNN)** algorithm."
)

# -------------------------------------------------
# Sidebar – CLEAR LABELS (NO CSS tricks)
# -------------------------------------------------
with st.sidebar:
    st.markdown("<div class='sidebar-title'>👤 Customer Details</div>", unsafe_allow_html=True)

    st.markdown("**Age**")
    age = st.slider("Age", 18, 70, 30)

    st.markdown("**Annual Income**")
    income = st.number_input("Annual Income", min_value=10000, max_value=200000, value=50000)

    st.markdown("**Loan Amount**")
    loan_amount = st.number_input("Loan Amount", min_value=1000, max_value=100000, value=20000)

    st.markdown("**Credit History Length (Years)**")
    credit_history_length = st.slider("Credit History Length", 0, 30, 5)

    st.markdown("**K Value (Number of Neighbors)**")
    k_value = st.slider("K Value", 1, 15, 5)

# -------------------------------------------------
# Load Dataset
# -------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "credit_risk_dataset.csv")

df = pd.read_csv(DATA_PATH)

# -------------------------------------------------
# Dataset Preview
# -------------------------------------------------
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader("📁 Dataset Preview")
st.dataframe(df.head())
st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------------------------
# Feature Selection
# -------------------------------------------------
X = df[
    ["person_age", "person_income", "loan_amnt", "cb_person_cred_hist_length"]
]
y = df["loan_status"]

# -------------------------------------------------
# Scaling & Model
# -------------------------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

user_input = np.array([[age, income, loan_amount, credit_history_length]])
user_input_scaled = scaler.transform(user_input)

knn = KNeighborsClassifier(n_neighbors=k_value)
knn.fit(X_scaled, y)

# -------------------------------------------------
# Prediction Button
# -------------------------------------------------
if st.button("🔮 Predict Customer Risk", key="predict_btn"):

    prediction = knn.predict(user_input_scaled)[0]
    neighbors = knn.kneighbors(user_input_scaled, return_distance=False)[0]
    neighbor_classes = y.iloc[neighbors]

    majority_class = neighbor_classes.mode()[0]
    majority_label = "High Risk" if majority_class == 1 else "Low Risk"

    neighbor_table = df.iloc[neighbors].copy()
    neighbor_table["Risk Label"] = neighbor_table["loan_status"].map(
        {1: "High Risk", 0: "Low Risk"}
    )
    neighbor_table.drop("loan_status", axis=1, inplace=True)

    # -------------------------------------------------
    # Prediction Result
    # -------------------------------------------------
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("🔮 Prediction Result")

    if prediction == 1:
        st.markdown("<div class='result-high'>🔴 High Risk Customer</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='result-low'>🟢 Low Risk Customer</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # -------------------------------------------------
    # KNN Explanation
    # -------------------------------------------------
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("📌 KNN Explanation")
    st.write(f"**K Value:** {k_value}")
    st.write(f"**Majority Class Among Neighbors:** {majority_label}")
    st.dataframe(neighbor_table)
    st.markdown("</div>", unsafe_allow_html=True)

    # -------------------------------------------------
    # Business Insight
    # -------------------------------------------------
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("💡 Business Insight")
    st.info(
        "The prediction is based on similarity with nearby customers in feature space. "
        "Customers with similar age, income, loan amount, and credit history tend to "
        "exhibit similar credit risk behavior."
    )
    st.markdown("</div>", unsafe_allow_html=True)
