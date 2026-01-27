import streamlit as st
import numpy as np
import pandas as pd
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# -----------------------------
# App Config
# -----------------------------
st.set_page_config(page_title="Customer Risk Prediction System", layout="wide")

# -----------------------------
# Custom CSS
# -----------------------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #f5f7fa, #e4ecf7);
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1f4fd8, #3f7cff);
}
[data-testid="stSidebar"] * {
    color: white !important;
}
.main-title {
    font-size: 42px;
    font-weight: 700;
    color: #1f4fd8;
}
.card {
    background: white;
    padding: 25px;
    border-radius: 15px;
    box-shadow: 0 10px 25px rgba(0,0,0,0.1);
    margin-bottom: 20px;
}
.high-risk {
    color: #ff4b4b;
    font-size: 32px;
    font-weight: bold;
}
.low-risk {
    color: #2ecc71;
    font-size: 32px;
    font-weight: bold;
}
div.stButton > button {
    background-color: #1f4fd8;
    color: white;
    padding: 12px 25px;
    border-radius: 12px;
    font-size: 18px;
    font-weight: bold;
}
div.stButton > button:hover {
    background-color: #163bbd;
    transform: scale(1.05);
}
</style>
""", unsafe_allow_html=True)
st.markdown("""
<style>
/* ===== INPUT TEXT VISIBILITY FIX ===== */

/* Text inside input boxes */
input, textarea {
    color: #000000 !important;
    background-color: #ffffff !important;
}

/* Number input text */
div[data-baseweb="input"] input {
    color: #000000 !important;
    background-color: #ffffff !important;
}

/* Selectbox selected value */
div[data-baseweb="select"] span {
    color: #000000 !important;
}

/* Slider value text */
div[data-testid="stSlider"] span {
    color: #000000 !important;
    font-weight: 600;
}

/* Placeholder text */
input::placeholder {
    color: #666666 !important;
}

/* Labels clarity */
label {
    font-weight: 600 !important;
}

/* Sidebar input boxes */
[data-testid="stSidebar"] input,
[data-testid="stSidebar"] textarea,
[data-testid="stSidebar"] div[data-baseweb="select"] {
    background-color: #ffffff !important;
    border-radius: 8px;
}
</style>
""", unsafe_allow_html=True)
st.markdown("""
<style>
/* ============================= */
/* LIGHT & CLEAN +/- BUTTON STYLE */
/* ============================= */

[data-testid="stNumberInput"] {
    background-color: #ffffff !important;
    border-radius: 8px;
}

/* + / - buttons */
[data-testid="stNumberInput"] button {
    background-color: #eef2ff !important;   /* light lavender-blue */
    color: #1f3fbf !important;              /* soft dark blue */
    border: 1px solid #d6ddff !important;
    min-width: 32px !important;
    height: 32px !important;
    border-radius: 6px !important;
}

/* Icons inside buttons */
[data-testid="stNumberInput"] button svg {
    fill: #1f3fbf !important;
}

/* Hover effect (slightly darker, still light) */
[data-testid="stNumberInput"] button:hover {
    background-color: #dde5ff !important;
}

/* Input text */
[data-testid="stNumberInput"] input {
    color: #000000 !important;
    background-color: #ffffff !important;
}
</style>
""", unsafe_allow_html=True)


# -----------------------------
# Header
# -----------------------------
st.markdown("<div class='main-title'>📊 Customer Risk Prediction System (KNN)</div>", unsafe_allow_html=True)
st.markdown("This system predicts customer risk by comparing them with similar customers.")

# -----------------------------
# Sidebar Inputs
# -----------------------------
st.sidebar.header("Customer Details")

age = st.sidebar.slider("Age", 18, 70, 30)
income = st.sidebar.number_input("Annual Income", 10000, 200000, 50000)
loan_amount = st.sidebar.number_input("Loan Amount", 1000, 100000, 20000)
credit_history_length = st.sidebar.slider("Credit History Length (Years)", 0, 30, 5)
k_value = st.sidebar.slider("K Value (Neighbors)", 1, 15, 5)

# -----------------------------
# Load Dataset
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "credit_risk_dataset.csv")
df = pd.read_csv(DATA_PATH)

# Dataset Preview
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader("📁 Dataset Preview")
st.dataframe(df.head())
st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------
# Feature Selection
# -----------------------------
X = df[["person_age", "person_income", "loan_amnt", "cb_person_cred_hist_length"]]
y = df["loan_status"]

# -----------------------------
# Scaling & Model
# -----------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

user_input = np.array([[age, income, loan_amount, credit_history_length]])
user_input_scaled = scaler.transform(user_input)

knn = KNeighborsClassifier(n_neighbors=k_value)
knn.fit(X_scaled, y)

# -----------------------------
# Prediction Button (ONLY ONE)
# -----------------------------
if st.button("🚀 Predict Customer Risk", key="predict_btn"):

    prediction = knn.predict(user_input_scaled)[0]
    neighbors = knn.kneighbors(user_input_scaled, return_distance=False)[0]
    neighbor_classes = y.iloc[neighbors]

    majority_class = neighbor_classes.mode()[0]
    majority_label = "High Risk" if majority_class == 1 else "Low Risk"

    neighbor_table = df.iloc[neighbors].copy()
    neighbor_table["Risk Label"] = neighbor_table["loan_status"].map({1: "High Risk", 0: "Low Risk"})
    neighbor_table.drop("loan_status", axis=1, inplace=True)

    # -----------------------------
    # Prediction Result
    # -----------------------------
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("🔮 Prediction Result")

    if prediction == 1:
        st.markdown("<div class='high-risk'>🔴 High Risk Customer</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='low-risk'>🟢 Low Risk Customer</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # -----------------------------
    # KNN Explanation
    # -----------------------------
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("📌 KNN Explanation")
    st.write(f"**🔢 K Value:** {k_value}")
    st.write(f"**👥 Majority Class:** {majority_label}")
    st.dataframe(neighbor_table)
    st.markdown("</div>", unsafe_allow_html=True)

    # -----------------------------
    # Business Insight
    # -----------------------------
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("💡 Business Insight")
    st.success(
        "This decision is based on similarity with nearby customers in feature space. "
        "Customers with similar age, income, loan amount, and credit history tend to "
        "exhibit similar credit risk behavior."
    )
    st.markdown("</div>", unsafe_allow_html=True)
