import streamlit as st
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt

# =========================
# Page Configuration
# =========================
st.set_page_config(
    page_title="Multiple Linear Regression App",
    layout="centered"
)

# =========================
# Load CSS
# =========================
def load_css(file):
    with open(file) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("styles.css")

# =========================
# App Header
# =========================
st.markdown("""
<div class="card">
    <h1>Multiple Linear Regression App</h1>
    <p><b>Predicting Tip Amount</b> using <b>Total Bill</b> and <b>Table Size</b></p>
</div>
""", unsafe_allow_html=True)

# =========================
# Load Dataset
# =========================
@st.cache_data
def load_data():
    return sns.load_dataset("tips")

df = load_data()

# =========================
# Dataset Preview
# =========================
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Dataset Preview")
st.dataframe(df[['total_bill', 'size', 'tip']].head())
st.markdown('</div>', unsafe_allow_html=True)

# =========================
# Prepare Data
# =========================
X = df[['total_bill', 'size']]
y = df['tip']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# Scaling
# =========================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =========================
# Train Model
# =========================
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# =========================
# Predictions
# =========================
y_pred = model.predict(X_test_scaled)

# =========================
# Evaluation Metrics
# =========================
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
adjusted_r2 = 1 - (1 - r2) * (len(y_test) - 1) / (len(y_test) - X_test.shape[1] - 1)

# =========================
# Visualization
# =========================
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Actual vs Predicted Tip Amount (Test Data)")

fig, ax = plt.subplots(figsize=(8, 6))

ax.scatter(y_test, y_pred, alpha=0.7)
ax.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    color='red',
    linewidth=2
)

ax.set_xlabel("Actual Tip Amount ($)")
ax.set_ylabel("Predicted Tip Amount ($)")
ax.set_title("Actual vs Predicted Tips")

st.pyplot(fig)
st.markdown('</div>', unsafe_allow_html=True)

# =========================
# Model Evaluation
# =========================
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Model Evaluation")

c1, c2, c3, c4 = st.columns(4)
c1.metric("MSE", f"{mse:.2f}")
c2.metric("R²", f"{r2:.2f}")
c3.metric("MAE", f"{mae:.2f}")
c4.metric("Adjusted R²", f"{adjusted_r2:.2f}")

st.markdown('</div>', unsafe_allow_html=True)

# =========================
# Model Interpretation
# =========================
st.markdown(f"""
<div class="card">
    <h3>Model Interpretation</h3>
    <p><b>Coefficient (Total Bill):</b> {model.coef_[0]:.2f}</p>
    <p><b>Coefficient (Table Size):</b> {model.coef_[1]:.2f}</p>
    <p><b>Intercept:</b> {model.intercept_:.2f}</p>
</div>
""", unsafe_allow_html=True)

# =========================
# Prediction (User Input)
# =========================
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Predict Tip Amount")

bill_amount = st.slider(
    "Select Total Bill Amount ($):",
    min_value=float(df['total_bill'].min()),
    max_value=float(df['total_bill'].max()),
    value=30.0,
    step=1.0
)

table_size = st.slider(
    "Select Table Size (Number of People):",
    min_value=int(df['size'].min()),
    max_value=int(df['size'].max()),
    value=2,
    step=1
)

# Create DataFrame with feature names (IMPORTANT)
input_df = pd.DataFrame(
    [[bill_amount, table_size]],
    columns=['total_bill', 'size']
)

# Scale & Predict
input_scaled = scaler.transform(input_df)
predicted_tip = model.predict(input_scaled)[0]

st.success(f"Predicted Tip Amount: ${predicted_tip:.2f}")
st.success(f"Predicted Total Amount (Bill + Tip): ${bill_amount + predicted_tip:.2f}")
st.markdown('</div>', unsafe_allow_html=True)
