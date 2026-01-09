import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

# ---------------- TITLE ----------------
st.markdown(
    "<h1 style='text-align:center;color:#4CAF50;'>📊 Customer Churn Prediction using Logistic Regression</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align:center;'>Analyze and predict customer churn using machine learning</p>",
    unsafe_allow_html=True
)

# ---------------- SIDEBAR ----------------
st.sidebar.header("⚙️ Model Settings")
test_size = st.sidebar.slider("Test Size (%)", 10, 40, 20) / 100
max_iter = st.sidebar.slider("Max Iterations", 100, 2000, 1000, step=100)

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    file_path = os.path.join(
        os.path.dirname(__file__),
        "WA_Fn-UseC_-Telco-Customer-Churn.csv"
    )
    return pd.read_csv(file_path)
df = load_data()

# ---------------- PREPROCESSING ----------------
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.dropna(inplace=True)

le = LabelEncoder()
for col in df.select_dtypes(include=['object']).columns:
    df[col] = le.fit_transform(df[col])

X = df.drop("Churn", axis=1)
y = df["Churn"]

# ---------------- SPLIT DATA ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=42
)

# ---------------- MODEL ----------------
model = LogisticRegression(max_iter=max_iter)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# ---------------- METRICS ----------------
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred, output_dict=True)

# ---------------- METRIC CARDS ----------------
st.markdown("##  Model Performance")

col1, col2, col3 = st.columns(3)
col1.metric("Accuracy", f"{accuracy:.2%}")
col2.metric("Total Customers", len(df))
col3.metric("Test Samples", len(y_test))

# ---------------- CONFUSION MATRIX ----------------
st.markdown("### 🔁 Confusion Matrix")

fig, ax = plt.subplots()
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
st.pyplot(fig)

# ---------------- CLASSIFICATION REPORT ----------------
with st.expander("📄 Classification Report"):
    st.dataframe(pd.DataFrame(class_report).transpose())

# ---------------- FEATURE IMPORTANCE ----------------
st.markdown("### 🔍 Feature Importance (Logistic Coefficients)")

coeff_df = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_[0]
}).sort_values(by="Coefficient", ascending=False)

st.dataframe(coeff_df, height=350)

fig2, ax2 = plt.subplots(figsize=(8, 6))
sns.barplot(
    data=coeff_df.head(10),
    x="Coefficient",
    y="Feature",
    ax=ax2,
    palette="viridis"
)
ax2.set_title("Top 10 Features Influencing Churn")
st.pyplot(fig2)

# ---------------- DATASET PREVIEW ----------------
with st.expander("📊 Dataset Preview"):
    st.dataframe(df.head())

# ---------------- FOOTER ----------------
st.markdown(
    "<hr><p style='text-align:center;color:gray;'>Built with ❤️ using Streamlit & Scikit-learn</p>",
    unsafe_allow_html=True
)
