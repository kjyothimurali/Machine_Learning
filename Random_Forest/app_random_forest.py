import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(
    page_title="Social Network Ads Prediction",
    page_icon="📢",
    layout="wide"
)

st.title("📢 Social Network Ads Prediction using Random Forest")
st.write("Predict whether a user will purchase a product based on age and salary.")

# -------------------------------
# LOAD DATA
# -------------------------------
@st.cache_data
def load_data():
    file_path = os.path.join(
        os.path.dirname(__file__),
        "Social_Network_Ads.csv"
    )
    return pd.read_csv(file_path)

df = load_data()

st.subheader("📊 Dataset Preview")
st.dataframe(df.head())

# -------------------------------
# DATA PREPROCESSING
# -------------------------------
st.subheader("🛠 Data Preprocessing")

df = df[['Age', 'EstimatedSalary', 'Purchased']]

X = df[['Age', 'EstimatedSalary']]
y = df['Purchased']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

st.write("Processed Features (Scaled):")
st.dataframe(pd.DataFrame(X_scaled, columns=X.columns).head())

# -------------------------------
# TRAIN-TEST SPLIT
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.25, random_state=42
)

# -------------------------------
# MODEL TRAINING
# -------------------------------
st.subheader("🌲 Random Forest Model Settings")

colA, colB = st.columns(2)
with colA:
    n_estimators = st.slider("Number of Trees", 50, 300, 100, step=50)
with colB:
    max_depth = st.slider("Max Depth", 2, 15, 5)

model = RandomForestClassifier(
    n_estimators=n_estimators,
    max_depth=max_depth,
    random_state=42
)
model.fit(X_train, y_train)

# -------------------------------
# MODEL EVALUATION
# -------------------------------
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

st.subheader("📈 Model Performance")

c1, c2, c3 = st.columns(3)
c1.metric("Accuracy", f"{accuracy:.2%}")
c2.metric("Training Samples", len(X_train))
c3.metric("Testing Samples", len(X_test))

# -------------------------------
# CONFUSION MATRIX
# -------------------------------
st.subheader("🔢 Confusion Matrix")

cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
st.pyplot(fig)

# -------------------------------
# CLASSIFICATION REPORT
# -------------------------------
with st.expander("📄 Classification Report"):
    st.text(classification_report(y_test, y_pred))

# -------------------------------
# FEATURE IMPORTANCE
# -------------------------------
st.subheader("⭐ Feature Importance")

importance = model.feature_importances_
features = X.columns

fig2, ax2 = plt.subplots()
sns.barplot(x=importance, y=features, ax=ax2, palette="viridis")
ax2.set_title("Feature Importance")
st.pyplot(fig2)

# -------------------------------
# USER INPUT PREDICTION
# -------------------------------
st.subheader("🧍 Predict Purchase (User Input)")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", 18, 70, 30)

with col2:
    salary = st.number_input("Estimated Salary", 15000, 150000, 50000)

user_data = scaler.transform([[age, salary]])

if st.button("🔮 Predict Purchase"):
    prediction = model.predict(user_data)[0]
    probability = model.predict_proba(user_data)[0][1]

    if prediction == 1:
        st.success("✅ User is likely to PURCHASE the product")
    else:
        st.error("❌ User is NOT likely to purchase the product")

    st.write(f"**Purchase Probability:** {probability:.2%}")
    st.progress(int(probability * 100))

# -------------------------------
# FOOTER
# -------------------------------
st.markdown("---")
st.markdown(
    "💡 **Model:** Random Forest Classifier | **Dataset:** Social Network Ads",
    unsafe_allow_html=True
)

