import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# -------------------------------
# App Title
# -------------------------------
st.set_page_config(page_title="Titanic Survival Prediction", layout="wide")
st.title("🚢 Titanic Survival Prediction using Decision Tree")

# -------------------------------
# Load Data
# -------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("titanic.csv")
    return df

df = load_data()

st.subheader("📊 Dataset Preview")
st.dataframe(df.head())

# -------------------------------
# Data Preprocessing
# -------------------------------
st.subheader("🛠 Data Preprocessing")

df = df[['Survived','Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']]
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

st.write("Processed Data Sample:")
st.dataframe(df.head())

# -------------------------------
# Train-Test Split
# -------------------------------
X = df.drop('Survived', axis=1)
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# -------------------------------
# Model Training
# -------------------------------
st.subheader("🌳 Decision Tree Model")

max_depth = st.slider("Select Max Depth of Tree", 2, 10, 4)

model = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
model.fit(X_train, y_train)

# -------------------------------
# Model Evaluation
# -------------------------------
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

st.metric("Model Accuracy", f"{accuracy:.2f}")

# Classification Report
st.subheader("📈 Classification Report")
st.text(classification_report(y_test, y_pred))

# -------------------------------
# Confusion Matrix
# -------------------------------
st.subheader("🔢 Confusion Matrix")

cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
st.pyplot(fig)

# -------------------------------
# Feature Importance
# -------------------------------
st.subheader("⭐ Feature Importance")

importance = model.feature_importances_
features = X.columns

fig2, ax2 = plt.subplots()
sns.barplot(x=importance, y=features, ax=ax2)
ax2.set_title("Feature Importance")
st.pyplot(fig2)

# -------------------------------
# User Input Prediction
# -------------------------------
st.subheader("🧍 Predict Survival (User Input)")

col1, col2, col3 = st.columns(3)

with col1:
    pclass = st.selectbox("Passenger Class", [1, 2, 3])
    sex = st.selectbox("Sex", ["Male", "Female"])
    age = st.number_input("Age", 1, 80, 25)

with col2:
    sibsp = st.number_input("Siblings/Spouses", 0, 8, 0)
    parch = st.number_input("Parents/Children", 0, 6, 0)
    fare = st.number_input("Fare", 0.0, 600.0, 30.0)

with col3:
    embarked = st.selectbox("Embarked", ["S", "C", "Q"])

sex = 1 if sex == "Female" else 0
embarked = {"S":0, "C":1, "Q":2}[embarked]

user_data = np.array([[pclass, sex, age, sibsp, parch, fare, embarked]])

if st.button("🔮 Predict Survival"):
    prediction = model.predict(user_data)
    result = "🟢 Survived" if prediction[0] == 1 else "🔴 Not Survived"
    st.success(f"Prediction: **{result}**")

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.markdown("💡 **Model:** Decision Tree Classifier | **Dataset:** Titanic")
