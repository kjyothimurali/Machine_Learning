# ===============================
# train_model.py
# ===============================

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.metrics import accuracy_score

# -------------------------------
# 1. Load Dataset
# -------------------------------
import os 
# Get current file directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Dataset path (same folder)
DATA_PATH = os.path.join(BASE_DIR, "train_u6lujuX_CVtuZ9i.csv")

# Load dataset
df = pd.read_csv(DATA_PATH)

print("✅ Dataset loaded successfully from same folder")
# -------------------------------
# 2. Handle Missing Values
# -------------------------------
# Categorical → mode
cat_cols = ['Gender', 'Married', 'Dependents', 'Self_Employed']
for col in cat_cols:
    df[col].fillna(df[col].mode()[0], inplace=True)

# Numerical → median
df['LoanAmount'].fillna(df['LoanAmount'].median(), inplace=True)
df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].median(), inplace=True)

# Credit History → mode
df['Credit_History'].fillna(df['Credit_History'].mode()[0], inplace=True)

# -------------------------------
# 3. Encoding
# -------------------------------
le = LabelEncoder()

binary_cols = ['Gender', 'Married', 'Education', 'Self_Employed', 'Loan_Status']
for col in binary_cols:
    df[col] = le.fit_transform(df[col])

df = pd.get_dummies(df, columns=['Dependents', 'Property_Area'], drop_first=True)

# -------------------------------
# 4. Feature / Target Split
# -------------------------------
X = df.drop(columns=['Loan_Status', 'Loan_ID'])
y = df['Loan_Status']

# Save feature names (VERY IMPORTANT for app)
joblib.dump(X.columns.tolist(), "feature_names.pkl")

# -------------------------------
# 5. Train-Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -------------------------------
# 6. Define Base Models
# -------------------------------
lr_model = Pipeline([
    ('scaler', StandardScaler()),
    ('lr', LogisticRegression(max_iter=3000))
])

dt_model = DecisionTreeClassifier(random_state=42)

rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# -------------------------------
# 7. Stacking Model
# -------------------------------
stacking_model = StackingClassifier(
    estimators=[
        ('lr', lr_model),
        ('dt', dt_model),
        ('rf', rf_model)
    ],
    final_estimator=LogisticRegression(max_iter=3000),
    cv=5
)

# -------------------------------
# 8. Train Models
# -------------------------------
lr_model.fit(X_train, y_train)
dt_model.fit(X_train, y_train)
rf_model.fit(X_train, y_train)
stacking_model.fit(X_train, y_train)

# -------------------------------
# 9. Evaluation (Optional)
# -------------------------------
print("Random Forest Accuracy:", accuracy_score(y_test, rf_model.predict(X_test)))
print("Stacking Accuracy:", accuracy_score(y_test, stacking_model.predict(X_test)))

# -------------------------------
# 10. Save Models
# -------------------------------
joblib.dump(lr_model, "lr_model.pkl")
joblib.dump(dt_model, "dt_model.pkl")
joblib.dump(rf_model, "rf_model.pkl")
joblib.dump(stacking_model, "stacking_model.pkl")

print("✅ Models trained and saved successfully!")
