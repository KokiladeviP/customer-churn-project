# ============================================================
# Telecom Customer Churn Prediction & Retention Strategy
# Random Forest Classifier
# ============================================================

import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(BASE_DIR, "telecom_churn.csv")
MODEL_DIR = os.path.join(BASE_DIR, "model")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

MODEL_PATH = os.path.join(MODEL_DIR, "churn_model.pkl")
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "churn_analysis_output.csv")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------------------------------------------------
# Load Dataset
# ------------------------------------------------------------
data = pd.read_csv(DATA_PATH)

# ------------------------------------------------------------
# Encode Categorical Columns
# ------------------------------------------------------------
le = LabelEncoder()
for col in data.select_dtypes(include="object").columns:
    data[col] = le.fit_transform(data[col])

# ------------------------------------------------------------
# Feature & Target
# ------------------------------------------------------------
X = data.drop("Churn", axis=1)
y = data["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ------------------------------------------------------------
# Train Random Forest
# ------------------------------------------------------------
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
model.fit(X_train, y_train)

# ------------------------------------------------------------
# Evaluate Model
# ------------------------------------------------------------
y_pred = model.predict(X_test)
print("\nModel Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# ------------------------------------------------------------
# Save Model
# ------------------------------------------------------------
joblib.dump(model, MODEL_PATH)

# ------------------------------------------------------------
# Predict for ALL Customers
# ------------------------------------------------------------
data["Churn_Probability"] = model.predict_proba(X)[:, 1]

data["Churn_Prediction"] = data["Churn_Probability"].apply(
    lambda x: "Yes" if x >= 0.5 else "No"
)

# ------------------------------------------------------------
# BUSINESS-OPTIMIZED RISK FACTOR (FINAL LOGIC)
# ------------------------------------------------------------
def assign_risk_factor(prob):
    if prob >= 0.95:
        return "High Risk"
    elif prob >= 0.08:
        return "Medium Risk"
    else:
        return "Low Risk"

data["Risk_Factor"] = data["Churn_Probability"].apply(assign_risk_factor)

# ------------------------------------------------------------
# RETENTION STRATEGY BASED ON RISK
# ------------------------------------------------------------
def retention_strategy(risk):
    if risk == "High Risk":
        return "Instant Discount / Cashback"
    elif risk == "Medium Risk":
        return "Free Data / Validity Extension"
    else:
        return "Priority Customer Support"

data["Top_Retention_Strategy"] = data["Risk_Factor"].apply(retention_strategy)

# ------------------------------------------------------------
# Save Final Output
# ------------------------------------------------------------
data.to_csv(OUTPUT_PATH, index=False)

print("\n✅ Model saved at:", MODEL_PATH)
print("✅ Output saved at:", OUTPUT_PATH)
print("✅ Churn Prediction, Risk Factor & Retention Strategy completed")
