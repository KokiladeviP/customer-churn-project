import pandas as pd
import matplotlib.pyplot as plt
import os

# ---------------------------------------
# Load output file
# ---------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "output", "churn_analysis_output.csv")

df = pd.read_csv(DATA_PATH)

# ================= GRAPH 1 =================
# Risk Factor Distribution
plt.figure()
df["Risk_Factor"].value_counts().plot(kind="bar")
plt.title("Customer Churn Risk Distribution")
plt.xlabel("Risk Level")
plt.ylabel("Number of Customers")
plt.show()

# ================= GRAPH 2 =================
# Retention Strategy Distribution
plt.figure()
df["Top_Retention_Strategy"].value_counts().plot(kind="bar")
plt.title("Recommended Retention Strategies")
plt.xlabel("Retention Strategy")
plt.ylabel("Number of Customers")
plt.show()

# ================= GRAPH 3 =================
# Churn Probability Distribution
plt.figure()
df["Churn_Probability"].hist(bins=10)
plt.title("Churn Probability Distribution")
plt.xlabel("Churn Probability")
plt.ylabel("Number of Customers")
plt.show()

# ================= GRAPH 4 (OPTIONAL BUT GOOD) =================
# High Risk Customers Only
high_risk = df[df["Risk_Factor"] == "High Risk"]

plt.figure()
high_risk["Churn_Probability"].hist(bins=10)
plt.title("High Risk Customers - Churn Probability")
plt.xlabel("Churn Probability")
plt.ylabel("Number of Customers")
plt.show()
