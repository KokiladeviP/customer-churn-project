import pandas as pd
import matplotlib.pyplot as plt
import os

# -----------------------------------------
# Load data
# -----------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "output", "churn_analysis_output.csv")

df = pd.read_csv(DATA_PATH)

# =================================================
# 1️⃣ PIE CHART – Risk Factor Proportion
# =================================================
plt.figure()
df["Risk_Factor"].value_counts().plot(kind="pie", autopct="%1.1f%%")
plt.title("Customer Risk Factor Proportion")
plt.ylabel("")
plt.show()

# =================================================
# 2️⃣ HORIZONTAL BAR – Retention Strategy Count
# =================================================
plt.figure()
df["Top_Retention_Strategy"].value_counts().plot(kind="barh")
plt.title("Retention Strategy Requirement")
plt.xlabel("Number of Customers")
plt.ylabel("Retention Strategy")
plt.show()

# =================================================
# 3️⃣ STACKED BAR – Risk vs Retention Strategy
# =================================================
risk_retention = df.groupby(
    ["Risk_Factor", "Top_Retention_Strategy"]
).size().unstack()

risk_retention.plot(kind="bar", stacked=True)
plt.title("Risk Level vs Retention Strategy Mapping")
plt.xlabel("Risk Factor")
plt.ylabel("Number of Customers")
plt.legend(title="Retention Strategy")
plt.show()

# =================================================
# 4️⃣ HISTOGRAM – Churn Probability (Overall)
# =================================================
plt.figure()
plt.hist(df["Churn_Probability"], bins=10)
plt.title("Overall Churn Probability Distribution")
plt.xlabel("Churn Probability")
plt.ylabel("Number of Customers")
plt.show()

# =================================================
# 5️⃣ HISTOGRAM – Churn Probability by Risk
# =================================================
plt.figure()
df[df["Risk_Factor"] == "High Risk"]["Churn_Probability"].hist(bins=10)
df[df["Risk_Factor"] == "Medium Risk"]["Churn_Probability"].hist(bins=10)
df[df["Risk_Factor"] == "Low Risk"]["Churn_Probability"].hist(bins=10)
plt.title("Churn Probability by Risk Category")
plt.xlabel("Churn Probability")
plt.ylabel("Number of Customers")
plt.legend(["High Risk", "Medium Risk", "Low Risk"])
plt.show()

# =================================================
# 6️⃣ BOX PLOT – Churn Probability vs Risk
# =================================================
plt.figure()
df.boxplot(column="Churn_Probability", by="Risk_Factor")
plt.title("Churn Probability Spread by Risk Level")
plt.suptitle("")
plt.xlabel("Risk Factor")
plt.ylabel("Churn Probability")
plt.show()
