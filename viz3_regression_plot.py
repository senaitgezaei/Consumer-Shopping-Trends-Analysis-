import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix

matplotlib.use('TkAgg')

# Load dataset
df = pd.read_csv("Consumer_Shopping_Trends_2026 (6).csv")

# Standardize labels
df['shopping_preference'] = df['shopping_preference'].str.upper()

# VISUAL 3: CURVE FITTING

# Create a new column for total spending
df['total_spend'] = df['avg_online_spend'] + df['avg_store_spend']

plt.figure(figsize=(8,6))

# Fit a second-order polynomial regression line to the data
sns.regplot(
    data=df,
    x="daily_internet_hours",
    y="total_spend",
    order=2,
    scatter_kws={'alpha':0.3}
)

plt.title("Internet Usage vs Total Spending")
plt.xlabel("Daily Internet Hours")
plt.ylabel("Total Spending ($)")
plt.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()