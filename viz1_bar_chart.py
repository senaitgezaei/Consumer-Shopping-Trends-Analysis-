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


# VISUAL 1: BAR CHART

# Calculate average online spend for each shopping preference and sort them
avg_spend = df.groupby('shopping_preference')['avg_online_spend'].mean().sort_values(ascending=False)

plt.figure(figsize=(8,6))
avg_spend.plot(kind='bar')

plt.title("Average Online Spending by Shopping Preference")
plt.xlabel("Shopping Preference")
plt.ylabel("Average Online Spend ($ per month)")
plt.grid(axis='y', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()