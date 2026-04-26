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


# LOGISTIC REGRESSION MODEL

# Prepare data for modeling
features = ['age','daily_internet_hours','monthly_online_orders','avg_online_spend','avg_store_spend']
X = df[features]
y = df['shopping_preference']

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.25, stratify=y_encoded, random_state=42
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train logistic regression model
model = LogisticRegression(multi_class='multinomial', max_iter=500)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)


# VISUAL 2: CONFUSION MATRIX

# Compute confusion matrix and normalize it
cm = confusion_matrix(y_test, y_pred, normalize='true')

# Plot the confusion matrix as a heatmap
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, cmap='Blues',
            xticklabels=le.classes_,
            yticklabels=le.classes_,
            fmt='.2f')

plt.title("Logistic Regression Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.tight_layout()
plt.show()