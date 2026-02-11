# Pension Fraud Detection using Machine Learning

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ----------------------------
# 1. Load Dataset
# ----------------------------
# Replace with your dataset file
df = pd.read_csv("pension_data.csv")

print("First 5 rows:")
print(df.head())

# ----------------------------
# 2. Data Preprocessing
# ----------------------------

# Example features (modify based on your dataset)
# Assume 'is_fraud' is target column (0 = Genuine, 1 = Fraud)
X = df.drop("is_fraud", axis=1)
y = df["is_fraud"]

# Handle missing values (simple method)
X = X.fillna(X.mean())

# Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ----------------------------
# 3. Train-Test Split
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# ----------------------------
# 4. Train Model
# ----------------------------
model = LogisticRegression()
model.fit(X_train, y_train)

# ----------------------------
# 5. Predictions
# ----------------------------
y_pred = model.predict(X_test)

# ----------------------------
# 6. Evaluation
# ----------------------------
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ----------------------------
# 7. Predict New Citizen
# ----------------------------
# Example new data (must match feature order)
new_citizen = np.array([[65, 2, 15000, 1]])  # Example values

new_scaled = scaler.transform(new_citizen)
prediction = model.predict(new_scaled)

if prediction[0] == 1:
    print("⚠ Fraud Detected")
else:
    print("✔ Genuine Beneficiary")
