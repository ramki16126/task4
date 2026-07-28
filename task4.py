import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    roc_auc_score,
    roc_curve
)

# Load dataset
df = pd.read_csv("data.csv")

# Remove unnecessary columns
df.drop(columns=["id", "Unnamed: 32"], inplace=True)

# Convert diagnosis to numeric
# M = 1, B = 0
df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})

# Features and Target
X = df.drop("diagnosis", axis=1)
y = df["diagnosis"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Standardize features
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train Logistic Regression model
model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix")
print(cm)

disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title("Confusion Matrix")
plt.show()

# Classification Report
print("\nClassification Report")
print(classification_report(y_test, y_pred))

# ROC-AUC Score
prob = model.predict_proba(X_test)[:, 1]

auc = roc_auc_score(y_test, prob)

print("ROC-AUC Score:", auc)

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, prob)

plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
plt.plot([0,1],[0,1],'--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

# Threshold Tuning
print("\nThreshold Tuning (0.4)")

custom_pred = (prob >= 0.4).astype(int)

cm2 = confusion_matrix(y_test, custom_pred)

print(cm2)

print(classification_report(y_test, custom_pred))

print("\nTask 4 Completed Successfully!")