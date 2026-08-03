import time
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import IsolationForest


# ------------------------------
# Load dataset
# ------------------------------
df = pd.read_csv("sensor_data_with_anomalies.csv")

values = df["value"].values
labels = df["label"].values


# ------------------------------
# Convert labels to binary
# 0 = normal
# 1 = anomaly (drift or poison)
# ------------------------------
binary_labels = []

for l in labels:
    if l == 0:
        binary_labels.append(0)
    else:
        binary_labels.append(1)

binary_labels = np.array(binary_labels)


# ------------------------------
# Prepare data for sklearn
# ------------------------------
X = values.reshape(-1, 1)


# ------------------------------
# Train Isolation Forest
# ------------------------------
model = IsolationForest(
    contamination=0.08,   # ~300 anomalies out of 3600
    random_state=42
)

start_time = time.time()

model.fit(X)
preds = model.predict(X)

end_time = time.time()

print("\nIsolation Forest Runtime:", end_time - start_time, "seconds")


# ------------------------------
# Convert sklearn output
# 1 = normal
# -1 = anomaly
# ------------------------------
predictions = []

for p in preds:
    if p == 1:
        predictions.append(0)
    else:
        predictions.append(1)

predictions = np.array(predictions)


# ------------------------------
# Evaluation
# ------------------------------
print("\nIsolation Forest Results")

print("\nConfusion Matrix:")
print(confusion_matrix(binary_labels, predictions))

print("\nClassification Report:")
print(classification_report(binary_labels, predictions))