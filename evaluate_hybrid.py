import pandas as pd
import numpy as np
import joblib
import time
from sklearn.metrics import classification_report, confusion_matrix

from fog_node import FogNode

# ------------------------------
# Load dataset
# ------------------------------

df = pd.read_csv("sensor_data_with_anomalies.csv")

values = df["value"].values
labels = df["label"].values


# ------------------------------
# Load trained ML model
# ------------------------------

rf_model = joblib.load("rf_model.pkl")


# ------------------------------
# Initialize Fog Node
# ------------------------------

fog = FogNode()

window = 10
buffer = []

predictions = []

# Store windows that require ML prediction
ml_windows = []
ml_indices = []


# ------------------------------
# Start timing
# ------------------------------

start = time.time()


# ------------------------------
# First pass: CUSUM + collect ML windows
# ------------------------------

for i, v in enumerate(values):

    buffer.append(v)

    if len(buffer) > window:
        buffer.pop(0)

    cusum_result = fog.process_value(v)

    # Poison detected immediately
    if cusum_result == 2:
        predictions.append(2)
        continue

    # Suspicious trigger → use ML
    if fog.cusum_score > 0.5 * fog.cusum_h and len(buffer) == window:

        ml_windows.append(buffer.copy())
        ml_indices.append(i)

        predictions.append(-1)  # placeholder

    else:
        predictions.append(cusum_result)


# ------------------------------
# Batch Random Forest prediction
# ------------------------------

if len(ml_windows) > 0:

    ml_windows = np.array(ml_windows)

    ml_preds = rf_model.predict(ml_windows)

    for idx, pred in zip(ml_indices, ml_preds):
        predictions[idx] = pred


# ------------------------------
# End timing
# ------------------------------

end = time.time()

runtime = end - start
latency = runtime / len(values)

predictions = np.array(predictions)


# ------------------------------
# Print results
# ------------------------------

print("\nHybrid Runtime:", runtime, "seconds")
print("Average Latency:", latency, "seconds per sample")

print("\nConfusion Matrix:")
print(confusion_matrix(labels, predictions))

print("\nClassification Report:")
print(classification_report(labels, predictions))