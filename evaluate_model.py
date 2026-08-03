import time
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

from fog_node import FogNode


# ------------------------------
# Detrending function
# ------------------------------
def detrend_signal(data, window=60):
    """
    Removes seasonal component using rolling mean subtraction.
    """

    series = pd.Series(data)

    rolling_mean = series.rolling(window=window, center=True).mean()

    detrended = series - rolling_mean

    # Replace NaN edges with 0
    detrended = detrended.fillna(0)

    return detrended.values


# ------------------------------
# Load dataset
# ------------------------------
df = pd.read_csv("sensor_data_with_anomalies.csv")

values = df["value"].values
labels = df["label"].values


# ------------------------------
# Apply detrending
# ------------------------------
detrended_data = detrend_signal(values, window=60)


# ------------------------------
# Initialize fog node
# ------------------------------
fog = FogNode()

predictions = []


# ------------------------------
# Detection loop
# ------------------------------
start_time = time.time()

for raw_value, detrended_value in zip(values, detrended_data):

    pred = fog.process_value(detrended_value)

    predictions.append(pred)

end_time = time.time()

print("\nCUSUM Runtime:", end_time - start_time, "seconds")


predictions = np.array(predictions)


# ------------------------------
# Evaluation
# ------------------------------
print("\nConfusion Matrix:")
print(confusion_matrix(labels, predictions))

print("\nClassification Report:")
print(classification_report(labels, predictions))