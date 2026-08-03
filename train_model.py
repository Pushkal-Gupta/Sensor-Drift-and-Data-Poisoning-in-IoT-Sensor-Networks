import pandas as pd
from random_forest_model import train_random_forest
import joblib

df = pd.read_csv("sensor_data_with_anomalies.csv")

values = df["value"].values
labels = df["label"].values

model = train_random_forest(values, labels)

joblib.dump(model, "rf_model.pkl")

print("Random Forest trained and saved.")