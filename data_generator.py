import numpy as np
import pandas as pd

def generate_normal_data(n=300):
    """
    Generates normal sensor data with small noise
    """
    data = np.random.normal(loc=28, scale=0.5, size=n)
    return data


def generate_drift_data(n=300):
    """
    Generates sensor data with gradual drift
    """
    drift = np.linspace(0, 6, n)          # gradual increase
    noise = np.random.normal(0, 0.5, n)
    data = 28 + drift + noise
    return data


def generate_poisoned_data(n=300, poison_rate=0.05):
    """
    Generates sensor data with sudden abnormal spikes
    """
    data = np.random.normal(loc=28, scale=0.5, size=n)
    num_poison = int(n * poison_rate)

    poison_indices = np.random.choice(n, num_poison, replace=False)
    for i in poison_indices:
        data[i] = data[i] + np.random.uniform(20, 40)  # big spike

    return data


def generate_dataset():
    normal = generate_normal_data()
    drift = generate_drift_data()
    poisoned = generate_poisoned_data()

    df = pd.DataFrame({
        "Normal": normal,
        "Drift": drift,
        "Poisoned": poisoned
    })

    df.to_csv("sensor_data.csv", index=False)
    print("Sensor data generated and saved as sensor_data.csv")


if __name__ == "__main__":
    generate_dataset()
