import numpy as np
import pandas as pd


def inject_drift(data, start_index, length, slope):
    """
    Injects linear upward drift into a copy of data.
    """
    data_drifted = data.copy()
    labels = np.zeros(len(data))  # 0 = Normal

    for i in range(length):
        idx = start_index + i
        if idx < len(data_drifted):
            data_drifted[idx] += slope * i
            labels[idx] = 1  # 1 = Drift

    return data_drifted, labels


def inject_poisoning(data, labels, indices, magnitude):
    """
    Injects sudden spikes into data.
    """
    data_poisoned = data.copy()

    for idx in indices:
        if idx < len(data_poisoned):
            data_poisoned[idx] += magnitude
            labels[idx] = 2  # 2 = Poisoned

    return data_poisoned, labels


if __name__ == "__main__":

    from dataset_loader import load_temperature_data

    # Load original temperature data
    temps = load_temperature_data("1_Daily_minimum_temps.csv")

    # Inject drift
    drifted_data, labels = inject_drift(
        temps,
        start_index=1000,
        length=300,
        slope=0.02
    )

    # Inject poisoning spikes
    poisoned_data, labels = inject_poisoning(
        drifted_data,
        labels,
        indices=[500, 1500, 2500],
        magnitude=15
    )

    print("Injection complete.")
    print("Drift labels count:", np.sum(labels == 1))
    print("Poison labels count:", np.sum(labels == 2))

    # -----------------------------
    # SAVE DATASET FOR EVALUATION
    # -----------------------------
    df = pd.DataFrame({
        "value": poisoned_data,
        "label": labels
    })

    df.to_csv("sensor_data_with_anomalies.csv", index=False)

    print("Dataset saved as: sensor_data_with_anomalies.csv")