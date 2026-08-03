import pandas as pd

def load_temperature_data(filepath):
    df = pd.read_csv(filepath)

    # Convert temperature column to numeric (important)
    df['Temp'] = pd.to_numeric(df['Temp'], errors='coerce')

    # Drop any missing values
    df = df.dropna()

    temps = df['Temp'].values.astype(float)

    return temps

if __name__ == "__main__":
    temps = load_temperature_data("1_Daily_minimum_temps.csv")
    print("Total samples:", len(temps))
    print("First 10 values:", temps[:10])
    print("Data type:", temps.dtype)