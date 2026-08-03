import numpy as np
from sklearn.ensemble import RandomForestClassifier

def train_random_forest(values, labels):

    X = []
    y = []

    window = 10

    for i in range(len(values) - window):
        segment = values[i:i+window]
        X.append(segment)
        y.append(np.max(labels[i:i+window]))

    X = np.array(X)
    y = np.array(y)

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=8,
        class_weight="balanced",
        random_state=42
    )

    model.fit(X, y)

    return model