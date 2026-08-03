import numpy as np
from fog_node import FogNode

class HybridDetector:

    def __init__(self, ml_model):
        self.fog = FogNode()
        self.ml_model = ml_model
        self.buffer = []
        self.window = 10

    def process(self, value):

        self.buffer.append(value)

        if len(self.buffer) > self.window:
            self.buffer.pop(0)

        cusum_result = self.fog.process_value(value)

        # Stage 1 result
        if cusum_result == 2:
            return 2  # poison

        # Suspicious trigger
        if self.fog.cusum_score > 0.5 * self.fog.cusum_h:

            if len(self.buffer) == self.window:
                X = np.array(self.buffer).reshape(1, -1)

                ml_pred = self.ml_model.predict(X)[0]

                return ml_pred

        return cusum_result