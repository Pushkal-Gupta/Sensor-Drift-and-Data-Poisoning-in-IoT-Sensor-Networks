import numpy as np

class FogNode:
    def __init__(self,
                 poison_threshold=8,
                 detrend_window=90,
                 cusum_k=0.15,
                 cusum_h=18):

        self.poison_threshold = poison_threshold
        self.detrend_window = detrend_window
        self.cusum_k = cusum_k
        self.cusum_h = cusum_h

        self.raw_buffer = []
        self.cusum_score = 0

    def process_value(self, value):

        self.raw_buffer.append(value)

        if len(self.raw_buffer) > self.detrend_window:
            self.raw_buffer.pop(0)

        # Wait until detrend window fills
        if len(self.raw_buffer) < self.detrend_window:
            return 0

        rolling_mean = np.mean(self.raw_buffer)

        # -----------------
        # Poison Detection (High-pass)
        # -----------------
        if abs(value - rolling_mean) > self.poison_threshold:
            return 2  # POISON

        # -----------------
        # Detrend
        # -----------------
        residual = value - rolling_mean

        # -----------------
        # Residual CUSUM
        # -----------------
        deviation = residual - self.cusum_k
        self.cusum_score = max(0, self.cusum_score + deviation)

        if self.cusum_score > self.cusum_h:
            return 1  # DRIFT

        return 0