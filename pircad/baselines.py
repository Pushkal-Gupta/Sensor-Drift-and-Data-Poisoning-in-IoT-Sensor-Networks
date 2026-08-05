"""
Comparison systems.

Gen1FHAD
    A faithful re-implementation of the Generation-1 disclosure: rolling-mean
    detrending, a high-pass magnitude test for poisoning, CUSUM on the detrended
    signal for drift, and a windowed supervised classifier activated only on
    flagged windows.  It sees a single raw stream, exactly as in the original
    work.  To keep the comparison fair its classifier is trained on the same
    episodes as the Gen-2 root-cause model, and is allowed the same four labels,
    so any gap is attributable to architecture rather than to label vocabulary.

AdaptiveRecalibratingBaseline
    The realistic state of the art for the retraining experiment: a detector that
    tracks an adaptive baseline, and that responds to a sustained deviation by
    recalibrating -- because a sustained deviation "must be drift".  It makes no
    distinction between drift and injection, which is exactly the property a slow
    false-data-injection attack exploits.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .simulator import Episode
from .types import RootCause


# --------------------------------------------------------------------------
# Generation 1
# --------------------------------------------------------------------------
def rolling_mean_detrend(x: np.ndarray, window: int = 60) -> np.ndarray:
    """Centred rolling-mean subtraction, as in the Gen-1 disclosure."""
    s = np.asarray(x, dtype=float)
    n = len(s)
    out = np.zeros(n)
    half = window // 2
    csum = np.concatenate([[0.0], np.cumsum(s)])
    for i in range(n):
        a, b = max(0, i - half), min(n, i + half + 1)
        out[i] = s[i] - (csum[b] - csum[a]) / (b - a)
    return out


class Gen1FHAD:
    """CUSUM + window classifier on the raw stream (Generation 1)."""

    def __init__(self, model=None, detrend_window: int = 60,
                 poison_threshold: float = 8.0, cusum_k: float = 0.15,
                 cusum_h: float = 18.0, ml_window: int = 10,
                 trigger_frac: float = 0.5):
        self.model = model
        self.detrend_window = detrend_window
        self.poison_threshold = poison_threshold
        self.cusum_k = cusum_k
        self.cusum_h = cusum_h
        self.ml_window = ml_window
        self.trigger_frac = trigger_frac
        self.invocations = 0

    # -- training ---------------------------------------------------------
    @staticmethod
    def make_windows(values: np.ndarray, labels: np.ndarray, w: int
                     ) -> Tuple[np.ndarray, np.ndarray]:
        X, y = [], []
        for i in range(len(values) - w):
            X.append(values[i:i + w])
            seg = labels[i:i + w]
            nz = seg[seg != int(RootCause.NORMAL)]
            y.append(int(nz[-1]) if nz.size else int(RootCause.NORMAL))
        return np.asarray(X), np.asarray(y)

    def fit(self, episodes: Sequence[Episode], channel: str = "T_in",
            seed: int = 0) -> "Gen1FHAD":
        from sklearn.ensemble import RandomForestClassifier
        Xs, ys = [], []
        for ep in episodes:
            v = rolling_mean_detrend(ep.obs[channel], self.detrend_window)
            lab = ep.labels.copy()
            if ep.target_channel is not None and ep.target_channel != channel:
                lab = np.full_like(lab, int(RootCause.NORMAL))
            X, y = self.make_windows(v, lab, self.ml_window)
            Xs.append(X); ys.append(y)
        X = np.vstack(Xs); y = np.concatenate(ys)
        self.model = RandomForestClassifier(
            n_estimators=300, max_depth=12, min_samples_leaf=8,
            class_weight="balanced_subsample", n_jobs=-1, random_state=seed)
        self.model.fit(X, y)
        return self

    # -- inference --------------------------------------------------------
    def run(self, ep: Episode, channel: str = "T_in") -> np.ndarray:
        values = rolling_mean_detrend(ep.obs[channel], self.detrend_window)
        n = len(values)
        preds = np.full(n, int(RootCause.NORMAL))
        raw_buf: List[float] = []
        buf: List[float] = []
        cusum = 0.0
        ml_idx, ml_win = [], []
        self.invocations = 0

        for i, v in enumerate(values):
            buf.append(v)
            if len(buf) > self.ml_window:
                buf.pop(0)
            raw_buf.append(v)
            if len(raw_buf) > 90:
                raw_buf.pop(0)
            if len(raw_buf) < 90:
                continue
            rmean = float(np.mean(raw_buf))
            if abs(v - rmean) > self.poison_threshold:
                preds[i] = int(RootCause.DATA_POISONING)
                continue
            cusum = max(0.0, cusum + (v - rmean) - self.cusum_k)
            if cusum > self.cusum_h:
                preds[i] = int(RootCause.SENSOR_DRIFT)
            if (cusum > self.trigger_frac * self.cusum_h
                    and len(buf) == self.ml_window and self.model is not None):
                ml_idx.append(i); ml_win.append(list(buf))

        if ml_win:
            self.invocations = len(ml_win)
            out = self.model.predict(np.asarray(ml_win))
            for i, p in zip(ml_idx, out):
                preds[i] = int(p)
        return preds


# --------------------------------------------------------------------------
# The system a slow injection is designed to walk
# --------------------------------------------------------------------------
@dataclass
class AdaptiveTrace:
    detected: np.ndarray
    offset: np.ndarray
    baseline: np.ndarray
    recalibrations: int = 0


class AdaptiveRecalibratingBaseline:
    """Detect deviation -> assume drift -> recalibrate -> keep learning.

    No cause distinction and no adaptation gate.  This is the behaviour the
    invention's governor is designed to prevent, and the experiment in
    experiments/exp2_poisoned_adaptation.py measures the difference.
    """

    def __init__(self, alpha: float = 0.002, k_sigma: float = 4.0,
                 recal_gain: float = 0.02, warmup: int = 600):
        self.alpha = alpha
        self.k_sigma = k_sigma
        self.recal_gain = recal_gain
        self.warmup = warmup

    def run(self, series: np.ndarray) -> AdaptiveTrace:
        x = np.asarray(series, dtype=float)
        n = len(x)
        det = np.zeros(n, dtype=bool)
        off = np.zeros(n)
        base = np.zeros(n)

        mu = float(np.mean(x[:self.warmup]))
        sd = float(np.std(x[:self.warmup])) or 1e-6
        offset = 0.0
        recals = 0
        for i in range(n):
            v = x[i] - offset
            base[i] = mu
            off[i] = offset
            if i < self.warmup:
                continue
            z = (v - mu) / sd
            if abs(z) > self.k_sigma:
                det[i] = True
                # "it must be drift" -> absorb it
                offset += self.recal_gain * (v - mu)
                recals += 1
            # unconditional adaptation of the reference
            mu = (1 - self.alpha) * mu + self.alpha * v
        return AdaptiveTrace(det, off, base, recals)
