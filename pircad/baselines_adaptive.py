"""
Self-calibrating comparison systems for the adaptation-poisoning experiment.

Each one forwards a *corrected* temperature to the cloud and adapts its
correction online, which is precisely the property a slow injection exploits.
None of them has a cause vocabulary.

KalmanBiasEstimator     augmented-state Kalman filter, x = [T, b], y = T + b,
                        driven by the same first-principles vessel model the
                        fog node uses.  The textbook model-based way to
                        estimate and remove a sensor bias online.
PageHinkleyRecalibrator detect a persistent shift with a Page-Hinkley test,
                        then recalibrate by the shift's estimated size and
                        restart -- the "detect drift -> recalibrate" pattern of
                        drift-adaptive IoT pipelines.
(AdaptiveRecalibratingBaseline in baselines.py is the third: an EWMA
reference with z-test-triggered absorption.)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import PlantParams


@dataclass
class CorrectionTrace:
    offset: np.ndarray          # correction subtracted from the raw reading
    detected: np.ndarray        # per-sample alarm/adaptation flag


class KalmanBiasEstimator:
    def __init__(self, p: PlantParams = None, q_T: float = 0.02 ** 2,
                 q_b: float = 2e-4 ** 2, r: float = 0.05 ** 2,
                 param_error: float = 1.0, warmup: int = 600):
        p = p or PlantParams()
        # same identified (imperfect) parameters as the fog node's twin
        self.C = p.C * (1 + 0.030 * param_error)
        self.k = p.k_loss * (1 - 0.025 * param_error)
        self.Q = p.Q * (1 + 0.010 * param_error)
        self.dt = p.dt
        self.qT, self.qb, self.r = q_T, q_b, r
        self.warmup = warmup

    def run(self, y: np.ndarray, u: np.ndarray, T_amb: np.ndarray) -> CorrectionTrace:
        n = len(y)
        a = 1.0 - self.dt * self.k / self.C
        F = np.array([[a, 0.0], [0.0, 1.0]])
        H = np.array([1.0, 1.0])
        Qm = np.diag([self.qT, self.qb])
        x = np.array([y[0], 0.0])
        P = np.diag([0.1, 1e-6])
        off = np.zeros(n); det = np.zeros(n, dtype=bool)
        for i in range(n):
            if i > 0:
                bu = self.dt / self.C * (self.Q * u[i - 1] + self.k * T_amb[i - 1])
                x = F @ x + np.array([bu, 0.0])
                P = F @ P @ F.T + Qm
            S = H @ P @ H + self.r
            K = P @ H / S
            innov = y[i] - H @ x
            if i < self.warmup:
                # commissioning: the sensor is known good, pin the bias at zero
                K = np.array([K[0], 0.0])
            x = x + K * innov
            P = (np.eye(2) - np.outer(K, H)) @ P
            off[i] = x[1]
            det[i] = abs(innov) > 3.0 * np.sqrt(S)
        return CorrectionTrace(off, det)


class PageHinkleyRecalibrator:
    """Page-Hinkley on the deviation from a frozen commissioning reference."""

    def __init__(self, delta: float = 0.05, lam: float = 25.0,
                 window: int = 200, warmup: int = 600):
        self.delta, self.lam, self.window, self.warmup = delta, lam, window, warmup

    def run(self, y: np.ndarray) -> CorrectionTrace:
        n = len(y)
        mu = float(np.mean(y[:self.warmup]))
        off = np.zeros(n); det = np.zeros(n, dtype=bool)
        offset = 0.0
        m_up = m_dn = 0.0; M_up = M_dn = 0.0
        buf = []
        for i in range(n):
            off[i] = offset
            if i < self.warmup:
                continue
            d = (y[i] - offset) - mu
            buf.append(d)
            if len(buf) > self.window:
                buf.pop(0)
            m_up += d - self.delta; M_up = min(M_up, m_up)
            m_dn += -d - self.delta; M_dn = min(M_dn, m_dn)
            if (m_up - M_up) > self.lam or (m_dn - M_dn) > self.lam:
                det[i] = True
                offset += float(np.mean(buf))       # "it is drift": recalibrate
                buf.clear()
                m_up = m_dn = M_up = M_dn = 0.0
        return CorrectionTrace(off, det)
