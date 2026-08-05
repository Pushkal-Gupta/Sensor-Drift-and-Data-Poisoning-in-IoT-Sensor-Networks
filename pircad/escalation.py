"""
E6 -- Confidence-gated escalation.

Only the cases the fog node cannot resolve confidently leave the fog node.  Two
things follow, and both are engineering effects rather than accuracy claims:

  * the expensive classifier runs on a small fraction of samples, so the fog node
    keeps its latency and energy budget;
  * the expensive classifier is allowed to be slow, because it is off the
    real-time path.  Escalated samples are *held* (never forwarded, never used
    for adaptation) until the verdict returns, and if it never returns the local
    verdict stands.

The service is asynchronous by construction: `submit` returns a ticket, `poll`
returns verdicts that have become due.  `latency_samples` models the round trip
to whatever backend is used -- a heavier model in the cloud, or a gate-model
quantum backend for which minute-scale round trips are realistic today.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .types import RootCause


@dataclass
class Ticket:
    id: int
    k_submitted: int
    k_due: int
    channel: str
    x: np.ndarray
    local_cause: RootCause
    local_confidence: float


@dataclass
class Verdict:
    ticket: Ticket
    cause: RootCause
    confidence: float
    backend: str


class EscalationBackend:
    name = "abstract"

    def classify(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError


class LocalHeavyBackend(EscalationBackend):
    """A heavier conventional classifier, e.g. hosted at the cloud tier."""

    name = "gradient_boosting"

    def __init__(self, model=None, seed: int = 0):
        if model is None:
            from sklearn.ensemble import HistGradientBoostingClassifier
            model = HistGradientBoostingClassifier(
                max_iter=300, learning_rate=0.08, max_depth=6, random_state=seed)
        self.model = model

    def fit(self, X, y) -> "LocalHeavyBackend":
        self.model.fit(X, y)
        return self

    def classify(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        p = self.model.predict_proba(X)
        idx = np.argmax(p, axis=1)
        return self.model.classes_[idx], p[np.arange(len(idx)), idx]


class QuantumEscalationBackend(EscalationBackend):
    """Gate-model quantum-kernel classifier (statevector-simulated here)."""

    name = "quantum_kernel_svm"

    def __init__(self, n_qubits: int = 4, reps: int = 2, seed: int = 0):
        from .quantum import QuantumKernelClassifier
        self.model = QuantumKernelClassifier(n_qubits=n_qubits, reps=reps, seed=seed)

    def fit(self, X, y) -> "QuantumEscalationBackend":
        self.model.fit(X, y)
        return self

    def classify(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        p = self.model.predict_proba(X)
        idx = np.argmax(p, axis=1)
        return self.model.classes_[idx], p[np.arange(len(idx)), idx]

    def resource_estimate(self) -> dict:
        return self.model.resource_estimate()


class EscalationService:
    """Asynchronous, confidence-gated second opinion."""

    def __init__(self, backend: Optional[EscalationBackend] = None,
                 latency_samples: int = 60, max_in_flight: int = 64,
                 min_gain: float = 0.0):
        self.backend = backend
        self.latency = latency_samples
        self.max_in_flight = max_in_flight
        self.min_gain = min_gain
        self._next_id = 0
        self._pending: List[Ticket] = []
        self.submitted = 0
        self.resolved = 0
        self.dropped_overflow = 0
        self.overridden = 0

    @property
    def available(self) -> bool:
        return self.backend is not None

    @property
    def in_flight(self) -> int:
        return len(self._pending)

    def submit(self, k: int, channel: str, x: np.ndarray,
               local_cause: RootCause, local_conf: float) -> Optional[Ticket]:
        if not self.available:
            return None
        if len(self._pending) >= self.max_in_flight:
            self.dropped_overflow += 1
            return None
        t = Ticket(self._next_id, k, k + self.latency, channel,
                   np.asarray(x, float), local_cause, local_conf)
        self._next_id += 1
        self._pending.append(t)
        self.submitted += 1
        return t

    def poll(self, k: int) -> List[Verdict]:
        """Return verdicts that have become due at sample k."""
        if not self._pending:
            return []
        due = [t for t in self._pending if t.k_due <= k]
        if not due:
            return []
        self._pending = [t for t in self._pending if t.k_due > k]
        X = np.vstack([t.x for t in due])
        labels, confs = self.backend.classify(X)
        out = []
        for t, lab, c in zip(due, labels, confs):
            self.resolved += 1
            cause = RootCause(int(lab))
            if cause is not t.local_cause:
                self.overridden += 1
            out.append(Verdict(t, cause, float(c), self.backend.name))
        return out

    def flush(self) -> List[Verdict]:
        """Resolve everything still in flight (end of run)."""
        if not self._pending:
            return []
        return self.poll(max(t.k_due for t in self._pending))

    def stats(self) -> Dict[str, float]:
        return {
            "escalations_submitted": float(self.submitted),
            "escalations_resolved": float(self.resolved),
            "escalations_overridden": float(self.overridden),
            "escalations_dropped_overflow": float(self.dropped_overflow),
            "escalation_latency_samples": float(self.latency),
            "backend": self.backend.name if self.backend else "none",
        }
