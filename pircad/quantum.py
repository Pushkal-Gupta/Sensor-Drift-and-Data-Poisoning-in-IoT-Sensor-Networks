"""
Optional advanced classifier: a gate-model quantum-kernel classifier.

Position in the invention: this is NOT the detector.  It is a conditionally
invoked second-opinion service that only ever sees the small minority of cases
the fog node could not resolve with confidence (see escalation.py).  It sits off
the real-time path, is invoked asynchronously, and the system degrades to its
local verdict if the backend is slow or unavailable.  Framing it that way is
deliberate: published quantum-kernel pipelines on real NISQ hardware report
useful-but-modest accuracy and cloud round trips measured in minutes, so a claim
that puts a quantum stage in the detection path would not be enabled.

Implementation.  A ZZ-feature-map fidelity kernel

    K(x, y) = |<phi(x)|phi(y)>|^2 ,   |phi(x)> = U(x) H^n U(x) H^n |0>^n

evaluated here by exact statevector simulation in numpy (n <= 10 qubits, so the
2^n amplitudes are cheap).  The same kernel matrix is what a gate-model backend
would return from swap-test / compute-uncompute circuits; swapping this simulator
for a hardware backend changes nothing above it.  No quantum advantage is claimed
or required -- the escalation stage is a fallback, and the invention stands on the
fog-layer disambiguation with or without it.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def _hadamard_matrix(n: int) -> np.ndarray:
    H1 = np.array([[1.0, 1.0], [1.0, -1.0]]) / np.sqrt(2.0)
    H = np.array([[1.0]])
    for _ in range(n):
        H = np.kron(H, H1)
    return H


def _zz_phases(X: np.ndarray, n_qubits: int) -> np.ndarray:
    """Diagonal phase applied by one ZZ-feature-map layer, for every basis state.

    Returns an (n_samples, 2**n) array of phases.
    """
    dim = 1 << n_qubits
    bits = ((np.arange(dim)[:, None] >> np.arange(n_qubits)[None, :]) & 1)
    z = 1.0 - 2.0 * bits                      # (dim, n_qubits) in {+1,-1}

    phase = X @ z.T                           # single-qubit RZ terms
    for i in range(n_qubits):
        for j in range(i + 1, n_qubits):
            coef = (np.pi - X[:, i]) * (np.pi - X[:, j])       # (n_samples,)
            phase = phase + np.outer(coef, z[:, i] * z[:, j])
    return phase


def statevectors(X: np.ndarray, n_qubits: int, reps: int = 2) -> np.ndarray:
    """|phi(x)> for every row of X.  Shape (n_samples, 2**n_qubits), complex."""
    dim = 1 << n_qubits
    H = _hadamard_matrix(n_qubits)
    psi = np.zeros((X.shape[0], dim), dtype=complex)
    psi[:, 0] = 1.0
    ph = _zz_phases(X, n_qubits)
    for _ in range(reps):
        psi = psi @ H.T
        psi = psi * np.exp(1j * ph)
    return psi


def fidelity_kernel(A: np.ndarray, B: Optional[np.ndarray] = None,
                    n_qubits: int = 4, reps: int = 2) -> np.ndarray:
    """Quantum-kernel Gram matrix |<phi(a)|phi(b)>|^2."""
    pa = statevectors(A, n_qubits, reps)
    pb = pa if B is None else statevectors(B, n_qubits, reps)
    return np.abs(pa.conj() @ pb.T) ** 2


class QuantumKernelClassifier:
    """SVM over the quantum fidelity kernel, with classical feature compression.

    The compression stage (standardise -> PCA -> scale to [0, pi]) is what makes
    a 4-6 qubit register sufficient for a 22-dimensional residual feature vector;
    it is a standard requirement of any near-term quantum-kernel pipeline.
    """

    def __init__(self, n_qubits: int = 4, reps: int = 2, C: float = 4.0,
                 seed: int = 0):
        self.n_qubits = n_qubits
        self.reps = reps
        self.C = C
        self.seed = seed
        self._scaler = None
        self._pca = None
        self._svc = None
        self._Xtr: Optional[np.ndarray] = None

    def _encode(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        if fit:
            self._scaler = StandardScaler().fit(X)
            Z = self._scaler.transform(X)
            self._pca = PCA(n_components=self.n_qubits,
                            random_state=self.seed).fit(Z)
            W = self._pca.transform(Z)
            self._lo = W.min(axis=0)
            self._hi = W.max(axis=0)
        else:
            Z = self._scaler.transform(X)
            W = self._pca.transform(Z)
        span = np.maximum(self._hi - self._lo, 1e-9)
        return np.clip((W - self._lo) / span, 0.0, 1.0) * np.pi

    def fit(self, X: np.ndarray, y: np.ndarray) -> "QuantumKernelClassifier":
        from sklearn.svm import SVC
        Xe = self._encode(np.asarray(X, float), fit=True)
        K = fidelity_kernel(Xe, None, self.n_qubits, self.reps)
        self._svc = SVC(kernel="precomputed", C=self.C, probability=True,
                        random_state=self.seed).fit(K, y)
        self._Xtr = Xe
        return self

    def _kernel_to_train(self, X: np.ndarray) -> np.ndarray:
        Xe = self._encode(np.asarray(X, float), fit=False)
        return fidelity_kernel(Xe, self._Xtr, self.n_qubits, self.reps)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._svc.predict(self._kernel_to_train(X))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._svc.predict_proba(self._kernel_to_train(X))

    @property
    def classes_(self):
        return self._svc.classes_

    # informational: circuit resources a hardware backend would need
    def resource_estimate(self) -> dict:
        n = self.n_qubits
        two_q = self.reps * n * (n - 1)          # 2 CX per ZZ pair per rep
        return {
            "qubits": n,
            "reps": self.reps,
            "single_qubit_gates": self.reps * (n + n),
            "two_qubit_gates": two_q,
            "circuit_depth_estimate": self.reps * (1 + 2 * (n - 1)),
            "kernel_circuits_per_pair": 1,
        }
