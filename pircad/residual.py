"""
E2 -- Residual generation and residual-characteristic feature extraction.

The residual r[k] = measured[k] - expected[k] is the only quantity the rest of
the pipeline looks at.  Working on the residual rather than the raw stream is
what makes the downstream stages physically meaningful: it removes the seasonal,
diurnal and actuation-driven variation that a raw-signal detector has to model
away with rolling means.

The feature set below is chosen so that it separates *causes*, not magnitudes.
Two deviations of identical size and identical growth rate -- one a drifting
transducer, one a false-data injection -- differ in these ways:

  slope_ratio      benign ageing is bounded by a manufacturer envelope; an
                   injected ramp need not respect it.
  monotone_frac    ageing bias is smooth and one-directional.
  corner           an injection has an onset: a discontinuity in the residual's
                   second difference that a physical ageing process cannot have.
  acf_shift,       a drifting transducer still observes the real process, so the
  var_ratio,       high-frequency innovation statistics of the channel are
  flatness         preserved; a substituted stream carries the attacker's noise,
                   not the transducer's.
  dyn_coherence    THE decisive one.  A healthy-but-drifting sensor still tracks
                   the modelled dynamics of the plant, so its residual increments
                   are uncorrelated with the predicted increments.  A substituted
                   stream does not follow the real micro-dynamics, so its residual
                   increments become the *negative* of the predicted increments.
                   Forging this requires a real-time exact model of the plant and
                   of the controller's switching phase.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional

import numpy as np

from .config import ResidualConfig

FEATURE_NAMES: List[str] = [
    "z_now", "z_abs_max", "r_mean_norm", "r_std_norm",
    "slope_norm", "slope_ratio", "monotone_frac",
    "corner", "jump_max", "exceed_frac",
    "acf_shift", "var_ratio", "flatness", "hf_ratio", "kurt_incr",
    "dyn_coherence", "persistence",
]


@dataclass
class BaselineStats:
    """Residual statistics learned on a clean window and then FROZEN.

    Freezing matters: if these statistics kept adapting, a sufficiently slow
    false-data injection would inflate its own normalisation and disappear.
    They are only allowed to re-adapt while the adaptation governor reports the
    channel as trusted (see governor.py).
    """

    mu: float = 0.0            # nominal residual offset (model bias, not a fault)
    sigma: float = 1.0
    dr_sigma: float = 1.0
    dr_acf1: float = 0.0
    flatness: float = 1.0
    frozen: bool = False


def _acf1(x: np.ndarray) -> float:
    if x.size < 3:
        return 0.0
    x = x - x.mean()
    d = float(np.dot(x, x))
    if d <= 1e-18:
        return 0.0
    return float(np.dot(x[:-1], x[1:]) / d)


def _spectral_flatness(x: np.ndarray) -> float:
    """Wiener entropy: 1.0 for white noise, ->0 for a tonal/smooth signal."""
    if x.size < 8:
        return 1.0
    x = x - x.mean()
    ps = np.abs(np.fft.rfft(x)) ** 2
    ps = ps[1:]
    if ps.size == 0 or not np.all(np.isfinite(ps)):
        return 1.0
    ps = np.maximum(ps, 1e-18)
    return float(np.exp(np.mean(np.log(ps))) / np.mean(ps))


def _hf_ratio(x: np.ndarray) -> float:
    if x.size < 8:
        return 0.5
    x = x - x.mean()
    ps = np.abs(np.fft.rfft(x)) ** 2
    ps = ps[1:]
    if ps.sum() <= 1e-18:
        return 0.5
    half = ps.size // 2
    return float(ps[half:].sum() / ps.sum())


def _kurtosis(x: np.ndarray) -> float:
    if x.size < 4:
        return 0.0
    x = x - x.mean()
    s = x.std()
    if s <= 1e-12:
        return 0.0
    return float(np.mean((x / s) ** 4) - 3.0)


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 4 or b.size < 4:
        return 0.0
    a = a - a.mean()
    b = b - b.mean()
    da, db = float(np.dot(a, a)), float(np.dot(b, b))
    if da <= 1e-18 or db <= 1e-18:
        return 0.0
    return float(np.dot(a, b) / np.sqrt(da * db))


def _smooth(x: np.ndarray, span: int) -> np.ndarray:
    if span <= 1 or x.size < span:
        return x
    kern = np.ones(span) / span
    return np.convolve(x, kern, mode="valid")


class ChannelResidualEngine:
    """Residual buffer and feature extractor for a single channel."""

    def __init__(self, channel: str, cfg: ResidualConfig, envelope_rate: float):
        self.channel = channel
        self.cfg = cfg
        self.envelope_rate = max(1e-12, envelope_rate)
        self.r: Deque[float] = deque(maxlen=cfg.window)
        self.pred: Deque[float] = deque(maxlen=cfg.window)
        self.base = BaselineStats()
        self._warm: List[float] = []
        self._warm_ready = False

    # -- warmup ------------------------------------------------------------
    @property
    def baseline_ready(self) -> bool:
        return self._warm_ready

    def observe_warmup(self, residual: float) -> None:
        if self._warm_ready:
            return
        self._warm.append(residual)
        if len(self._warm) >= self.cfg.baseline_window:
            self._finalise_baseline()

    def _finalise_baseline(self) -> None:
        # discard the first half: the observers and the duty estimate are still
        # converging there, and a baseline fitted on the transient is too tight
        a = np.asarray(self._warm, dtype=float)
        a = a[len(a) // 2:]
        med = float(np.mean(a))
        mad = float(np.median(np.abs(a - np.median(a))))
        # the residual is structured (actuation ripple, model bias), so the MAD
        # alone underestimates its spread and the trigger would over-fire
        sigma = max(1.4826 * mad, float(a.std()))
        d = np.diff(a)
        dmed = float(np.median(d))
        dmad = float(np.median(np.abs(d - dmed)))
        self.base = BaselineStats(
            mu=med,
            sigma=max(sigma, 1e-9),
            dr_sigma=max(1.4826 * dmad, float(d.std()), 1e-9),
            dr_acf1=_acf1(d),
            flatness=_spectral_flatness(a),
            frozen=False,
        )
        self._warm_ready = True

    # -- streaming ---------------------------------------------------------
    def push(self, residual: float, prediction: float) -> float:
        """Append a residual; return its centred, normalised value."""
        self.r.append(residual)
        self.pred.append(prediction)
        return (residual - self.base.mu) / self.base.sigma

    def refresh_baseline(self, residual: float, allowed: bool) -> None:
        """Very slow re-estimation of the nominal model bias -- only while trusted.

        The scale (sigma) is never re-estimated after warmup: an adaptive scale
        is precisely the handle a slow injection uses to normalise itself away.
        """
        if not allowed or not self._warm_ready:
            return
        a = 1e-4
        d = float(np.clip(residual - self.base.mu,
                          -2.0 * self.base.sigma, 2.0 * self.base.sigma))
        self.base.mu += a * d

    # -- features ----------------------------------------------------------
    def features(self) -> Dict[str, float]:
        n = len(self.r)
        f = {name: 0.0 for name in FEATURE_NAMES}
        if n < max(16, self.cfg.smooth + 4):
            return f

        r = np.asarray(self.r, dtype=float) - self.base.mu
        pred = np.asarray(self.pred, dtype=float)
        sig = self.base.sigma
        z = r / sig

        f["z_now"] = float(z[-1])
        f["z_abs_max"] = float(np.max(np.abs(z)))
        f["r_mean_norm"] = float(z.mean())
        f["r_std_norm"] = float(z.std())

        # slope: least squares over the window, in channel units per sample
        idx = np.arange(n, dtype=float)
        idx -= idx.mean()
        denom = float(np.dot(idx, idx))
        slope = float(np.dot(idx, r) / denom) if denom > 0 else 0.0
        f["slope_norm"] = slope * n / sig
        f["slope_ratio"] = abs(slope) / self.envelope_rate

        d = np.diff(r)
        if slope != 0.0 and d.size:
            f["monotone_frac"] = float(np.mean(np.sign(d) == np.sign(slope)))

        sm = _smooth(r, self.cfg.smooth)
        if sm.size >= 3:
            dd = np.diff(sm, n=2)
            f["corner"] = float(np.max(np.abs(dd)) / (sig / np.sqrt(self.cfg.smooth)))
        f["jump_max"] = float(np.max(np.abs(d)) / self.base.dr_sigma) if d.size else 0.0
        f["exceed_frac"] = float(np.mean(np.abs(z) > 3.0))

        f["acf_shift"] = _acf1(d) - self.base.dr_acf1
        f["var_ratio"] = float(np.log((d.std() + 1e-12) / self.base.dr_sigma))
        f["flatness"] = _spectral_flatness(r) - self.base.flatness
        f["hf_ratio"] = _hf_ratio(r)
        f["kurt_incr"] = _kurtosis(d)

        # decisive feature: does the reported stream still follow the modelled
        # micro-dynamics of the process?
        dp = np.diff(pred)
        if dp.size and float(np.var(dp)) > 1e-16:
            f["dyn_coherence"] = -_corr(d, dp)

        f["persistence"] = float(np.mean(np.abs(z) > 1.5))
        return f

    def current_bias(self) -> float:
        """Robust estimate of the standing bias, used for bounded recalibration."""
        if not self.r:
            return 0.0
        return float(np.median(np.asarray(self.r, dtype=float)) - self.base.mu)


class ResidualEngine:
    """Residual engines for every channel of the process model."""

    def __init__(self, channels, cfg: ResidualConfig, envelope: Dict[str, float]):
        self.cfg = cfg
        self.engines: Dict[str, ChannelResidualEngine] = {
            ch: ChannelResidualEngine(ch, cfg, envelope.get(ch, 1e-3))
            for ch in channels
        }

    def __getitem__(self, ch: str) -> ChannelResidualEngine:
        return self.engines[ch]

    @property
    def baseline_ready(self) -> bool:
        return all(e.baseline_ready for e in self.engines.values())

    def compute(self, values: Dict[str, float], expected: Dict[str, float]
                ) -> Dict[str, float]:
        return {ch: values[ch] - expected[ch]
                for ch in self.engines if ch in expected and ch in values}
