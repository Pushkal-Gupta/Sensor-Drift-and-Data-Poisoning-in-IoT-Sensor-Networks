"""
E3 -- Lightweight statistical trigger, running on the residual.

This is the only part of Gen-1 FHAD that survives structurally, and it is
deliberately demoted: CUSUM is no longer the detector, it is the cheap gate that
decides when the expensive root-cause stage is allowed to run.  Because it now
operates on a residual whose nominal distribution is stationary by construction,
it needs no rolling-mean detrending and no seasonal handling.

Three interchangeable embodiments are provided (CUSUM / EWMA / Page-Hinkley) plus
an instantaneous spike test, so that the claim does not depend on any one of them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from .config import TriggerConfig


@dataclass
class TriggerState:
    fired: bool = False
    stat: float = 0.0
    which: str = ""


class TwoSidedCUSUM:
    def __init__(self, k: float, h: float):
        self.k, self.h = k, h
        self.hi = 0.0
        self.lo = 0.0

    def update(self, z: float) -> float:
        self.hi = max(0.0, self.hi + z - self.k)
        self.lo = max(0.0, self.lo - z - self.k)
        return max(self.hi, self.lo)

    def reset(self) -> None:
        self.hi = self.lo = 0.0


class EWMAChart:
    def __init__(self, lam: float, L: float):
        self.lam, self.L = lam, L
        self.e = 0.0
        self._var_scale = lam / (2 - lam)

    def update(self, z: float) -> float:
        self.e = self.lam * z + (1 - self.lam) * self.e
        return abs(self.e) / (self._var_scale ** 0.5)

    def reset(self) -> None:
        self.e = 0.0


class PageHinkley:
    def __init__(self, delta: float, lam: float):
        self.delta, self.lam = delta, lam
        self.m = 0.0
        self.mean = 0.0
        self.n = 0
        self.m_min = 0.0

    def update(self, z: float) -> float:
        self.n += 1
        self.mean += (z - self.mean) / self.n
        self.m += z - self.mean - self.delta
        self.m_min = min(self.m_min, self.m)
        return self.m - self.m_min

    def reset(self) -> None:
        self.m = self.m_min = 0.0
        self.mean = 0.0
        self.n = 0


class ChannelTrigger:
    """Bank of statistical monitors for one channel's residual."""

    def __init__(self, cfg: TriggerConfig):
        self.cfg = cfg
        self.cusum = TwoSidedCUSUM(cfg.cusum_k, cfg.cusum_h)
        self.ewma = EWMAChart(cfg.ewma_lambda, cfg.ewma_L)
        self.ph = PageHinkley(cfg.ph_delta, cfg.ph_lambda)

    def update(self, z: float) -> TriggerState:
        c = self.cusum.update(z)
        e = self.ewma.update(z)
        p = self.ph.update(z)

        # A chart that has signalled is restarted, so that a sustained deviation
        # produces repeated firings rather than one latched flag.  Between
        # firings the governor keeps the root-cause stage engaged.
        if abs(z) >= self.cfg.spike_z:
            return TriggerState(True, abs(z), "spike")
        if c >= self.cfg.cusum_h:
            self.cusum.reset()
            return TriggerState(True, c / self.cfg.cusum_h, "cusum")
        if e >= self.cfg.ewma_L:
            self.ewma.reset()
            return TriggerState(True, e / self.cfg.ewma_L, "ewma")
        if p >= self.cfg.ph_lambda:
            self.ph.reset()
            return TriggerState(True, p / self.cfg.ph_lambda, "page_hinkley")
        return TriggerState(False, max(c / self.cfg.cusum_h, e / self.cfg.ewma_L), "")

    def reset(self) -> None:
        self.cusum.reset()
        self.ewma.reset()
        self.ph.reset()


class TriggerBank:
    def __init__(self, channels, cfg: Optional[TriggerConfig] = None):
        cfg = cfg or TriggerConfig()
        self.triggers: Dict[str, ChannelTrigger] = {
            ch: ChannelTrigger(cfg) for ch in channels
        }

    def __getitem__(self, ch: str) -> ChannelTrigger:
        return self.triggers[ch]

    def reset(self, ch: Optional[str] = None) -> None:
        if ch is None:
            for t in self.triggers.values():
                t.reset()
        else:
            self.triggers[ch].reset()
