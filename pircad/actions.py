"""
E8 -- Differential response control.

The single largest departure from Gen-1 FHAD and from D1-D4.  In the cited art a
detected anomaly has exactly one consequence: the sample is filtered out (or an
alarm is raised).  Here the consequence is a function of the *cause*:

    benign sensor drift   a bounded, rate-limited, audited calibration correction
                          is applied and the data keeps flowing.  Nothing is
                          discarded: a drifting sensor is still a useful sensor.
    data poisoning        the data is quarantined, adaptation is frozen (E7), a
                          security indication is raised, and nothing is forwarded.
    process event         the data is forwarded *and* alarmed, and calibration is
                          explicitly forbidden -- this is the case where "just
                          recalibrate" would erase a real fault.
    ambiguous             held pending escalation (E6); no irreversible action.

The recalibration path is deliberately conservative: correction is a fraction of
the estimated bias, rate-limited per sample, capped in total, and every
adjustment is written to an audit log.  A capped, rate-limited corrector cannot
absorb an attack faster than the detector can classify it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .config import FogConfig
from .types import Action, Alert, RootCause


@dataclass
class CalibrationRecord:
    k: int
    channel: str
    delta: float
    total: float
    confidence: float


class CalibrationManager:
    """Bounded, rate-limited, audited correction of benign transducer drift."""

    def __init__(self, channels, cfg: FogConfig):
        self.cfg = cfg
        self.offset: Dict[str, float] = {c: 0.0 for c in channels}
        self.audit: List[CalibrationRecord] = []

    def apply(self, channel: str, raw: float) -> float:
        return raw - self.offset[channel]

    def correct(self, k: int, channel: str, residual_bias: float,
                confidence: float) -> float:
        """Nudge the calibration offset towards the estimated standing bias."""
        cap = self.cfg.recal_total_cap.get(channel, np.inf)
        step = self.cfg.recal_gain * residual_bias
        step = float(np.clip(step, -self.cfg.recal_rate_limit,
                             self.cfg.recal_rate_limit))
        new = float(np.clip(self.offset[channel] + step, -cap, cap))
        delta = new - self.offset[channel]
        if delta != 0.0:
            self.offset[channel] = new
            self.audit.append(CalibrationRecord(k, channel, delta, new, confidence))
        return delta

    def total(self, channel: str) -> float:
        return self.offset[channel]

    def saturated(self, channel: str) -> bool:
        cap = self.cfg.recal_total_cap.get(channel, np.inf)
        return abs(self.offset[channel]) >= cap - 1e-9


@dataclass
class ResponsePlan:
    action: Action
    forward: bool
    allow_adaptation: bool
    allow_calibration: bool
    alert: Optional[Alert] = None
    note: str = ""


class DifferentialResponseController:
    """Maps (root cause, confidence) to the control action."""

    def __init__(self, cfg: FogConfig):
        self.cfg = cfg
        self.alerts: List[Alert] = []

    def plan(self, k: int, channel: str, cause: RootCause,
             confidence: float) -> ResponsePlan:
        if confidence < self.cfg.confidence_escalate and cause is not RootCause.NORMAL:
            return ResponsePlan(Action.ESCALATE, forward=False,
                                allow_adaptation=False, allow_calibration=False,
                                note="confidence below local-decision threshold")

        if cause is RootCause.NORMAL:
            return ResponsePlan(Action.FORWARD, True, True, False, note="nominal")

        if cause is RootCause.SENSOR_DRIFT:
            return ResponsePlan(
                Action.RECALIBRATE_AND_FORWARD, forward=True,
                allow_adaptation=False, allow_calibration=True,
                alert=Alert(k, channel, "maintenance",
                            "benign transducer drift; bounded recalibration applied",
                            confidence),
                note="drift: correct and keep forwarding")

        if cause is RootCause.DATA_POISONING:
            a = Alert(k, channel, "security",
                      "suspected data poisoning; data quarantined and model "
                      "adaptation inhibited", confidence)
            self.alerts.append(a)
            return ResponsePlan(Action.QUARANTINE, forward=False,
                                allow_adaptation=False, allow_calibration=False,
                                alert=a, note="poisoning: quarantine + freeze")

        if cause is RootCause.PROCESS_EVENT:
            a = Alert(k, channel, "process",
                      "genuine process excursion; instruments consistent - "
                      "forwarded, calibration inhibited", confidence)
            self.alerts.append(a)
            return ResponsePlan(Action.ALERT_AND_FORWARD, forward=True,
                                allow_adaptation=False, allow_calibration=False,
                                alert=a, note="process event: never recalibrate")

        return ResponsePlan(Action.ESCALATE, False, False, False, note="unknown cause")


@dataclass
class ForwardingAccount:
    """Bandwidth bookkeeping for the selective-forwarding pipeline."""

    bytes_per_sample: int = 16
    raw_samples: int = 0
    forwarded_samples: int = 0
    suppressed_poison: int = 0
    held_pending: int = 0
    records: List[int] = field(default_factory=list)

    def note(self, forwarded: bool, action: Action) -> None:
        self.raw_samples += 1
        if forwarded:
            self.forwarded_samples += 1
        elif action is Action.QUARANTINE:
            self.suppressed_poison += 1
        elif action is Action.ESCALATE:
            self.held_pending += 1

    @property
    def bandwidth_saved(self) -> float:
        if self.raw_samples == 0:
            return 0.0
        return 1.0 - self.forwarded_samples / self.raw_samples
