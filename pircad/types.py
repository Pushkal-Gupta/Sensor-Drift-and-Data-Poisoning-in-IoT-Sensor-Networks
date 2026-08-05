"""Shared value types for the PIRCAD fog node."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, Optional


class RootCause(IntEnum):
    """Root cause assigned to a detected deviation.

    The essential difference from Gen-1 FHAD: FHAD emitted {normal, drift,
    poison} as *anomaly labels*.  Here these are *causes*, they are decided on
    physically distinct evidence, and each one drives a different control
    action.  PROCESS_EVENT is the safety-critical fourth class: the deviation is
    real, the sensors are healthy, and the plant itself has moved -- such a
    sample must never be "recalibrated away".
    """

    NORMAL = 0
    SENSOR_DRIFT = 1
    DATA_POISONING = 2
    PROCESS_EVENT = 3
    UNKNOWN = 4


class Action(IntEnum):
    """Differential control action applied by the fog node."""

    FORWARD = 0                  # nominal: forward (possibly summarised)
    RECALIBRATE_AND_FORWARD = 1  # benign drift: bounded correction, keep flowing
    QUARANTINE = 2               # poisoning: hold data, freeze adaptation, alert
    ESCALATE = 3                 # ambiguous: hold pending advanced classification
    ALERT_AND_FORWARD = 4        # genuine process excursion: forward + alert


#: Actions after which the sample is still delivered to the cloud.
FORWARDING_ACTIONS = (Action.FORWARD, Action.RECALIBRATE_AND_FORWARD,
                      Action.ALERT_AND_FORWARD)


@dataclass
class Frame:
    """One synchronous multi-channel sample presented to the fog node."""

    k: int                              # sample index
    t: float                            # timestamp (s)
    values: Dict[str, float]            # channel -> reported measurement
    actuation: Dict[str, float] = field(default_factory=dict)
    #: known excitation / command signals of the monitored process.  These are
    #: what make the "excitation coherence" test possible (see residual.py).


@dataclass
class Decision:
    """Per-channel outcome for one frame."""

    k: int
    channel: str
    root_cause: RootCause
    confidence: float
    action: Action
    forwarded: bool
    adaptation_allowed: bool
    residual: float = 0.0
    residual_norm: float = 0.0
    trigger_stat: float = 0.0
    triggered: bool = False
    escalated: bool = False
    calibration_offset: float = 0.0
    features: Optional[Dict[str, float]] = None
    note: str = ""


@dataclass
class Alert:
    k: int
    channel: str
    kind: str
    detail: str
    confidence: float
