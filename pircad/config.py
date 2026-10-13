"""Configuration objects for the plant, the sensors and the fog node."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple


# --------------------------------------------------------------------------
# Physical plant: a heated, sealed vessel instrumented with four sensors.
# Chosen because it gives three independent, textbook physical constraints
# between the channels, which is exactly what the invention exploits.
# --------------------------------------------------------------------------
@dataclass
class PlantParams:
    dt: float = 1.0            # sampling interval, s
    C: float = 5000.0          # thermal capacitance, J/K
    k_loss: float = 20.0       # loss coefficient to ambient, W/K
    Q: float = 400.0           # heater power, W
    setpoint: float = 30.0     # bang-bang controller setpoint, degC
    hysteresis: float = 0.30   # controller hysteresis band, degC
    T_ref: float = 20.0        # gas-law reference temperature, degC
    P_ref: float = 101325.0    # gas-law reference pressure, Pa
    e_vap: float = 1426.0      # partial pressure of water vapour, Pa (conserved)
    amb_mean: float = 22.0     # ambient mean, degC
    amb_amp: float = 1.5       # ambient cycle amplitude, degC
    amb_period: float = 3600.0 # ambient cycle period, s
    T_init: float = 29.0


@dataclass
class SensorParams:
    """Per-channel transducer model: coloured (AR-1) noise + quantisation."""

    sigma: float
    phi: float = 0.6           # AR(1) coefficient -> physical 1/f-ish colour
    lsb: float = 0.0           # quantisation step, 0 = none


DEFAULT_SENSORS: Dict[str, SensorParams] = {
    "T_in":  SensorParams(sigma=0.05, phi=0.60, lsb=0.01),
    "T_amb": SensorParams(sigma=0.08, phi=0.55, lsb=0.01),
    "P":     SensorParams(sigma=25.0, phi=0.40, lsb=1.0),
    "RH":    SensorParams(sigma=0.30, phi=0.50, lsb=0.05),
}

CHANNELS: Tuple[str, ...] = ("T_in", "T_amb", "P", "RH")


# --------------------------------------------------------------------------
# Fog node
# --------------------------------------------------------------------------
@dataclass
class TriggerConfig:
    """Lightweight statistical stage running on the *residual*, not the raw signal."""

    cusum_k: float = 0.60      # slack, in units of residual sigma
    cusum_h: float = 12.0      # decision interval, in units of residual sigma
    ewma_lambda: float = 0.10
    ewma_L: float = 4.0
    ph_delta: float = 0.25     # Page-Hinkley magnitude tolerance
    ph_lambda: float = 14.0
    spike_z: float = 6.0       # instantaneous |z| that fires on its own


@dataclass
class ResidualConfig:
    window: int = 120          # residual analysis window, samples
    baseline_window: int = 900 # clean window used to learn residual statistics
    smooth: int = 9            # smoothing span for corner/slope estimation


@dataclass
class DriftEnvelope:
    """Manufacturer/aging envelope for *benign* drift of each channel.

    A deviation that grows faster than the physically plausible ageing rate of
    the transducer cannot be benign drift.  This is a per-channel physical
    prior, not a tuned ML threshold.
    """

    max_rate: Dict[str, float] = field(default_factory=lambda: {
        "T_in": 3.0e-3,    # K per sample (dt = 1 s in this testbed)
        "T_amb": 3.0e-3,
        "P": 1.5,          # Pa per sample
        "RH": 2.0e-2,      # %RH per sample
    })


@dataclass
class FogConfig:
    trigger: TriggerConfig = field(default_factory=TriggerConfig)
    residual: ResidualConfig = field(default_factory=ResidualConfig)
    envelope: DriftEnvelope = field(default_factory=DriftEnvelope)

    #: hysteresis on the conditional-activation gate: once the trigger fires the
    #: root-cause stage stays engaged for this many samples.  Keeping the gate
    #: independent of the root-cause verdict is what lets the training corpus be
    #: collected with the classifier absent.
    engage_hold: int = 20

    # --- root-cause / confidence ---
    confidence_accept: float = 0.62      # act locally at or above this
    confidence_escalate: float = 0.62    # below this -> escalate (E6)
    localisation_threshold: float = 0.55 # single-channel explanation share (E4)

    # --- bounded recalibration (E8) ---
    recal_gain: float = 0.25             # fraction of estimated bias applied
    recal_rate_limit: float = 5.0e-3     # max correction per sample (channel units)
    recal_total_cap: Dict[str, float] = field(default_factory=lambda: {
        "T_in": 10.0, "T_amb": 10.0, "P": 3000.0, "RH": 20.0,
    })

    # --- adaptation governor (E7) ---
    freeze_hold: int = 400               # samples of enforced freeze after poison
    clean_to_resume: int = 200           # consecutive clean samples before resume
    require_operator_ack: bool = False    # if True, freeze persists until ack()
    #: may sustained drift evidence re-attribute a quarantined channel to drift
    #: (True: availability) or does quarantine latch until the channel comes back
    #: clean (False: security)?  Latched is the default: in experiment 9 it had
    #: the same drift-correction error and far fewer spurious safety trips.
    allow_reattribution: bool = False

    #: Ablation switch.  With differential_response = False every detected
    #: deviation is answered the same way -- recalibrate and keep adapting --
    #: which is the behaviour of the prior art.  Everything else in the pipeline
    #: is unchanged, so the difference isolates the contribution of the
    #: cause-dependent action itself.
    differential_response: bool = True

    # --- forwarding / bandwidth (E8) ---
    summarise_every: int = 1             # 1 = forward every nominal sample
    bytes_per_sample: int = 16
