"""
Experimental testbed: a physical plant, its transducers, and attack models.

This module is *evidence-generating apparatus*, not part of the claimed system.
It exists so that drift and poisoning can be injected with matched magnitude and
matched rate -- the regime in which a conventional anomaly detector is provably
at chance, and in which the claimed disambiguation must earn its keep.

Plant
-----
A sealed, heated vessel.  State variable is the internal temperature T_in:

    C dT/dt = Q*u(t) - k (T_in - T_amb)              (Newton heating/cooling)

with u(t) the bang-bang heater command.  Two further channels are algebraically
slaved to T_in by textbook physics:

    P   = P_ref (T_in + 273.15)/(T_ref + 273.15)     (isochoric ideal gas)
    RH  = 100 e_vap / e_s(T_in),  e_s = Magnus       (conserved absolute humidity)

so the instrument set carries genuine physical redundancy.

Fault / attack models
---------------------
benign drift      the *transducer* degrades: reported = true + b(t).  The sensor
                  still observes the real process, so its noise colour and its
                  response to the heater cycle are preserved.
spike poisoning   isolated large substitutions.
stealth poisoning the stream is *replaced* by an attacker surrogate (a lagged
                  estimate of the process) plus a bias ramp.  Magnitude and ramp
                  rate are matched to benign drift on purpose.
replay poisoning  an earlier genuine window is replayed.  Correct noise colour,
                  wrong phase relative to the current heater command.
process event     the plant itself misbehaves (heater stuck on).  All channels
                  remain mutually consistent; the excursion is real.

The stealth-poisoning model encodes the one asymmetry an attacker cannot cheaply
remove: to forge a stream that survives both the cross-sensor consistency test
and the excitation-coherence test, the attacker needs a real-time, exact model of
the plant *and* of the controller's switching phase.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .config import CHANNELS, DEFAULT_SENSORS, PlantParams, SensorParams
from .types import RootCause


# --------------------------------------------------------------------------
# physics helpers
# --------------------------------------------------------------------------
def magnus_es(T_c: np.ndarray | float) -> np.ndarray | float:
    """Saturation vapour pressure (Pa) over water, Magnus form."""
    return 610.94 * np.exp(17.625 * T_c / (T_c + 243.04))


def magnus_es_deriv(T_c: float) -> float:
    """d e_s / dT (Pa/K)."""
    return magnus_es(T_c) * 17.625 * 243.04 / (T_c + 243.04) ** 2


def pressure_from_T(T_c, p: PlantParams):
    return p.P_ref * (T_c + 273.15) / (p.T_ref + 273.15)


def rh_from_T(T_c, p: PlantParams):
    return 100.0 * p.e_vap / magnus_es(T_c)


# --------------------------------------------------------------------------
# episode container
# --------------------------------------------------------------------------
@dataclass
class Episode:
    """One simulated run."""

    t: np.ndarray
    truth: Dict[str, np.ndarray]          # noiseless physical values
    obs: Dict[str, np.ndarray]            # what the fog node actually receives
    actuation: Dict[str, np.ndarray]      # known command signals
    labels: np.ndarray                    # per-sample ground-truth RootCause
    target_channel: Optional[str]         # channel carrying the fault, if any
    scenario: str
    params: PlantParams
    meta: Dict[str, float] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.t)

    def event_slice(self) -> Tuple[int, int]:
        idx = np.flatnonzero(self.labels != RootCause.NORMAL)
        if idx.size == 0:
            return (0, 0)
        return (int(idx[0]), int(idx[-1]) + 1)


# --------------------------------------------------------------------------
# plant + transducers
# --------------------------------------------------------------------------
def simulate_plant(n: int, p: PlantParams, rng: np.random.Generator,
                   stuck_heater_from: Optional[int] = None
                   ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Integrate the plant.  Returns (truth channels, actuation channels)."""
    t = np.arange(n, dtype=float) * p.dt
    T_amb = (p.amb_mean
             + p.amb_amp * np.sin(2 * np.pi * t / p.amb_period)
             + 0.3 * np.sin(2 * np.pi * t / (p.amb_period * 0.37) + 1.1))
    # slow weather random walk, so the ambient is not perfectly predictable
    T_amb = T_amb + np.cumsum(rng.normal(0.0, 0.004, n))

    T_in = np.empty(n)
    u_cmd = np.zeros(n)      # controller output -- this is what the fog node knows
    u_phys = np.zeros(n)     # what the heater actually does
    T = p.T_init
    heating = True
    for k in range(n):
        if heating and T > p.setpoint + p.hysteresis / 2:
            heating = False
        elif (not heating) and T < p.setpoint - p.hysteresis / 2:
            heating = True
        u_cmd[k] = 1.0 if heating else 0.0
        # A genuine process fault: the contactor welds shut.  The command keeps
        # cycling normally, the physical heater does not.  All four sensors stay
        # mutually consistent -- only the plant has moved.
        u_phys[k] = 1.0 if (stuck_heater_from is not None
                            and k >= stuck_heater_from) else u_cmd[k]
        T_in[k] = T
        dT = (p.Q * u_phys[k] - p.k_loss * (T - T_amb[k])) / p.C
        T = T + dT * p.dt
    u = u_cmd

    truth = {
        "T_in": T_in,
        "T_amb": T_amb,
        "P": pressure_from_T(T_in, p),
        "RH": rh_from_T(T_in, p),
    }
    return truth, {"u_heater": u_cmd, "u_phys": u_phys}


def _coloured_noise(n: int, sp: SensorParams, rng: np.random.Generator) -> np.ndarray:
    """AR(1) noise with the requested marginal sigma."""
    w = rng.normal(0.0, 1.0, n)
    e = np.empty(n)
    e[0] = w[0]
    a = np.sqrt(max(1e-12, 1.0 - sp.phi ** 2))
    for k in range(1, n):
        e[k] = sp.phi * e[k - 1] + a * w[k]
    return sp.sigma * e


def transduce(truth: Dict[str, np.ndarray],
              sensors: Dict[str, SensorParams],
              rng: np.random.Generator) -> Dict[str, np.ndarray]:
    """Apply the transducer model (coloured noise + quantisation) to each channel."""
    obs = {}
    for ch, arr in truth.items():
        sp = sensors[ch]
        y = arr + _coloured_noise(len(arr), sp, rng)
        if sp.lsb > 0:
            y = np.round(y / sp.lsb) * sp.lsb
        obs[ch] = y
    return obs


# --------------------------------------------------------------------------
# fault / attack injectors
# --------------------------------------------------------------------------
def inject_drift(obs: Dict[str, np.ndarray], labels: np.ndarray, channel: str,
                 start: int, rate: float, curvature: float = 0.0) -> None:
    """Benign transducer drift: a smooth additive bias on top of the real signal.

    Everything else about the channel -- its noise colour, its response to the
    heater cycle -- is untouched, because the sensor is still measuring the real
    process.  That preservation is what the root-cause stage detects.
    """
    n = len(labels)
    idx = np.arange(start, n)
    tau = (idx - start).astype(float)
    bias = rate * tau + curvature * tau ** 2
    obs[channel][start:] += bias
    labels[start:] = RootCause.SENSOR_DRIFT


def inject_spike_poison(obs: Dict[str, np.ndarray], labels: np.ndarray,
                        channel: str, indices, magnitude: float,
                        rng: np.random.Generator) -> None:
    """Classic coarse poisoning: isolated substituted values."""
    for i in indices:
        if 0 <= i < len(labels):
            sign = 1.0 if rng.random() < 0.5 else -1.0
            obs[channel][i] += sign * magnitude * rng.uniform(0.8, 1.4)
            labels[i] = RootCause.DATA_POISONING


def inject_stealth_poison(obs: Dict[str, np.ndarray], truth: Dict[str, np.ndarray],
                          labels: np.ndarray, channel: str, start: int,
                          rate: float, rng: np.random.Generator,
                          surrogate_alpha: float = 0.02,
                          attacker_sigma: Optional[float] = None) -> None:
    """False-data injection matched in magnitude and rate to benign drift.

    The attacker substitutes its own surrogate of the process (an exponentially
    smoothed, therefore *stale*, estimate) and adds the same bias ramp a drifting
    transducer would produce.  Amplitude-wise the two are indistinguishable; what
    differs is that the substituted stream no longer carries the plant's real
    micro-dynamics or the transducer's real noise colour.
    """
    n = len(labels)
    true_arr = truth[channel]
    sigma = attacker_sigma if attacker_sigma is not None else 0.05

    # surrogate: causal EMA of the true signal (attacker observes it slowly)
    sur = np.empty(n)
    s = true_arr[max(0, start - 1)]
    for k in range(n):
        s = (1 - surrogate_alpha) * s + surrogate_alpha * true_arr[k]
        sur[k] = s

    tau = np.arange(n - start, dtype=float)
    forged = sur[start:] + rate * tau + rng.normal(0.0, sigma, n - start)
    obs[channel][start:] = forged
    labels[start:] = RootCause.DATA_POISONING


def inject_replay_poison(obs: Dict[str, np.ndarray], labels: np.ndarray,
                         channel: str, start: int, length: int,
                         source_offset: int, bias: float = 0.0) -> None:
    """Replay a genuine earlier window: right statistics, wrong phase."""
    n = len(labels)
    end = min(n, start + length)
    src0 = max(0, start - source_offset)
    src1 = src0 + (end - start)
    if src1 > n:
        return
    obs[channel][start:end] = obs[channel][src0:src1] + bias
    labels[start:end] = RootCause.DATA_POISONING


# --------------------------------------------------------------------------
# scenario factory
# --------------------------------------------------------------------------
def make_episode(scenario: str,
                 n: int = 4000,
                 seed: int = 0,
                 target: str = "T_in",
                 onset: Optional[int] = None,
                 rate: Optional[float] = None,
                 plant: Optional[PlantParams] = None,
                 sensors: Optional[Dict[str, SensorParams]] = None) -> Episode:
    """Build one labelled episode.

    scenario in {clean, drift, stealth_poison, spike_poison, replay_poison,
                 process_event}

    `rate` defaults to 1.5e-3 units/sample for both drift and stealth poisoning
    so that the two are rate-matched by construction.
    """
    p = plant or PlantParams()
    sensors = sensors or DEFAULT_SENSORS
    rng = np.random.default_rng(seed)
    onset = onset if onset is not None else n // 3
    rate = rate if rate is not None else 1.5e-3

    stuck = onset if scenario == "process_event" else None
    truth, actuation = simulate_plant(n, p, rng, stuck_heater_from=stuck)
    obs = transduce(truth, sensors, rng)
    labels = np.full(n, RootCause.NORMAL, dtype=int)
    tgt: Optional[str] = target

    if scenario == "clean":
        tgt = None
    elif scenario == "drift":
        inject_drift(obs, labels, target, onset, rate)
    elif scenario == "stealth_poison":
        inject_stealth_poison(obs, truth, labels, target, onset, rate, rng)
    elif scenario == "spike_poison":
        idx = rng.choice(np.arange(onset, n), size=max(3, (n - onset) // 200),
                         replace=False)
        inject_spike_poison(obs, labels, target, idx, magnitude=8.0, rng=rng)
    elif scenario == "replay_poison":
        inject_replay_poison(obs, labels, target, onset,
                             length=(n - onset), source_offset=900, bias=0.0)
    elif scenario == "process_event":
        labels[onset:] = RootCause.PROCESS_EVENT
        tgt = None
    else:
        raise ValueError(f"unknown scenario {scenario!r}")

    return Episode(
        t=np.arange(n, dtype=float) * p.dt,
        truth=truth, obs=obs, actuation=actuation, labels=labels,
        target_channel=tgt, scenario=scenario, params=p,
        meta={"onset": float(onset), "rate": float(rate), "seed": float(seed)},
    )


def episode_bank(n_per: int = 4000, seeds=range(6), scenarios=None) -> List[Episode]:
    """A balanced bank of episodes for training / evaluating the root-cause stage."""
    scenarios = scenarios or ["clean", "drift", "stealth_poison", "spike_poison",
                              "replay_poison", "process_event"]
    eps: List[Episode] = []
    for s in seeds:
        for sc in scenarios:
            eps.append(make_episode(sc, n=n_per, seed=int(s) * 97 + hash(sc) % 1000))
    return eps


def frames(ep: Episode):
    """Iterate an episode as Frame objects."""
    from .types import Frame
    for k in range(len(ep)):
        yield Frame(
            k=k, t=float(ep.t[k]),
            values={ch: float(ep.obs[ch][k]) for ch in CHANNELS},
            # only the *commanded* heater state is available to the fog node;
            # u_phys is ground truth and is deliberately withheld.
            actuation={"u_heater": float(ep.actuation["u_heater"][k])},
        )
