"""
Extended scenario factory for the journal evaluation (experiments 6-10).

`simulator.make_episode` fixes one attacker and one plant.  This module adds the
two axes a reviewer will ask about:

attacker capability   how much the adversary knows about the live process.
                      `fidelity` is the smoothing factor of the attacker's
                      surrogate (0.02 = a stale estimate, 1.0 = the exact
                      real-time process state); `noise` is either "white" or
                      "matched" (the transducer's own AR(1) colour);
                      "mitm" adds the ramp to the *genuine* reported stream, an
                      on-path attacker that is observationally identical to
                      benign drift on the attacked channel; "coordinated" forges
                      T_in, P and RH together from one fake temperature, so the
                      instrument-side constraints stay satisfied.
distribution shift    twin parameter mismatch, transducer noise, ambient swing,
                      setpoint and drift/attack rates outside the training range.

Like `simulator.py`, this is evidence-generating apparatus, not part of the
claimed system.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, Optional

import numpy as np

from .config import DEFAULT_SENSORS, PlantParams, SensorParams
from .simulator import (Episode, _coloured_noise, inject_drift,
                        inject_replay_poison, inject_spike_poison,
                        pressure_from_T, rh_from_T, simulate_plant, transduce)
from .types import RootCause


@dataclass(frozen=True)
class Attacker:
    fidelity: float = 0.02      # surrogate EMA factor; 1.0 = exact live state
    noise: str = "white"        # "white" | "matched"
    mode: str = "substitute"    # "substitute" | "mitm" | "coordinated"


@dataclass(frozen=True)
class Shift:
    noise_scale: float = 1.0    # multiplies every transducer sigma
    amb_amp_scale: float = 1.0  # multiplies the ambient swing
    setpoint: Optional[float] = None
    plant_scale: float = 1.0    # multiplies C and k_loss of the *true* plant


DEFAULT_ATTACKER = Attacker()
NO_SHIFT = Shift()


def _sensors(shift: Shift) -> Dict[str, SensorParams]:
    return {c: replace(sp, sigma=sp.sigma * shift.noise_scale)
            for c, sp in DEFAULT_SENSORS.items()}


def _plant(shift: Shift) -> PlantParams:
    p = PlantParams()
    p = replace(p, amb_amp=p.amb_amp * shift.amb_amp_scale,
                C=p.C * shift.plant_scale, k_loss=p.k_loss * shift.plant_scale)
    if shift.setpoint is not None:
        p = replace(p, setpoint=shift.setpoint, T_init=shift.setpoint - 1.0)
    return p


def _surrogate(true_arr: np.ndarray, start: int, alpha: float) -> np.ndarray:
    n = len(true_arr)
    sur = np.empty(n)
    s = true_arr[max(0, start - 1)]
    for k in range(n):
        s = (1 - alpha) * s + alpha * true_arr[k]
        sur[k] = s
    return sur


def _attack_noise(n: int, sp: SensorParams, kind: str,
                  rng: np.random.Generator) -> np.ndarray:
    if kind == "matched":
        return _coloured_noise(n, sp, rng)
    return rng.normal(0.0, sp.sigma, n)


def inject_stealth(obs, truth, labels, channel: str, start: int, rate: float,
                   att: Attacker, sensors: Dict[str, SensorParams],
                   p: PlantParams, rng: np.random.Generator) -> None:
    """Rate-matched injection under a given attacker capability."""
    n = len(labels)
    tau = np.arange(n - start, dtype=float)
    ramp = rate * tau

    if att.mode == "mitm":
        # on-path: the genuine reading plus the ramp
        obs[channel][start:] = obs[channel][start:] + ramp
    elif att.mode == "coordinated":
        # one fake temperature drives every instrument the attacker controls,
        # so P(T) and RH(T) remain mutually consistent
        sur = _surrogate(truth["T_in"], start, att.fidelity)
        T_fake = sur[start:] + ramp
        obs["T_in"][start:] = T_fake + _attack_noise(n - start, sensors["T_in"],
                                                     att.noise, rng)
        obs["P"][start:] = (pressure_from_T(T_fake, p)
                            + _attack_noise(n - start, sensors["P"], att.noise, rng))
        obs["RH"][start:] = (rh_from_T(T_fake, p)
                             + _attack_noise(n - start, sensors["RH"], att.noise, rng))
    else:
        sur = _surrogate(truth[channel], start, att.fidelity)
        obs[channel][start:] = (sur[start:] + ramp
                                + _attack_noise(n - start, sensors[channel],
                                                att.noise, rng))
    labels[start:] = RootCause.DATA_POISONING


def make_episode_v2(scenario: str, n: int = 4000, seed: int = 0,
                    onset: Optional[int] = None, rate: Optional[float] = None,
                    attacker: Attacker = DEFAULT_ATTACKER,
                    shift: Shift = NO_SHIFT, target: str = "T_in",
                    sign: int = 1) -> Episode:
    """Same scenarios as `simulator.make_episode`, with attacker and shift knobs.

    With the defaults it reproduces `make_episode` sample for sample, which the
    test suite checks.

    `sign` sets the polarity of the event: the direction of a drift or attack
    ramp, and for a process event whether the heater fails welded on (+1,
    overheating) or open (-1, cooling).  The original bank is +1 throughout,
    which lets a classifier learn the sign as a shortcut; the journal bank
    draws it at random.
    """
    p = _plant(shift)
    sensors = _sensors(shift)
    rng = np.random.default_rng(seed)
    onset = onset if onset is not None else n // 3
    rate = rate if rate is not None else 1.5e-3
    rate = rate * (1 if sign >= 0 else -1)

    stuck = onset if scenario == "process_event" else None
    truth, actuation = simulate_plant(n, p, rng, stuck_heater_from=stuck,
                                      stuck_value=1.0 if sign >= 0 else 0.0)
    obs = transduce(truth, sensors, rng)
    labels = np.full(n, RootCause.NORMAL, dtype=int)
    tgt: Optional[str] = target

    if scenario == "clean":
        tgt = None
    elif scenario == "drift":
        inject_drift(obs, labels, target, onset, rate)
    elif scenario == "stealth_poison":
        if attacker == DEFAULT_ATTACKER:
            # bit-identical to the original injector (same RNG consumption)
            from .simulator import inject_stealth_poison
            inject_stealth_poison(obs, truth, labels, target, onset, rate, rng)
        else:
            inject_stealth(obs, truth, labels, target, onset, rate, attacker,
                           sensors, p, rng)
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
        t=np.arange(n, dtype=float) * p.dt, truth=truth, obs=obs,
        actuation=actuation, labels=labels, target_channel=tgt,
        scenario=scenario, params=p,
        meta={"onset": float(onset), "rate": float(rate), "seed": float(seed),
              "fidelity": attacker.fidelity, "attack_mode": attacker.mode,
              "attack_noise": attacker.noise, "sign": float(sign)},
    )
