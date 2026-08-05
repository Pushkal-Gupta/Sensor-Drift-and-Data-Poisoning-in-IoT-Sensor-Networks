"""
E4 -- Cross-sensor physical-consistency graph and fault localisation.

This stage answers a question that is *orthogonal* to drift-versus-poisoning:

    is the deviation on the instrument side, or has the plant itself moved?

Getting that wrong is the dangerous failure mode of every "detect anomaly ->
recalibrate" system: a genuine process excursion gets calibrated away, and the
system is now blind to a real fault.

Method.  A set of physical constraints g_i(y) = 0 is evaluated on the reported
measurements.  Their violations form a vector in constraint space.  Each
candidate cause -- a bias on channel c, or a perturbation of a process parameter
-- has a *signature*: the direction that cause pushes the violation vector.  The
signatures are obtained by perturbing the process model itself, so no labelled
fault data is needed.  Localisation is then the hypothesis whose signature best
explains the observed violation direction.

Constraints for the vessel testbed (all evaluated on *reported measurements*
only -- deliberately not on the process model's internal state, because a
closed-loop state estimate tracks a slow bias and would hide it):

    g_gas       P - P(T)                        isochoric ideal gas
    g_humidity  RH - RH(T)                      conserved absolute humidity
    g_dyn       [T(k) - T(k-m)] - sum of the m predicted increments, integrated
                open loop from the measurements themselves (Newton + heater cmd)
    g_energy    ema(T - T_amb) - Q*duty/k       quasi-steady energy balance
    g_control   ema(T) - setpoint               the loop is supposed to regulate

The five have deliberately different sensitivities:

    a bias on T      violates g_gas and g_humidity strongly -- no process fault
                     can do that, because P and RH really did follow the real
                     temperature.
    a bias on T_amb  touches only the two balance constraints.
    a process fault  leaves both instrument constraints untouched but breaks the
                     open-loop increment check, the energy balance AND the
                     control constraint: the loop stopped regulating.

g_control is what separates "the ambient sensor reads high" from "there is an
unmodelled heat input": only the latter drives the plant off its setpoint.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

from .twin import PhysicsVesselTwin

PROCESS_HYPOTHESIS = "PROCESS"


@dataclass
class Localisation:
    hypothesis: str = ""
    score: float = 0.0          # fraction of violation energy explained
    runner_up: str = ""
    runner_up_score: float = 0.0
    magnitude: float = 0.0      # ||g|| in sigma units
    violations: Dict[str, float] = field(default_factory=dict)

    @property
    def is_process_side(self) -> bool:
        return self.hypothesis == PROCESS_HYPOTHESIS

    @property
    def margin(self) -> float:
        return self.score - self.runner_up_score


class VesselConsistencyMonitor:
    """Constraint evaluation, baseline calibration and hypothesis localisation."""

    CONSTRAINTS = ("g_gas", "g_humidity", "g_dyn", "g_energy", "g_control")

    def __init__(self, twin: PhysicsVesselTwin, window: int = 120,
                 duty_span: int = 180, horizon: int = 30,
                 setpoint: Optional[float] = None):
        self.twin = twin
        self.window = window
        self.duty_span = duty_span
        self.horizon = horizon
        self.setpoint = setpoint if setpoint is not None else twin.p.setpoint
        self.buf: Dict[str, Deque[float]] = {
            c: deque(maxlen=window) for c in self.CONSTRAINTS
        }
        self._hist: Deque[tuple] = deque(maxlen=horizon + 1)
        self._ema: Dict[str, float] = {}
        self._warm: Dict[str, List[float]] = {c: [] for c in self.CONSTRAINTS}
        self.mean = {c: 0.0 for c in self.CONSTRAINTS}
        self.sigma = {c: 1.0 for c in self.CONSTRAINTS}
        self.ready = False
        self.signatures: Dict[str, np.ndarray] = {}
        self._op_point: Optional[Dict[str, float]] = None

    # -- constraint evaluation --------------------------------------------
    def _ema_update(self, key: str, x: float, span: int) -> float:
        a = 2.0 / (span + 1.0)
        self._ema[key] = x if key not in self._ema else (1 - a) * self._ema[key] + a * x
        return self._ema[key]

    def evaluate(self, values: Dict[str, float], expected: Dict[str, float],
                 actuation: Dict[str, float]) -> Dict[str, float]:
        T, Ta = values["T_in"], values["T_amb"]
        P, RH = values["P"], values["RH"]
        u = actuation.get("u_heater", 0.0)
        tw = self.twin

        g = {
            "g_gas": P - tw.P_of_T(T),
            "g_humidity": RH - tw.RH_of_T(T),
        }

        # open-loop increment check over the last `horizon` samples, integrated
        # from the measurements themselves so that a constant bias cancels
        self._hist.append((T, Ta, u))
        if len(self._hist) > self.horizon:
            T0 = self._hist[0][0]
            pred_delta = 0.0
            for (Tj, Taj, uj) in list(self._hist)[:-1]:
                pred_delta += tw.dt * (tw.Q_hat * uj - tw.k_hat * (Tj - Taj)) / tw.C_hat
            raw = (T - T0) - pred_delta
        else:
            raw = 0.0
        g["g_dyn"] = self._ema_update("g_dyn", raw, 60)

        duty = self._ema_update("duty", u, self.duty_span)
        Tm = self._ema_update("T_in", T, self.duty_span)
        Tam = self._ema_update("T_amb", Ta, self.duty_span)
        g["g_energy"] = (Tm - Tam) - tw.Q_hat * duty / tw.k_hat
        g["g_control"] = Tm - self.setpoint
        return g

    def push(self, g: Dict[str, float]) -> None:
        for c in self.CONSTRAINTS:
            self.buf[c].append(g[c])
        if not self.ready:
            for c in self.CONSTRAINTS:
                self._warm[c].append(g[c])

    def sigma_floors(self) -> Dict[str, float]:
        """Physical resolution floors for the constraint scales.

        A constraint must not be normalised below the resolution at which it is
        physically meaningful.  The control constraint is the important case: its
        statistical noise after heavy smoothing is far smaller than the band the
        controller actually regulates within, and normalising by that noise makes
        every instrument bias look like a loss of regulation -- which is exactly
        the drift / process-fault confusion this floor removes.
        """
        p = self.twin.p
        # Chosen by sweeping the floor and reading the localisation margins:
        # too low and any instrument bias masquerades as a loss of regulation;
        # too high and a process fault becomes indistinguishable from a biased
        # ambient sensor, since neither then has a distinguishing component.
        return {"g_control": p.hysteresis / 3.0, "g_energy": p.hysteresis / 2.0}

    def finalise_baseline(self, values: Dict[str, float]) -> None:
        """Freeze constraint statistics and build the signature dictionary."""
        floors = self.sigma_floors()
        for c in self.CONSTRAINTS:
            a = np.asarray(self._warm[c][len(self._warm[c]) // 4:], dtype=float)
            if a.size < 8:
                continue
            med = float(np.median(a))
            mad = float(np.median(np.abs(a - med)))
            self.mean[c] = med
            self.sigma[c] = max(1.4826 * mad if mad > 0 else float(a.std()),
                                floors.get(c, 0.0), 1e-9)
        self._op_point = dict(values)
        self.signatures = self._build_signatures(values)
        self.ready = True

    # -- signature dictionary ---------------------------------------------
    def _build_signatures(self, v: Dict[str, float]) -> Dict[str, np.ndarray]:
        """Perturb each hypothesis in the *model* and record the induced direction.

        No labelled fault data is used: the directions come from the physics.
        """
        tw = self.twin
        T = v["T_in"]
        s = self.sigma
        m_dt = self.horizon * tw.dt
        sig: Dict[str, np.ndarray] = {}

        # --- a +1 unit bias on each measured channel ------------------------
        # T bias: the two instrument constraints move hard; in g_dyn the measured
        # increment cancels but the loss term does not; the balance and control
        # constraints shift by the bias itself.
        sig["T_in"] = np.array([
            -tw.dP_dT() / s["g_gas"],
            -tw.dRH_dT(T) / s["g_humidity"],
            (m_dt * tw.k_hat / tw.C_hat) / s["g_dyn"],
            1.0 / s["g_energy"],
            1.0 / s["g_control"],
        ])
        sig["T_amb"] = np.array([
            0.0, 0.0,
            -(m_dt * tw.k_hat / tw.C_hat) / s["g_dyn"],
            -1.0 / s["g_energy"],
            0.0,                      # a mis-reading ambient sensor cannot move
        ])                            # the plant off its setpoint
        sig["P"] = np.array([1.0 / s["g_gas"], 0.0, 0.0, 0.0, 0.0])
        sig["RH"] = np.array([0.0, 1.0 / s["g_humidity"], 0.0, 0.0, 0.0])

        # --- a +1 W unmodelled heat input in the plant ----------------------
        # instrument constraints untouched; the open-loop increment check, the
        # energy balance and the control constraint all move together.
        sig[PROCESS_HYPOTHESIS] = np.array([
            0.0, 0.0,
            (m_dt / tw.C_hat) / s["g_dyn"],
            (1.0 / tw.k_hat) / s["g_energy"],
            (1.0 / tw.k_hat) / s["g_control"],
        ])
        return {k: w / (np.linalg.norm(w) + 1e-18) for k, w in sig.items()}

    # -- localisation ------------------------------------------------------
    def normalised_violation(self, span: Optional[int] = None) -> np.ndarray:
        span = span or self.window
        out = []
        for c in self.CONSTRAINTS:
            a = np.asarray(self.buf[c], dtype=float)[-span:]
            m = float(a.mean()) if a.size else 0.0
            out.append((m - self.mean[c]) / self.sigma[c])
        return np.asarray(out)

    #: Optional floor below which localisation refuses to attribute at all.
    #: Left at zero after measurement: suppressing weak attributions was expected
    #: to help (a compensated channel has little left to attribute, so the winner
    #: is decided by noise) but measured *worse* on the held-out bank -- the
    #: localisation score and margin already encode that weakness, and the
    #: root-cause model uses them.  Raise it if a deployment's constraint set is
    #: noisier than this one's.
    MIN_MAGNITUDE = 0.0

    def localise(self, span: Optional[int] = None) -> Localisation:
        if not self.ready:
            return Localisation()
        g = self.normalised_violation(span)
        norm = float(np.linalg.norm(g))
        viol = {c: float(x) for c, x in zip(self.CONSTRAINTS, g)}
        # Attributing a sub-threshold violation is worse than saying nothing: the
        # winning hypothesis is then decided by noise, and a channel whose bias
        # has just been successfully compensated away would be re-attributed to
        # an arbitrary cause.  Report "unresolved" and let the residual character
        # and the standing channel state decide.
        if norm < self.MIN_MAGNITUDE:
            return Localisation("", 0.0, "", 0.0, norm, viol)

        scored: List[Tuple[str, float]] = []
        for name, col in self.signatures.items():
            proj = float(np.dot(col, g))
            scored.append((name, (proj / norm) ** 2))
        scored.sort(key=lambda kv: kv[1], reverse=True)
        best, second = scored[0], (scored[1] if len(scored) > 1 else ("", 0.0))
        return Localisation(best[0], best[1], second[0], second[1], norm, viol)


class NullConsistencyMonitor:
    """Degenerate monitor for single-channel deployments (no physical redundancy).

    The invention still functions -- residual generation, statistical triggering,
    root-cause disambiguation on residual character, and differential response all
    remain -- but process-versus-instrument separation is unavailable, so every
    localisation is reported as "unresolved" and the root-cause stage falls back
    on the temporal-character features alone.
    """

    ready = True
    signatures: Dict[str, np.ndarray] = {}

    def evaluate(self, values, expected, actuation) -> Dict[str, float]:
        return {}

    def push(self, g) -> None:
        pass

    def finalise_baseline(self, values) -> None:
        pass

    def localise(self, span=None) -> Localisation:
        return Localisation()
