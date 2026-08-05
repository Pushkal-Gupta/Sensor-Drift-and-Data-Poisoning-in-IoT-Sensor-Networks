"""
The claimed system, assembled.

Per frame, at the fog node:

    raw ---> calibration ---> process model ---> residual ---> statistical
                                (E1)              (E2)         trigger (E3)
                                                                   |
                                                        (only if triggered)
                                                                   v
                              cross-sensor consistency (E4) ---> root cause (E5)
                                                                   |
                                                       confidence gate (E6)
                                                                   v
                                        differential response (E8) + adaptation
                                        governor (E7) + selective forwarding

Everything downstream of the trigger runs only on triggered channels, so the
nominal cost per sample is one model evaluation, one subtraction and one CUSUM
update.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np

from .actions import (CalibrationManager, DifferentialResponseController,
                      ForwardingAccount)
from .config import CHANNELS, FogConfig
from .consistency import (PROCESS_HYPOTHESIS, Localisation,
                          NullConsistencyMonitor, VesselConsistencyMonitor)
from .escalation import EscalationService
from .governor import AdaptationGovernor, ChannelState
from .residual import ResidualEngine
from .rootcause import build_vector
from .triggers import TriggerBank
from .twin import PhysicsVesselTwin, ProcessModel
from .types import Action, Alert, Decision, Frame, RootCause

#: severity order used to reduce per-channel outcomes to one frame verdict
_SEVERITY = {
    RootCause.NORMAL: 0,
    RootCause.SENSOR_DRIFT: 1,
    RootCause.PROCESS_EVENT: 2,
    RootCause.UNKNOWN: 3,
    RootCause.DATA_POISONING: 4,
}


def _state_cause(state: ChannelState):
    """Standing channel classification, used between trigger firings."""
    if state is ChannelState.DRIFTING:
        return RootCause.SENSOR_DRIFT, 0.90
    if state in (ChannelState.QUARANTINED, ChannelState.RECOVERING):
        return RootCause.DATA_POISONING, 0.90
    return RootCause.NORMAL, 1.0


@dataclass
class RunResult:
    channels: List[str]
    per_channel: Dict[str, List[Decision]] = field(default_factory=dict)
    frame_cause: List[int] = field(default_factory=list)
    frame_action: List[int] = field(default_factory=list)
    forwarded: List[bool] = field(default_factory=list)
    alerts: List[Alert] = field(default_factory=list)
    account: Optional[ForwardingAccount] = None
    calibration_audit: list = field(default_factory=list)
    calibration_final: Dict[str, float] = field(default_factory=dict)
    governor_log: List[str] = field(default_factory=list)
    escalation: Dict[str, float] = field(default_factory=dict)
    classifier_invocations: int = 0
    n_samples: int = 0
    runtime_s: float = 0.0
    warmup_end: int = 0
    poisoned_admitted_to_training: int = 0
    admitted_to_training: int = 0
    poisoned_admitted_by_channel: Dict[str, int] = field(default_factory=dict)
    admitted_by_channel: Dict[str, int] = field(default_factory=dict)
    #: False for comparison systems that have no adaptation gate at all, so that
    #: "retraining protection" is reported as undefined rather than as perfect.
    has_governor: bool = True

    @property
    def avg_latency_s(self) -> float:
        return self.runtime_s / max(1, self.n_samples)

    def causes(self) -> np.ndarray:
        return np.asarray(self.frame_cause, dtype=int)


class PIRCADFogNode:
    """Physics-informed root-cause disambiguating fog node."""

    def __init__(self,
                 twin: Optional[ProcessModel] = None,
                 classifier=None,
                 cfg: Optional[FogConfig] = None,
                 channels=CHANNELS,
                 consistency=None,
                 escalation: Optional[EscalationService] = None,
                 reclassify_every: int = 3,
                 feature_probe: Optional[Callable] = None):
        self.cfg = cfg or FogConfig()
        self.channels = list(channels)
        self.twin = twin if twin is not None else PhysicsVesselTwin()
        self.classifier = classifier
        self.reclassify_every = max(1, reclassify_every)
        self.feature_probe = feature_probe

        if consistency is None:
            consistency = (VesselConsistencyMonitor(self.twin)
                           if isinstance(self.twin, PhysicsVesselTwin)
                           else NullConsistencyMonitor())
        self.consistency = consistency

        self.res = ResidualEngine(self.channels, self.cfg.residual,
                                  self.cfg.envelope.max_rate)
        self.trig = TriggerBank(self.channels, self.cfg.trigger)
        self.gov = AdaptationGovernor(self.channels, self.cfg.freeze_hold,
                                      self.cfg.clean_to_resume,
                                      self.cfg.require_operator_ack)
        self.cal = CalibrationManager(self.channels, self.cfg)
        self.ctrl = DifferentialResponseController(self.cfg)
        self.esc = escalation or EscalationService(backend=None)

        self._warm_n = 0
        self._warm_done = False
        self._invocations = 0
        self._last_class: Dict[str, tuple] = {}
        self._engage_until: Dict[str, int] = {c: -1 for c in self.channels}

    # ------------------------------------------------------------------
    def run(self, frames_iter, truth_labels: Optional[np.ndarray] = None) -> RunResult:
        out = RunResult(channels=list(self.channels))
        out.per_channel = {c: [] for c in self.channels}
        out.account = ForwardingAccount(self.cfg.bytes_per_sample)
        pending_index: Dict[int, tuple] = {}     # ticket id -> (k, channel)

        t0 = time.perf_counter()
        for frame in frames_iter:
            k = frame.k
            is_poison_truth = (truth_labels is not None and k < len(truth_labels)
                               and int(truth_labels[k]) == int(RootCause.DATA_POISONING))

            decisions = self._step(frame, out, pending_index, is_poison_truth)

            # frame-level reduction
            worst = max(decisions.values(), key=lambda d: _SEVERITY[d.root_cause])
            out.frame_cause.append(int(worst.root_cause))
            out.frame_action.append(int(worst.action))
            fwd = all(d.forwarded for d in decisions.values())
            out.forwarded.append(fwd)
            out.account.note(fwd, Action(worst.action))
            for c, d in decisions.items():
                out.per_channel[c].append(d)
            out.n_samples += 1

        # resolve anything still in flight
        for v in self.esc.flush():
            self._apply_verdict(v, out, pending_index)

        out.runtime_s = time.perf_counter() - t0
        out.alerts = list(self.ctrl.alerts)
        out.calibration_audit = list(self.cal.audit)
        out.calibration_final = dict(self.cal.offset)
        out.governor_log = list(self.gov.log)
        out.escalation = self.esc.stats()
        out.classifier_invocations = self._invocations
        out.warmup_end = self.cfg.residual.baseline_window
        out.admitted_to_training = self.gov.samples_admitted_for_training
        out.poisoned_admitted_to_training = self.gov.poisoned_samples_admitted
        out.admitted_by_channel = dict(self.gov.admitted_by_channel)
        out.poisoned_admitted_by_channel = dict(self.gov.poisoned_admitted_by_channel)
        return out

    # ------------------------------------------------------------------
    def _step(self, frame: Frame, out: RunResult, pending_index: Dict[int, tuple],
              is_poison_truth: bool) -> Dict[str, Decision]:
        k = frame.k

        # E8 (input side): apply the standing calibration correction
        corrected = Frame(k=k, t=frame.t,
                          values={c: self.cal.apply(c, frame.values[c])
                                  for c in self.channels},
                          actuation=dict(frame.actuation))

        # E1: expected response
        expected = self.twin.predict(corrected)

        # E2: residuals
        residual = {c: corrected.values[c] - expected[c] for c in self.channels}

        # E4 (evaluation half): physical constraint violations
        g = self.consistency.evaluate(corrected.values, expected, corrected.actuation)
        if g:
            self.consistency.push(g)

        # ---------------- warmup: learn residual statistics ----------------
        if not self._warm_done:
            for c in self.channels:
                self.res[c].observe_warmup(residual[c])
            self._warm_n += 1
            if self._warm_n >= self.cfg.residual.baseline_window:
                self.consistency.finalise_baseline(corrected.values)
                self._warm_done = True
            self.twin.update(corrected, {c: 1.0 for c in self.channels})
            dec = {c: Decision(k, c, RootCause.NORMAL, 1.0, Action.FORWARD,
                               True, True, residual[c], 0.0, note="warmup")
                   for c in self.channels}
            return dec

        # ---------------- E2/E3: residual push + statistical trigger --------
        zs, states = {}, {}
        for c in self.channels:
            z = self.res[c].push(residual[c], expected[c])
            zs[c] = z
            states[c] = self.trig[c].update(z)
            self.gov.note_trigger(c, states[c].fired)

        # conditional-activation gate, decided before any classification so that
        # it stays independent of the root-cause verdict
        engaged_map: Dict[str, bool] = {}
        for c in self.channels:
            if states[c].fired:
                self._engage_until[c] = k + self.cfg.engage_hold
            # A quarantined channel is re-evaluated continuously; a drifting one
            # is not, because its residual is being held at zero by its own
            # correction and re-evaluating it there would just read "normal".
            engaged_map[c] = (k <= self._engage_until[c]
                              or self.gov.state(c) in (ChannelState.QUARANTINED,
                                                       ChannelState.RECOVERING))
        loc = (self.consistency.localise() if any(engaged_map.values())
               else Localisation())

        # ---------------- E5..E8 per channel --------------------------------
        decisions: Dict[str, Decision] = {}
        for c in self.channels:
            st = states[c]
            gstate = self.gov.state(c)
            engaged = engaged_map[c]
            feats = None
            cause, conf = RootCause.NORMAL, 1.0
            compensating = abs(self.cal.total(c)) > 2.0 * self.res[c].base.sigma

            if engaged:
                # the root-cause stage is re-evaluated at a configurable
                # interval; between evaluations the last verdict stands
                # a fresh firing always re-evaluates, so an isolated substituted
                # sample is never answered from a cached verdict
                due = (st.fired or k % self.reclassify_every == 0
                       or c not in self._last_class)
                if due or self.feature_probe is not None:
                    feats = self.res[c].features()
                if self.feature_probe is not None:
                    self.feature_probe(k, c, feats, loc, st.fired)
                if self.classifier is not None:
                    if due:
                        r = self.classifier.classify(c, feats, loc)
                        self._invocations += 1
                        self._last_class[c] = (r.cause, r.confidence)
                    cause, conf = self._last_class[c]
                # A compensated drifting channel reads "normal" precisely because
                # the correction is working, and what is left of the constraint
                # violations is too weak to attribute reliably.  The standing
                # classification therefore holds while the correction is in
                # force.  Two verdicts may displace it: poisoning (which must
                # revoke the correction), and a process-side localisation from
                # the physical hypothesis test (which must never be masked).
                if (compensating and gstate is ChannelState.DRIFTING
                        and cause is not RootCause.DATA_POISONING
                        and loc.hypothesis != PROCESS_HYPOTHESIS):
                    cause, conf = RootCause.SENSOR_DRIFT, 0.90
            else:
                # Not engaged.  The reported cause is the channel's *standing*
                # classification, not a fresh verdict: a drifting sensor whose
                # bias is being actively compensated is still a drifting sensor,
                # and a quarantined channel stays quarantined between triggers.
                self._last_class.pop(c, None)
                cause, conf = _state_cause(gstate)

            # ablation: collapse every cause onto the prior-art response
            acted = cause
            if not self.cfg.differential_response and cause is not RootCause.NORMAL:
                acted = RootCause.SENSOR_DRIFT
            plan = self.ctrl.plan(k, c, acted, conf)

            # ---- escalation (E6) -----------------------------------------
            escalated = False
            if plan.action is Action.ESCALATE and self.esc.available and feats is not None:
                ticket = self.esc.submit(k, c, build_vector(c, feats, loc), cause, conf)
                if ticket is not None:
                    pending_index[ticket.id] = (k, c)
                    escalated = True
                else:
                    plan = self.ctrl.plan(k, c, cause, max(conf, 0.999))
            elif plan.action is Action.ESCALATE and not self.esc.available:
                # no backend: fall back to the conservative local verdict
                plan = self.ctrl.plan(k, c, cause, max(conf, 0.999))

            # ---- E7: governance ------------------------------------------
            # Only a *fresh* verdict advances the state machine; the standing
            # classification reported above must not re-arm the state that
            # produced it, or a channel could never age back to trusted.
            self.gov.note_decision(k, c,
                                   acted if engaged else RootCause.NORMAL,
                                   conf if engaged else 1.0,
                                   triggered=st.fired,
                                   compensating=compensating)
            if plan.action is Action.QUARANTINE:
                self.gov.note_quarantine_sample(c, k, frame.values[c])

            # ---- E8: bounded recalibration --------------------------------
            delta = 0.0
            if plan.allow_calibration and self.gov.trust(c):
                delta = self.cal.correct(k, c, self.res[c].current_bias(), conf)

            self.res[c].refresh_baseline(residual[c], self.gov.adaptation_allowed(c))
            self.gov.admit_for_training(c, is_poisoned_truth=is_poison_truth)

            decisions[c] = Decision(
                k=k, channel=c, root_cause=cause, confidence=conf,
                action=plan.action, forwarded=plan.forward,
                adaptation_allowed=self.gov.adaptation_allowed(c),
                residual=residual[c], residual_norm=zs[c],
                trigger_stat=st.stat, triggered=st.fired, escalated=escalated,
                calibration_offset=self.cal.total(c),
                features=feats, note=plan.note,
            )

        # ---------------- resolve due escalations ---------------------------
        for v in self.esc.poll(k):
            self._apply_verdict(v, out, pending_index)

        # ---------------- E1/E7: governed state re-anchoring ----------------
        self.twin.update(corrected, self.gov.trust_weights())
        return decisions

    # ------------------------------------------------------------------
    def _apply_verdict(self, verdict, out: RunResult,
                       pending_index: Dict[int, tuple]) -> None:
        """Retroactively settle a held sample once the advanced verdict returns."""
        key = pending_index.pop(verdict.ticket.id, None)
        if key is None:
            return
        k, c = key
        idx = k              # decisions are appended one per sample, in order
        if not (0 <= idx < len(out.per_channel.get(c, []))):
            return
        d = out.per_channel[c][idx]
        plan = self.ctrl.plan(k, c, verdict.cause, max(verdict.confidence, 0.999))
        d.root_cause = verdict.cause
        d.confidence = verdict.confidence
        d.action = plan.action
        d.forwarded = plan.forward
        d.note = f"settled by {verdict.backend}"
        self.gov.note_decision(k, c, verdict.cause, verdict.confidence,
                               triggered=True)
        if idx < len(out.frame_cause):
            if _SEVERITY[verdict.cause] > _SEVERITY[RootCause(out.frame_cause[idx])]:
                out.frame_cause[idx] = int(verdict.cause)
                out.frame_action[idx] = int(plan.action)
            if plan.forward and out.account is not None:
                out.account.forwarded_samples += 1
                out.account.held_pending = max(0, out.account.held_pending - 1)
