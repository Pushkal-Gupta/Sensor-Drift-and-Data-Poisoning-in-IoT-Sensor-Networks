"""
E7 -- Adaptation governor: inhibition of model updates on suspect data.

This is the feature with the clearest technical effect, and the one every
adaptive anomaly detector in the cited art lacks.  An adaptive system that keeps
learning from whatever it accepts can be walked: feed it a bias that grows slower
than its own adaptation, and the attack becomes the new normal.  After that the
detector is not merely fooled about the attack -- it is blind to genuine faults,
because the poisoned offset now cancels them.

The governor is a per-channel gate in front of *every* adaptive element:

    * the process model's state re-anchoring          (twin.update)
    * the residual normalisation statistics           (residual.refresh_baseline)
    * the calibration offsets                         (actions.CalibrationManager)
    * any retraining of the root-cause model itself   (admit_for_training)

A channel that is merely *triggered* already stops feeding the process model --
before any cause has been decided -- so the reference cannot be dragged while the
system is still deciding what it is looking at.  A channel classified as poisoned
is frozen for a minimum hold, then requires a run of clean samples (and, if
configured, an operator acknowledgement) before adaptation resumes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Deque, Dict, List, Optional
from collections import deque

from .types import RootCause


class ChannelState(Enum):
    TRUSTED = "trusted"
    WATCH = "watch"            # trigger fired, cause not yet decided
    DRIFTING = "drifting"      # benign drift, bounded correction in progress
    QUARANTINED = "quarantined"
    RECOVERING = "recovering"  # freeze hold served, counting clean samples


@dataclass
class ChannelGovernance:
    state: ChannelState = ChannelState.TRUSTED
    frozen_since: Optional[int] = None
    clean_run: int = 0
    quarantined_samples: int = 0
    poison_episodes: int = 0
    operator_ack: bool = False
    drift_run: int = 0          # consecutive drift verdicts while quarantined
    held: Deque[tuple] = field(default_factory=lambda: deque(maxlen=5000))


class AdaptationGovernor:
    def __init__(self, channels, freeze_hold: int = 400,
                 clean_to_resume: int = 200, require_operator_ack: bool = False,
                 allow_reattribution: bool = False):
        self.freeze_hold = freeze_hold
        #: policy: may sustained drift evidence move a quarantined channel to
        #: DRIFTING (availability), or is quarantine latched until the channel
        #: comes back clean (security)?  See the trade-off in experiment 9.
        self.allow_reattribution = allow_reattribution
        self.clean_to_resume = clean_to_resume
        self.require_operator_ack = require_operator_ack
        self.ch: Dict[str, ChannelGovernance] = {
            c: ChannelGovernance() for c in channels
        }
        # audit counters used for the "retraining protection" metric.  These are
        # kept per channel: a poisoned sample on one channel is not "admitted"
        # merely because a healthy channel of the same frame was admitted.
        self.samples_admitted_for_training = 0
        self.poisoned_samples_admitted = 0
        self.admitted_by_channel: Dict[str, int] = {c: 0 for c in channels}
        self.poisoned_admitted_by_channel: Dict[str, int] = {c: 0 for c in channels}
        self.global_freezes = 0
        self.log: List[str] = []

    # -- queries used by the rest of the pipeline --------------------------
    #: how much a channel in each state is allowed to influence the process
    #: model's state.  A channel is de-weighted the moment its trigger fires --
    #: before any cause has been decided -- and drops to zero once quarantined.
    TRUST_WEIGHT = {
        ChannelState.TRUSTED: 1.00,
        ChannelState.DRIFTING: 1.00,
        ChannelState.WATCH: 0.50,
        ChannelState.RECOVERING: 0.10,
        ChannelState.QUARANTINED: 0.00,
    }

    def trust(self, channel: str) -> bool:
        """May the process model re-anchor its state on this channel?"""
        return self.ch[channel].state in (ChannelState.TRUSTED, ChannelState.DRIFTING)

    def trust_weight(self, channel: str) -> float:
        return self.TRUST_WEIGHT[self.ch[channel].state]

    def trust_weights(self) -> Dict[str, float]:
        return {c: self.TRUST_WEIGHT[g.state] for c, g in self.ch.items()}

    def adaptation_allowed(self, channel: str) -> bool:
        """May any adaptive element consume this channel's sample?"""
        return self.ch[channel].state is ChannelState.TRUSTED

    def training_frozen(self) -> bool:
        """Global gate: no retraining of the detector while anything is suspect."""
        return any(g.state in (ChannelState.QUARANTINED, ChannelState.RECOVERING)
                   for g in self.ch.values())

    def state(self, channel: str) -> ChannelState:
        return self.ch[channel].state

    # -- transitions -------------------------------------------------------
    def note_trigger(self, channel: str, fired: bool) -> None:
        g = self.ch[channel]
        if g.state is ChannelState.TRUSTED and fired:
            g.state = ChannelState.WATCH
            self.log.append(f"{channel}: TRUSTED -> WATCH (trigger)")

    def note_decision(self, k: int, channel: str, cause: RootCause,
                      confidence: float, triggered: bool = False,
                      compensating: bool = False) -> None:
        """Advance the channel state machine.

        `compensating` is set by the caller while a standing calibration
        correction is in force.  A channel whose bias is being actively
        compensated must not age back to "trusted": the residual has been driven
        to zero *by the correction*, which is evidence that the correction is
        working, not evidence that the sensor recovered.  Clearing the flag there
        would restart the whole cycle and, worse, would re-open the adaptation
        path on a sensor that is known to be degraded.
        """
        g = self.ch[channel]

        if cause is RootCause.DATA_POISONING:
            if g.state is not ChannelState.QUARANTINED:
                g.poison_episodes += 1
                self.global_freezes += 1
                self.log.append(f"[{k}] {channel}: QUARANTINE "
                                f"(conf={confidence:.2f}) - adaptation frozen")
            g.state = ChannelState.QUARANTINED
            g.frozen_since = k if g.frozen_since is None else g.frozen_since
            g.clean_run = 0
            g.drift_run = 0
            return

        if cause is RootCause.SENSOR_DRIFT:
            if g.state in (ChannelState.QUARANTINED, ChannelState.RECOVERING):
                # A single drift verdict does not release a quarantined channel:
                # an attacker must not be able to buy recalibration by steering
                # one verdict.  But a quarantine must not be a life sentence
                # either -- one transient misread at the onset of genuine drift
                # would otherwise lock a serviceable sensor out of correction
                # for good.  Re-attribution therefore needs the freeze hold
                # served AND a sustained run of consecutive drift verdicts; the
                # correction it then unlocks is still bounded and audited.
                g.drift_run += 1
                if not self.allow_reattribution:
                    return
                served = (g.frozen_since is not None
                          and (k - g.frozen_since) >= self.freeze_hold)
                if (served and g.drift_run >= self.clean_to_resume
                        and not (self.require_operator_ack and not g.operator_ack)):
                    g.state = ChannelState.DRIFTING
                    g.frozen_since = None
                    g.drift_run = 0
                    self.log.append(f"[{k}] {channel}: re-attributed to DRIFT after "
                                    f"sustained drift evidence")
                return
            g.state = ChannelState.DRIFTING
            return
        g.drift_run = 0

        # NORMAL / PROCESS_EVENT / UNKNOWN
        if g.state is ChannelState.QUARANTINED:
            assert g.frozen_since is not None
            if (k - g.frozen_since) >= self.freeze_hold:
                g.state = ChannelState.RECOVERING
                g.clean_run = 0
                self.log.append(f"[{k}] {channel}: freeze hold served -> RECOVERING")
            return

        if g.state is ChannelState.RECOVERING:
            if self.require_operator_ack and not g.operator_ack:
                return
            if triggered:
                g.clean_run = 0
                return
            g.clean_run += 1
            if g.clean_run >= self.clean_to_resume:
                g.state = ChannelState.TRUSTED
                g.frozen_since = None
                g.operator_ack = False
                self.log.append(f"[{k}] {channel}: RECOVERING -> TRUSTED")
            return

        if g.state in (ChannelState.WATCH, ChannelState.DRIFTING):
            # a channel whose trigger is still firing is not "clean"
            g.clean_run = 0 if triggered else g.clean_run + 1
            # A drift verdict is sticky: the channel stays flagged (and its
            # bounded correction stays engaged) long after the residual has been
            # compensated away.  A bare watch clears quickly.
            need = self.clean_to_resume if g.state is ChannelState.DRIFTING else 30
            if g.state is ChannelState.DRIFTING and compensating:
                return
            if g.clean_run >= need:
                g.state = ChannelState.TRUSTED
                g.clean_run = 0
        else:
            g.clean_run = 0

    def note_quarantine_sample(self, channel: str, k: int, value: float) -> None:
        g = self.ch[channel]
        g.quarantined_samples += 1
        g.held.append((k, value))

    def operator_ack(self, channel: str) -> None:
        """Out-of-band human verification, as required by claim 14."""
        self.ch[channel].operator_ack = True
        self.log.append(f"{channel}: operator acknowledgement received")

    # -- the retraining gate ----------------------------------------------
    def admit_for_training(self, channel: str, is_poisoned_truth: Optional[bool] = None
                           ) -> bool:
        """Ask whether a sample may enter the adaptive-model training set.

        `is_poisoned_truth` is used only for the audit counters in the
        experiments; the gate itself never sees ground truth.
        """
        ok = self.adaptation_allowed(channel) and not self.training_frozen()
        if ok:
            self.samples_admitted_for_training += 1
            self.admitted_by_channel[channel] += 1
            if is_poisoned_truth:
                self.poisoned_samples_admitted += 1
                self.poisoned_admitted_by_channel[channel] += 1
        return ok
