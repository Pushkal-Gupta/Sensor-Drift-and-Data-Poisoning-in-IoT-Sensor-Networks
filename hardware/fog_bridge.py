#!/usr/bin/env python3
"""
Fog-side bridge for the PIRCAD edge node.

This is the tier the edge node reports to, and it is built from the *same*
modules as the multi-sensor fog node in ../pircad/ -- the residual engine, the
statistical trigger, the adaptation governor, the calibration manager, the
differential-response controller and the rule root-cause classifier are all
imported, not reimplemented.  Only the residual definition is specialised for a
two-channel DHT22, where the physical redundancy available is the Magnus
relation between temperature and humidity at constant vapour pressure:

    residual(k) = T_reported(k) - T_implied_by_humidity(k)
    T_implied   = magnus^-1( e_est / (RH/100) )

That residual is a genuine physics-informed quantity: a bias on the temperature
transducer moves it, a real thermal change does not, because a real change moves
both channels together.

Two ways to run it:

  live      subscribe to the node's MQTT topics
            python fog_bridge.py --host broker.hivemq.com
            (needs `pip install paho-mqtt`)

  replay    read a captured CSV, no broker and no hardware needed
            ./test_edge --csv > runs.csv
            python fog_bridge.py --replay runs.csv --config proposed

The data topic carries only what the node chose to forward; ground truth arrives
on a separate topic and is used for scoring only -- never as an input to the
pipeline.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, deque
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from pircad.actions import CalibrationManager, DifferentialResponseController
from pircad.config import FogConfig, ResidualConfig, TriggerConfig
from pircad.consistency import Localisation
from pircad.governor import AdaptationGovernor, ChannelState
from pircad.residual import ChannelResidualEngine
from pircad.rootcause import RootCauseResult, RuleRootCauseClassifier
from pircad.triggers import ChannelTrigger
from pircad.types import Action, RootCause

CHANNEL = "T_in"


class SingleNodeRule:
    """Root-cause rules for a node with exactly two coupled channels.

    ``pircad.rootcause.RuleRootCauseClassifier`` is written for the redundant
    multi-sensor case, where a constraint graph can attribute a violation to one
    instrument and several independent noise signatures corroborate each other.
    A single DHT22 has neither: there is one constraint, and the only evidence
    about *how* the deviation was produced is its size, its one-step reachability
    and its rate against the transducer's ageing envelope.  Applying the
    multi-sensor weights here mislabels a held injection as drift, so the rules
    are specialised rather than reused.  Every term is still a physical
    statement, and the ordering matters: reachability and rate must both be
    tested before the benign branch, or a smooth injection wins by default.
    """

    def __init__(self, jump_sigma: float = 8.0, rate_tol: float = 1.25,
                 evidence_floor: float = 3.0, monotone_min: float = 0.60,
                 drift_rate_min: float = 0.20):
        self.jump_sigma = jump_sigma
        self.rate_tol = rate_tol
        self.evidence_floor = evidence_floor
        self.monotone_min = monotone_min
        self.drift_rate_min = drift_rate_min

    def classify(self, channel: str, feats: Dict[str, float],
                 loc: Localisation) -> RootCauseResult:
        strength = max(abs(feats.get("r_mean_norm", 0.0)),
                       abs(feats.get("z_abs_max", 0.0)) / 2.0)
        if strength < self.evidence_floor:
            return RootCauseResult(RootCause.NORMAL, 0.80,
                                   {RootCause.NORMAL: 1.0}, "below evidence floor")

        jump = feats.get("jump_max", 0.0)
        slope_ratio = feats.get("slope_ratio", 0.0)
        monotone = feats.get("monotone_frac", 0.0)

        if jump > self.jump_sigma:
            return RootCauseResult(
                RootCause.DATA_POISONING, 0.95,
                {RootCause.DATA_POISONING: 0.95},
                f"one-step change unreachable for the process (jump={jump:.1f} sigma)")

        if slope_ratio > self.rate_tol:
            return RootCauseResult(
                RootCause.DATA_POISONING, 0.90,
                {RootCause.DATA_POISONING: 0.90},
                f"grows at {slope_ratio:.2f}x the transducer ageing envelope")

        if slope_ratio > self.drift_rate_min or monotone >= self.monotone_min:
            return RootCauseResult(
                RootCause.SENSOR_DRIFT, 0.85,
                {RootCause.SENSOR_DRIFT: 0.85},
                f"sustained at {slope_ratio:.2f}x the envelope, one-directional, "
                "and inside the ageing bound")

        return RootCauseResult(RootCause.UNKNOWN, 0.40,
                               {RootCause.UNKNOWN: 0.40}, "insufficient evidence")

# ---------------------------------------------------------------- psychrometry
def magnus_es(t_c: float) -> float:
    """Saturation vapour pressure over water (Pa), Magnus form."""
    return 610.94 * math.exp(17.625 * t_c / (t_c + 243.04))


def magnus_inv_t(es_pa: float) -> float:
    """Temperature at which e_s takes the given value."""
    z = math.log(max(es_pa, 1.0) / 610.94)
    d = 17.625 - z
    return 243.04 * z / d if abs(d) > 1e-4 else 0.0


def vapour_pressure(t_c: float, rh_pct: float) -> float:
    return 0.01 * rh_pct * magnus_es(t_c)


# --------------------------------------------------------------------- the node
class FogNode:
    """Single-node PIRCAD pipeline over a two-channel DHT22 stream."""

    def __init__(self, cfg: Optional[FogConfig] = None, warmup: int = 40,
                 envelope_rate: float = 0.030):
        self.cfg = cfg or FogConfig()
        self.cfg.residual = ResidualConfig(window=16, baseline_window=warmup, smooth=5)
        self.cfg.trigger = TriggerConfig()
        self.cfg.envelope.max_rate[CHANNEL] = envelope_rate
        self.cfg.recal_total_cap[CHANNEL] = 3.0
        self.cfg.recal_rate_limit = 0.02

        self.res = ChannelResidualEngine(CHANNEL, self.cfg.residual, envelope_rate)
        self.trig = ChannelTrigger(self.cfg.trigger)
        self.gov = AdaptationGovernor([CHANNEL], freeze_hold=12, clean_to_resume=10)
        self.cal = CalibrationManager([CHANNEL], self.cfg)
        self.ctrl = DifferentialResponseController(self.cfg)
        self.clf = SingleNodeRule()

        # Departure of the channel from physics INCLUDING whatever the
        # correction has already taken up.  ChannelResidualEngine buffers the
        # bare residual, which a working correction drives to zero -- so its
        # slope stops reporting the rate at which the channel is departing, and
        # an injection that is being (wrongly) corrected erases its own
        # evidence.  The rate test therefore runs on this instead.
        self.dev_hist: deque = deque(maxlen=self.cfg.residual.window)
        self.envelope_rate = envelope_rate
        self.e_est: Optional[float] = None
        self.vapour_alpha = 0.02
        self.k = 0
        self.engage_until = -1
        self.warmup = warmup

    def step(self, temp: float, hum: float) -> dict:
        """One sample. Returns the decision record."""
        self.k += 1
        t_corr = self.cal.apply(CHANNEL, temp)

        if self.e_est is None:
            self.e_est = vapour_pressure(t_corr, hum)

        # E1/E4: expected temperature from the *other* channel
        t_from_rh = magnus_inv_t(100.0 * self.e_est / max(hum, 1.0))
        residual = t_corr - t_from_rh

        # E2: residual statistics, learned on a clean window then frozen
        if not self.res.baseline_ready:
            self.res.observe_warmup(residual)
            self.e_est += self.vapour_alpha * (vapour_pressure(t_corr, hum) - self.e_est)
            return self._record(temp, t_corr, hum, residual, 0.0, RootCause.NORMAL,
                                1.0, Action.FORWARD, True, "warmup")

        self.dev_hist.append(residual + self.cal.total(CHANNEL))
        z = self.res.push(residual, t_from_rh)
        st = self.trig.update(z)
        self.gov.note_trigger(CHANNEL, st.fired)
        if st.fired:
            self.engage_until = self.k + self.cfg.engage_hold

        gstate = self.gov.state(CHANNEL)
        engaged = (self.k <= self.engage_until
                   or gstate in (ChannelState.QUARANTINED, ChannelState.RECOVERING))
        compensating = abs(self.cal.total(CHANNEL)) > 2.0 * self.res.base.sigma

        cause, conf = RootCause.NORMAL, 1.0
        if gstate in (ChannelState.QUARANTINED, ChannelState.RECOVERING):
            # E7: standing quarantine. Re-classifying here is what lets a held
            # injection drift into the benign branch and be absorbed.
            cause, conf = RootCause.DATA_POISONING, 0.90
        elif engaged:
            feats = self.res.features()
            feats["slope_ratio"] = abs(self._deviation_slope()) / self.envelope_rate
            # the constraint violation is on this channel by construction: a
            # single node has no other instrument to attribute it to
            mag = abs(feats.get("r_mean_norm", 0.0))
            loc = Localisation(hypothesis=CHANNEL, score=1.0, runner_up="",
                               runner_up_score=0.0, magnitude=mag)
            r = self.clf.classify(CHANNEL, feats, loc)
            cause, conf = r.cause, r.confidence
            if cause is RootCause.NORMAL and compensating and gstate is ChannelState.DRIFTING:
                cause, conf = RootCause.SENSOR_DRIFT, 0.90
        elif compensating and gstate is ChannelState.DRIFTING:
            cause, conf = RootCause.SENSOR_DRIFT, 0.90

        plan = self.ctrl.plan(self.k, CHANNEL, cause, max(conf, 0.999))
        self.gov.note_decision(self.k, CHANNEL,
                               cause if engaged else RootCause.NORMAL,
                               conf, triggered=st.fired, compensating=compensating)

        # E8: bounded, rate-limited, audited correction -- drift only
        if plan.allow_calibration and self.gov.trust(CHANNEL):
            self.cal.correct(self.k, CHANNEL, self.res.current_bias(), conf)

        # E7: the vapour-pressure reference is a model too, and is gated
        if self.gov.adaptation_allowed(CHANNEL) and cause is RootCause.NORMAL:
            self.e_est += self.vapour_alpha * (vapour_pressure(t_corr, hum) - self.e_est)
        self.res.refresh_baseline(residual, self.gov.adaptation_allowed(CHANNEL))

        return self._record(temp, t_corr, hum, residual, z, cause, conf,
                            plan.action, plan.forward, plan.note)

    def _deviation_slope(self) -> float:
        """Least-squares slope of the uncorrected departure, degC per sample."""
        n = len(self.dev_hist)
        if n < 4:
            return 0.0
        x = np.arange(n, dtype=float)
        y = np.asarray(self.dev_hist, dtype=float)
        x -= x.mean()
        den = float(np.dot(x, x))
        return float(np.dot(x, y) / den) if den > 0 else 0.0

    def _record(self, temp, t_corr, hum, residual, z, cause, conf, action,
                forwarded, note) -> dict:
        return {
            "k": self.k, "temp_in": temp, "temp_out": t_corr, "hum": hum,
            "residual": residual, "z": z, "cause": RootCause(cause).name,
            "confidence": conf, "action": Action(action).name,
            "forwarded": bool(forwarded), "calib": self.cal.total(CHANNEL),
            "note": note,
        }


# ------------------------------------------------------------------- scoring
#: how the node's four-way verdict maps onto the injection's ground-truth class
CAUSE_TO_TRUTH = {
    "NORMAL": "CLEAN", "SENSOR_DRIFT": "DRIFT",
    "DATA_POISONING": "POISON", "PROCESS_EVENT": "POISON", "UNKNOWN": "UNKNOWN",
}


class Scorer:
    def __init__(self) -> None:
        self.n = 0
        self.correct = 0
        self.confusion: Counter = Counter()
        self.poison_forwarded = 0
        self.poison_total = 0
        self.drift_forwarded = 0
        self.drift_total = 0

    def add(self, truth: str, rec: dict) -> None:
        pred = CAUSE_TO_TRUTH.get(rec["cause"], "UNKNOWN")
        self.n += 1
        if pred == truth:
            self.correct += 1
        self.confusion[(truth, pred)] += 1
        if truth == "POISON":
            self.poison_total += 1
            if rec["forwarded"]:
                self.poison_forwarded += 1
        elif truth == "DRIFT":
            self.drift_total += 1
            if rec["forwarded"]:
                self.drift_forwarded += 1

    def report(self) -> str:
        if not self.n:
            return "no samples scored"
        labels = ["CLEAN", "DRIFT", "POISON", "UNKNOWN"]
        out = [f"\nsamples scored          : {self.n}",
               f"root-cause accuracy     : {self.correct / self.n:.3f}"]
        if self.poison_total:
            out.append(f"poisoning contained     : "
                       f"{1 - self.poison_forwarded / self.poison_total:.3f} "
                       f"({self.poison_forwarded}/{self.poison_total} forwarded)")
        if self.drift_total:
            out.append(f"drift kept flowing      : "
                       f"{self.drift_forwarded / self.drift_total:.3f}")
        out.append("\nconfusion (rows = ground truth, columns = verdict)")
        out.append("            " + "".join(f"{l:>10}" for l in labels))
        for t in labels[:3]:
            row = "".join(f"{self.confusion[(t, p)]:>10}" for p in labels)
            if any(self.confusion[(t, p)] for p in labels):
                out.append(f"{t:<12}" + row)
        return "\n".join(out)


# -------------------------------------------------------------------- replay
def run_replay(path: Path, config: str, quiet: bool, out_csv: Optional[Path]) -> int:
    rows = list(csv.DictReader(path.open()))
    if not rows:
        print(f"{path}: no rows", file=sys.stderr)
        return 1
    if "config" in rows[0]:
        rows = [r for r in rows if r["config"] == config]
        if not rows:
            print(f"no rows with config={config!r}", file=sys.stderr)
            return 1

    # Replay each injection mode as its own episode: the node is re-armed
    # between modes, exactly as it would be by a reset command.
    modes: List[str] = []
    for r in rows:
        m = r.get("mode", "SINGLE")
        if not modes or modes[-1] != m:
            modes.append(m)

    scorer = Scorer()
    written: List[dict] = []
    print(f"replaying {path.name}  ({len(rows)} rows, config={config})")
    for mode in modes:
        sub = [r for r in rows if r.get("mode", "SINGLE") == mode]
        node = FogNode()
        per = Scorer()
        for r in sub:
            rec = node.step(float(r["temp_reported"]), float(r["hum"]))
            truth = r.get("truth", "CLEAN")
            if node.res.baseline_ready:
                per.add(truth, rec)
                scorer.add(truth, rec)
            rec["mode"] = mode
            rec["truth"] = truth
            written.append(rec)
        acc = per.correct / per.n if per.n else float("nan")
        contained = (1 - per.poison_forwarded / per.poison_total
                     if per.poison_total else float("nan"))
        print(f"  {mode:<16} accuracy={acc:6.3f}"
              + (f"  poisoning contained={contained:5.3f}" if per.poison_total else "")
              + f"  final correction={node.cal.total(CHANNEL):+.3f} C")

    print(scorer.report())
    if out_csv:
        with out_csv.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(written[0].keys()))
            w.writeheader()
            w.writerows(written)
        print(f"\nper-sample decisions written to {out_csv}")
    return 0


# ---------------------------------------------------------------------- live
def run_live(host: str, port: int, prefix: str, out_csv: Optional[Path]) -> int:
    try:
        import paho.mqtt.client as mqtt
    except ImportError:
        print("live mode needs paho-mqtt:  pip install paho-mqtt\n"
              "or use  --replay <csv>  which needs no broker.", file=sys.stderr)
        return 2

    node = FogNode()
    scorer = Scorer()
    truth_by_seq: Dict[int, dict] = {}
    written: List[dict] = []

    def on_connect(client, userdata, flags, rc, *a):
        print(f"connected rc={rc}; subscribing to {prefix}/#")
        client.subscribe(f"{prefix}/data")
        client.subscribe(f"{prefix}/truth")

    def on_message(client, userdata, msg):
        try:
            payload = json.loads(msg.payload.decode())
        except (ValueError, UnicodeDecodeError):
            return
        if msg.topic.endswith("/truth"):
            # Ground truth is retained for SCORING only. It is never passed to
            # node.step(); the pipeline is blind to it by construction.
            truth_by_seq[int(payload["seq"])] = payload
            return
        if not msg.topic.endswith("/data"):
            return
        rec = node.step(float(payload["temp"]), float(payload["hum"]))
        seq = int(payload.get("seq", node.k))
        gt = truth_by_seq.pop(seq, None)
        truth = gt["gt"] if gt else "?"
        if node.res.baseline_ready and truth != "?":
            scorer.add(truth, rec)
        rec["truth"] = truth
        written.append(rec)
        print(f"seq={seq:<5} in={rec['temp_in']:6.2f} out={rec['temp_out']:6.2f} "
              f"resid={rec['residual']:+6.3f} -> {rec['cause']:<15} "
              f"{rec['action']:<24} truth={truth}")
        if scorer.n and scorer.n % 25 == 0:
            print(f"   [{scorer.n} scored, accuracy {scorer.correct / scorer.n:.3f}]")

    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(host, port, 60)
    print(f"listening on {host}:{port}  (Ctrl-C to stop and print the score)")
    try:
        client.loop_forever()
    except KeyboardInterrupt:
        print(scorer.report())
        if out_csv and written:
            with out_csv.open("w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(written[0].keys()))
                w.writeheader(); w.writerows(written)
            print(f"\nper-sample decisions written to {out_csv}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--replay", type=Path, help="CSV to replay instead of MQTT")
    ap.add_argument("--config", default="proposed",
                    help="which edge configuration's rows to replay")
    ap.add_argument("--host", default="broker.hivemq.com")
    ap.add_argument("--port", type=int, default=1883)
    ap.add_argument("--prefix", default="pircad/node1")
    ap.add_argument("--out", type=Path, help="write per-sample decisions here")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()

    if a.replay:
        return run_replay(a.replay, a.config, a.quiet, a.out)
    return run_live(a.host, a.port, a.prefix, a.out)


if __name__ == "__main__":
    raise SystemExit(main())
