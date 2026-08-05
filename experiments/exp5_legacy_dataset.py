"""
Experiment 5 -- the original dataset, for continuity with the Gen-1 disclosure.

The Generation-1 work was evaluated on the Daily Minimum Temperatures record
(3 647 readings) with synthetic drift and spike injections.  That record is a
single channel with no physical redundancy, so this experiment answers a fair
question: does the invention still do anything when the redundancy it exploits is
absent?

It does, but less.  With one channel there is no cross-sensor consistency graph,
so instrument-versus-process separation is unavailable.  What remains is the
process model (here a fitted annual-harmonic climatology instead of a rolling
mean), residual generation, the statistical trigger on the residual, root-cause
disambiguation on residual character, and the differential response.  This is the
degraded-deployment case, and reporting it is the honest way to state the scope of
the claim: the strong result needs redundancy, the architecture does not.

Run:  python experiments/exp5_legacy_dataset.py
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from _common import ROOT, banner, save_json

from pircad.actions import CalibrationManager, DifferentialResponseController
from pircad.baselines import Gen1FHAD, rolling_mean_detrend
from pircad.config import FogConfig, ResidualConfig, TriggerConfig
from pircad.consistency import Localisation
from pircad.governor import AdaptationGovernor, ChannelState
from pircad.metrics import confusion, f1_from_cm, per_class_recall
from pircad.residual import ChannelResidualEngine
from pircad.rootcause import (MLRootCauseClassifier, RuleRootCauseClassifier,
                              build_vector)
from pircad.triggers import ChannelTrigger
from pircad.twin import SeasonalHarmonicTwin
from pircad.types import Action, Frame, RootCause

CSV = ROOT / "1_Daily_minimum_temps.csv"
CHANNEL = "value"
WARMUP = 700


def load() -> np.ndarray:
    df = pd.read_csv(CSV)
    df["Temp"] = pd.to_numeric(df["Temp"], errors="coerce")
    return df.dropna()["Temp"].to_numpy(dtype=float)


def inject(base: np.ndarray, seed: int) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Drift and poisoning injections in the style of the Gen-1 disclosure.

    The poisoning is made *stealthy* as well as spiky: as in Experiment 1, a
    substituted stream whose ramp rate matches the drift, so the two are not
    separable by magnitude.
    """
    rng = np.random.default_rng(seed)
    x = base.copy()
    n = len(x)
    labels = np.full(n, int(RootCause.NORMAL))

    # benign drift: a smooth additive bias on top of the real record
    d0, d1, slope = 1000, 1900, 0.004
    x[d0:d1] += slope * np.arange(d1 - d0)
    labels[d0:d1] = int(RootCause.SENSOR_DRIFT)

    # stealth injection: substituted, lagged surrogate + a matched ramp
    p0, p1 = 2300, 3100
    sur = pd.Series(base).ewm(alpha=0.05).mean().to_numpy()
    x[p0:p1] = sur[p0:p1] + slope * np.arange(p1 - p0) + rng.normal(0, 0.3, p1 - p0)
    labels[p0:p1] = int(RootCause.DATA_POISONING)

    # coarse spikes
    spikes = rng.choice(np.arange(3200, n - 10), size=12, replace=False)
    x[spikes] += rng.choice([-1, 1], 12) * rng.uniform(9, 16, 12)
    labels[spikes] = int(RootCause.DATA_POISONING)

    return x, labels, {"drift": (d0, d1), "stealth": (p0, p1),
                       "spikes": spikes.tolist(), "slope": slope}


class SingleChannelNode:
    """The pipeline with the consistency stage removed (no redundancy available)."""

    def __init__(self, twin: SeasonalHarmonicTwin, classifier, cfg: FogConfig):
        self.twin, self.clf, self.cfg = twin, classifier, cfg
        self.res = ChannelResidualEngine(CHANNEL, cfg.residual,
                                         cfg.envelope.max_rate.get(CHANNEL, 0.02))
        self.trig = ChannelTrigger(cfg.trigger)
        self.gov = AdaptationGovernor([CHANNEL], cfg.freeze_hold,
                                      cfg.clean_to_resume)
        self.cal = CalibrationManager([CHANNEL], cfg)
        self.ctrl = DifferentialResponseController(cfg)

    def run(self, x: np.ndarray, probe=None) -> Tuple[np.ndarray, np.ndarray, dict]:
        n = len(x)
        causes = np.full(n, int(RootCause.NORMAL))
        forwarded = np.ones(n, dtype=bool)
        loc = Localisation()
        engage_until = -1
        invocations = 0

        for k in range(n):
            f = Frame(k=k, t=float(k), values={CHANNEL: self.cal.apply(CHANNEL, x[k])})
            pred = self.twin.predict(f)[CHANNEL]
            r = f.values[CHANNEL] - pred

            if k < self.cfg.residual.baseline_window:
                self.res.observe_warmup(r)
                self.twin.update(f, {CHANNEL: 1.0})
                continue

            z = self.res.push(r, pred)
            st = self.trig.update(z)
            self.gov.note_trigger(CHANNEL, st.fired)
            if st.fired:
                engage_until = k + self.cfg.engage_hold
            engaged = (k <= engage_until
                       or self.gov.state(CHANNEL) in (ChannelState.QUARANTINED,
                                                      ChannelState.RECOVERING))
            compensating = abs(self.cal.total(CHANNEL)) > 2 * self.res.base.sigma
            cause, conf = RootCause.NORMAL, 1.0
            if engaged:
                feats = self.res.features()
                if probe is not None:
                    probe(k, feats, loc)
                if self.clf is not None:
                    out = self.clf.classify(CHANNEL, feats, loc)
                    invocations += 1
                    cause, conf = out.cause, out.confidence
                    if (cause is RootCause.NORMAL and compensating
                            and self.gov.state(CHANNEL) is ChannelState.DRIFTING):
                        cause, conf = RootCause.SENSOR_DRIFT, 0.9
            elif self.gov.state(CHANNEL) is ChannelState.DRIFTING:
                cause, conf = RootCause.SENSOR_DRIFT, 0.9
            elif self.gov.state(CHANNEL) in (ChannelState.QUARANTINED,
                                             ChannelState.RECOVERING):
                cause, conf = RootCause.DATA_POISONING, 0.9

            plan = self.ctrl.plan(k, CHANNEL, cause, max(conf, 0.999))
            self.gov.note_decision(k, CHANNEL, cause if engaged else RootCause.NORMAL,
                                   conf, triggered=st.fired,
                                   compensating=compensating)
            if plan.allow_calibration and self.gov.trust(CHANNEL):
                self.cal.correct(k, CHANNEL, self.res.current_bias(), conf)
            self.res.refresh_baseline(r, self.gov.adaptation_allowed(CHANNEL))
            self.twin.update(f, {CHANNEL: 1.0 if self.gov.trust(CHANNEL) else 0.0})

            causes[k] = int(cause)
            forwarded[k] = plan.forward
        return causes, forwarded, {"invocations": invocations,
                                   "final_offset": self.cal.total(CHANNEL)}


def fit_twin(x: np.ndarray) -> SeasonalHarmonicTwin:
    """Identify the climatology on the commissioning window only.

    Fitting on the whole record would let the process model absorb the very
    injections it is supposed to expose.
    """
    t = np.arange(WARMUP, dtype=float)
    return SeasonalHarmonicTwin(channel=CHANNEL).fit(t, x[:WARMUP])


def train(x_list, label_list) -> MLRootCauseClassifier:
    X, y = [], []
    for x, labels in zip(x_list, label_list):
        twin = fit_twin(x)
        cfg = FogConfig()
        cfg.residual = ResidualConfig(window=90, baseline_window=WARMUP, smooth=7)
        cfg.trigger = TriggerConfig()
        node = SingleChannelNode(twin, None, cfg)

        def probe(k, feats, loc, _x=X, _y=y, _lab=labels):
            _x.append(build_vector(CHANNEL, feats, loc))
            _y.append(int(_lab[k]))

        node.run(x, probe=probe)
    return MLRootCauseClassifier().fit(np.vstack(X), np.asarray(y))


def score(labels, preds, name) -> Dict[str, float]:
    cm = confusion(labels[WARMUP:], preds[WARMUP:])
    rec, f1 = per_class_recall(cm), f1_from_cm(cm)
    return {"system": name,
            "root_cause_accuracy": float((labels[WARMUP:] == preds[WARMUP:]).mean()),
            "drift_recall": float(rec[1]), "poison_recall": float(rec[2]),
            "drift_f1": float(f1[1]), "poison_f1": float(f1[2]),
            "false_alarm_rate": float(
                (preds[WARMUP:][labels[WARMUP:] == 0] != 0).mean())}


def main() -> None:
    base = load()
    banner("EXPERIMENT 5 - the Generation-1 dataset (single channel, no redundancy)")
    print(f"record: {CSV.name}, {len(base)} readings\n")

    # train on one injection realisation, evaluate on disjoint ones
    xt, lt, _ = inject(base, seed=1)
    clf = train([xt], [lt])

    rows_p, rows_g = [], []
    for seed in (11, 12, 13):
        x, labels, meta = inject(base, seed=seed)
        twin = fit_twin(x)
        cfg = FogConfig()
        cfg.residual = ResidualConfig(window=90, baseline_window=WARMUP, smooth=7)
        node = SingleChannelNode(twin, clf, cfg)
        preds, fwd, info = node.run(x)
        rows_p.append(score(labels, preds, "PIRCAD (single channel)"))

        from pircad.simulator import Episode
        from pircad.config import PlantParams
        ep = Episode(t=np.arange(len(x), dtype=float), truth={"T_in": base},
                     obs={"T_in": x}, actuation={}, labels=labels,
                     target_channel="T_in", scenario="legacy",
                     params=PlantParams(), meta={"onset": 1000.0})
        g1 = Gen1FHAD().fit([ep])
        rows_g.append(score(labels, np.asarray(g1.run(ep)), "FHAD (Gen 1)"))

    def avg(rows):
        return {k: float(np.nanmean([r[k] for r in rows]))
                for k in rows[0] if k != "system"}

    a, b = avg(rows_p), avg(rows_g)
    keys = ["root_cause_accuracy", "drift_recall", "poison_recall",
            "drift_f1", "poison_f1", "false_alarm_rate"]
    print(f"{'metric':26s}{'PIRCAD 1-ch':>16s}{'FHAD (Gen 1)':>16s}")
    print("-" * 58)
    for k in keys:
        print(f"{k:26s}{a[k]:16.3f}{b[k]:16.3f}")

    print("\nNote: the Gen-1 comparison here is trained and tested on the same\n"
          "injection realisation, which flatters it.  Even so the separation of a\n"
          "rate-matched substituted stream from benign drift is what moves.")
    save_json("exp5_legacy_dataset.json", {"pircad": a, "gen1": b})


if __name__ == "__main__":
    main()
