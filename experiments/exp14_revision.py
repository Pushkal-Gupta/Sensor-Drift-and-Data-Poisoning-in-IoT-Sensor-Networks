"""
Experiment 14 -- analyses requested in the first-round review (Array).

Four parts, each saved to its own artifact so they can be run independently:

  threshold   Sensitivity to the confidence threshold (default 0.62) on the
              180-episode simulated bank and on the 312 Intel Lab test episodes.
  safety      The attack-then-fault experiment (exp. 9 design) repeated for the
              ablated variants -- shared estimator instead of leave-one-out
              observers, no cross-sensor localisation, no noise-signature
              features, gradient-boosting back-end, other thresholds -- and with
              an on-path (A4) and a coordinated (A5) attacker added.  This asks
              whether the elements that do not change in-distribution macro-F1
              change what reaches the safety function.
  dose        Dose-response: the self-calibrating baselines swept over their
              adaptation parameter (alarm rate vs. benign-drift correction
              error), attack duration, and the alarm threshold.
  reference   Error of the leave-one-out expectation on clean data, simulated
              vs. real (Intel Lab): the eps_ref term of Proposition 2.

Run:  python experiments/exp14_revision.py {threshold|safety|dose|reference|all}
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from v2_common import (WARMUP, banner, bootstrap_ci, effect_metrics, eval_specs,
                       macro_f1_episode, make_classifier, pmap, save_json,
                       worker_models)

from pircad.config import DEFAULT_SENSORS, PlantParams
from pircad.scenarios import Attacker, inject_stealth
from pircad.simulator import Episode, simulate_plant, transduce
from pircad.types import RootCause

THRESHOLDS = (0.40, 0.50, 0.62, 0.75, 0.90)

# ---------------------------------------------------------------------------
# shared: build a node for a named variant
# ---------------------------------------------------------------------------
VARIANTS = {
    # name: (exp11 row or None, backend, threshold)
    "PIRCAD": (None, "rf", 0.62),
    "PIRCAD (gradient boosting)": (None, "hgb", 0.62),
    "PIRCAD threshold 0.50": (None, "rf", 0.50),
    "PIRCAD threshold 0.80": (None, "rf", 0.80),
    "- noise-signature features": ("- noise-signature features", "rf", 0.62),
    "- dedicated observers (shared estimator)": ("- dedicated observers", "rf", 0.62),
    "- cross-sensor localisation": ("- cross-sensor consistency", "rf", 0.62),
}


def node_for(variant: str, threshold: Optional[float] = None, cfg=None):
    from pircad.config import FogConfig
    from pircad.fognode import PIRCADFogNode
    row, backend, thr = VARIANTS[variant]
    cfg = cfg or FogConfig()
    t = thr if threshold is None else threshold
    cfg.confidence_accept = t
    cfg.confidence_escalate = t
    if row is None:
        worker_models()
        return PIRCADFogNode(classifier=make_classifier(backend), cfg=cfg)
    import exp11_ablation as a11
    return a11.make_node(a11.STRUCTURAL.get(row, "full"), a11._classifier(row), cfg)


_ICLF = {}


def intel_clf():
    if not _ICLF:
        import joblib
        from pircad.rootcause import HybridRootCauseClassifier
        from v2_common import ARTIFACTS, accelerate
        _ICLF["rf"] = HybridRootCauseClassifier(accelerate(
            joblib.load(ARTIFACTS / "exp10_models.joblib")["rf"]))
    return _ICLF["rf"]


# ---------------------------------------------------------------------------
# Part 1: confidence threshold
# ---------------------------------------------------------------------------
def _thr_sim(args):
    spec, thr = args
    from pircad.metrics import evaluate_run
    from pircad.simulator import frames
    ep = spec.build()
    res = node_for("PIRCAD", thr).run(frames(ep), truth_labels=ep.labels)
    m = evaluate_run(ep, res, WARMUP)
    m.update(effect_metrics(ep, res))
    m["macro_f1"] = macro_f1_episode(ep.labels[WARMUP:], res.causes()[WARMUP:])
    m["scenario"] = spec.scenario; m["seed"] = spec.seed; m["threshold"] = thr
    if ep.target_channel is not None:
        d = res.per_channel[ep.target_channel][WARMUP:]
        m["held_frac"] = float(np.mean([not x.forwarded for x in d]))
    return m


def _thr_intel(args):
    spec, thr = args
    import exp10_intel_lab as e10
    from pircad.indoor import WARMUP_INDOOR, indoor_frames
    from pircad.metrics import evaluate_run
    from v2_common import effect_metrics as em
    clf = intel_clf()
    from pircad.fognode import PIRCADFogNode
    from pircad.indoor import INDOOR_CHANNELS, IndoorConsistencyMonitor, IndoorTwin, indoor_config
    cfg = indoor_config(); cfg.confidence_accept = thr; cfg.confidence_escalate = thr
    tw = IndoorTwin()
    ep = e10.build(spec)
    res = PIRCADFogNode(twin=tw, classifier=clf, cfg=cfg, channels=INDOOR_CHANNELS,
                        consistency=IndoorConsistencyMonitor(tw)).run(
        indoor_frames(ep), truth_labels=ep.labels)
    m = evaluate_run(ep, res, WARMUP_INDOOR)
    m.update(em(ep, res))
    m["macro_f1"] = macro_f1_episode(ep.labels[WARMUP_INDOOR:], res.causes()[WARMUP_INDOOR:])
    m["scenario"] = spec.scenario; m["seed"] = spec.seed; m["threshold"] = thr
    return m


TKEYS = ["macro_f1", "false_drift_rate", "false_poison_rate", "false_alarm_rate",
         "drift_recall", "poison_recall", "harm", "useful_frac", "held_frac"]


def part_threshold() -> None:
    banner("EXP 14a - confidence-threshold sensitivity")
    out = {"thresholds": list(THRESHOLDS), "sim": {}, "intel": {}}
    specs = eval_specs()
    for thr in THRESHOLDS:
        rows = pmap(_thr_sim, [(s, thr) for s in specs], f"sim thr {thr}")
        out["sim"][f"{thr:.2f}"] = _summ(rows, TKEYS)
        print(f"  sim thr {thr:.2f}  " + _line(out["sim"][f"{thr:.2f}"]), flush=True)
    import exp10_intel_lab as e10
    import json
    from v2_common import ARTIFACTS
    test_m = json.load(open(ARTIFACTS / "exp10_intel_lab.json"))["test_motes"]
    ispecs = e10.specs_for(test_m, 2)
    for thr in THRESHOLDS:
        rows = pmap(_thr_intel, [(s, thr) for s in ispecs], f"intel thr {thr}")
        out["intel"][f"{thr:.2f}"] = _summ(rows, TKEYS)
        print(f"  intel thr {thr:.2f}  " + _line(out["intel"][f"{thr:.2f}"]), flush=True)
    save_json("exp14_threshold.json", out)


def _summ(rows, keys):
    return {"summary": {k: bootstrap_ci([r.get(k, np.nan) for r in rows]) for k in keys},
            "by_scenario": {sc: {k: float(np.nanmean([r.get(k, np.nan) for r in rows
                                                       if r["scenario"] == sc]))
                                 for k in keys}
                            for sc in sorted({r["scenario"] for r in rows})},
            "rows": [{k: r.get(k) for k in keys + ["scenario", "seed"]} for r in rows]}


def _line(s):
    s = s["summary"]
    return " ".join(f"{k[:10]} {s[k][0]:.3f}" for k in
                    ("macro_f1", "false_drift_rate", "false_poison_rate", "false_alarm_rate",
                     "drift_recall"))


# ---------------------------------------------------------------------------
# Part 2: safety experiment with the ablated variants and A4 / A5
# ---------------------------------------------------------------------------
ALARMS = (32.0, 33.0, 34.0, 35.0, 36.0)
RATES = (-0.8e-3, -1.4e-3, -2.0e-3)
SEEDS = tuple(range(4242, 4262))
ATTACKERS = {"stale/white": Attacker(0.02, "white"),
             "fidelity 0.3/matched": Attacker(0.3, "matched"),
             "A4 on-path": Attacker(1.0, "matched", "mitm"),
             "A5 coordinated": Attacker(0.02, "matched", "coordinated")}


@dataclass(frozen=True)
class SafetySpec:
    seed: int
    rate: float
    attacker: str
    duration: int = 4500
    start: int = 1500

    @property
    def end(self):
        return self.start + self.duration

    @property
    def fault(self):
        return self.end + 600

    @property
    def n(self):
        return self.fault + 2400


def build_safety(spec: SafetySpec) -> Episode:
    p = PlantParams()
    rng = np.random.default_rng(spec.seed)
    truth, act = simulate_plant(spec.n, p, rng, stuck_heater_from=spec.fault)
    obs = transduce(truth, DEFAULT_SENSORS, rng)
    labels = np.full(spec.n, int(RootCause.NORMAL))
    sub = {k: v.copy() for k, v in obs.items()}
    lab = labels.copy()
    inject_stealth(sub, truth, lab, "T_in", spec.start, spec.rate,
                   ATTACKERS[spec.attacker], DEFAULT_SENSORS, p, rng)
    win = slice(spec.start, spec.end)
    for ch in (("T_in", "P", "RH") if spec.attacker == "A5 coordinated" else ("T_in",)):
        obs[ch][win] = sub[ch][win]
    labels[win] = RootCause.DATA_POISONING
    labels[spec.fault:] = RootCause.PROCESS_EVENT
    return Episode(t=np.arange(spec.n, dtype=float), truth=truth, obs=obs, actuation=act,
                   labels=labels, target_channel="T_in", scenario="absorb_then_fault",
                   params=p, meta={"onset": float(spec.start), "seed": float(spec.seed)})


def _first(x, thr, start):
    idx = np.flatnonzero(np.asarray(x)[start:] >= thr)
    return int(idx[0] + start) if idx.size else None


def safety_outcome(spec: SafetySpec, ep: Episode, offset, fwd, causes=None) -> Dict:
    reported = ep.obs["T_in"] - offset
    seen = np.where(fwd, reported, -np.inf)
    att = slice(spec.start, spec.end)
    out = {"poison_forwarded": float(fwd[att].mean()),
           "forwarded_attack_error": float(np.where(
               fwd[att], np.abs(reported[att] - ep.truth["T_in"][att]), 0.0).mean())}
    for a in ALARMS:
        tc = _first(ep.truth["T_in"], a, spec.fault)
        sc = _first(seen, a, spec.fault)
        first = _first(seen, a, spec.start)
        tag = f"{a:.0f}"
        out[f"alarm_{tag}"] = float(sc is not None) if tc is not None else float("nan")
        out[f"spurious_{tag}"] = float(first is not None and first < spec.fault)
        out[f"err_{tag}"] = float(reported[tc] - ep.truth["T_in"][tc]) if tc else float("nan")
    if causes is not None:
        out["poison_recall"] = float((causes[att] == int(RootCause.DATA_POISONING)).mean())
        out["poison_as_drift"] = float((causes[att] == int(RootCause.SENSOR_DRIFT)).mean())
    out.update({"seed": spec.seed, "rate": spec.rate, "attacker": spec.attacker,
                "duration": spec.duration})
    return out


def _run_variant(args):
    spec, variant = args
    from pircad.simulator import frames
    ep = build_safety(spec)
    res = node_for(variant).run(frames(ep), truth_labels=ep.labels)
    d = res.per_channel["T_in"]
    off = np.array([x.calibration_offset for x in d])
    fwd = np.array([x.forwarded for x in d])
    o = safety_outcome(spec, ep, off, fwd, res.causes())
    o["system"] = variant
    return o


def _drift_variant(args):
    spec, variant = args
    from pircad.scenarios import make_episode_v2
    from pircad.simulator import frames
    ep = make_episode_v2("drift", n=9000, seed=spec.seed, onset=1500, rate=spec.rate)
    res = node_for(variant).run(frames(ep), truth_labels=ep.labels)
    d = res.per_channel["T_in"]
    off = np.array([x.calibration_offset for x in d])
    fwd = np.array([x.forwarded for x in d])
    err = np.abs(ep.obs["T_in"][1500:] - off[1500:] - ep.truth["T_in"][1500:])
    f = fwd[1500:]
    return {"system": variant, "seed": spec.seed, "rate": spec.rate,
            "drift_fwd_frac": float(f.mean()),
            "drift_fwd_error": float(err[f].mean()) if f.any() else float("nan"),
            "drift_useful": float((f & (err < 0.5)).mean()),
            "drift_as_poison": float((res.causes()[1500:] == int(RootCause.DATA_POISONING)).mean())}


def _agg(rows, keys):
    return {k: bootstrap_ci([r.get(k, np.nan) for r in rows]) for k in keys}


def part_safety() -> None:
    banner("EXP 14b - safety function under the ablated variants, A0/A2/A4/A5")
    specs = [SafetySpec(s, r, a) for a in ATTACKERS for r in RATES for s in SEEDS]
    dspecs = [SafetySpec(s, r, "stale/white") for r in RATES for s in SEEDS]
    keys = ([f"alarm_{a:.0f}" for a in ALARMS] + [f"spurious_{a:.0f}" for a in ALARMS]
            + [f"err_{a:.0f}" for a in ALARMS]
            + ["poison_forwarded", "forwarded_attack_error", "poison_recall", "poison_as_drift"])
    dkeys = ["drift_fwd_frac", "drift_fwd_error", "drift_useful", "drift_as_poison"]
    out = {"systems": list(VARIANTS), "attackers": list(ATTACKERS), "by_attacker": {},
           "drift": {}, "rows": {}}
    for v in VARIANTS:
        rows = pmap(_run_variant, [(s, v) for s in specs], v)
        out["rows"][v] = rows
        out["by_attacker"][v] = {a: _agg([r for r in rows if r["attacker"] == a], keys)
                                 for a in ATTACKERS}
        drows = pmap(_drift_variant, [(s, v) for s in dspecs], v + " drift")
        out["drift"][v] = _agg(drows, dkeys)
        ba = out["by_attacker"][v]
        print(f"{v:42s} " + " ".join(
            f"{a[:6]} alarm {ba[a]['alarm_34'][0]:.2f} err {ba[a]['err_34'][0]:+.2f} "
            f"fwd {ba[a]['poison_forwarded'][0]:.2f} |" for a in ATTACKERS)
              + f" drift err {out['drift'][v]['drift_fwd_error'][0]:.2f}", flush=True)
    save_json("exp14_safety.json", out)


# ---------------------------------------------------------------------------
# Part 3: dose-response -- baseline tuning, attack duration, alarm threshold
# ---------------------------------------------------------------------------
GRID = {
    "Kalman bias estimator": [("q_b", v) for v in (0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05)],
    "EWMA recalibration": [("alpha", v) for v in (0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01)],
    "Page-Hinkley recalibration": [("lam", v) for v in (2.5, 5.0, 10.0, 20.0, 40.0, 80.0)],
}
DEFAULTS = {"Kalman bias estimator": 0.02, "EWMA recalibration": 0.002,
            "Page-Hinkley recalibration": 5.0}


def baseline_offset(name, value, ep):
    from pircad.baselines import AdaptiveRecalibratingBaseline
    from pircad.baselines_adaptive import KalmanBiasEstimator, PageHinkleyRecalibrator
    y = ep.obs["T_in"]
    if name == "Kalman bias estimator":
        return KalmanBiasEstimator(q_b=value ** 2).run(
            y, ep.actuation["u_heater"], ep.obs["T_amb"]).offset
    if name == "EWMA recalibration":
        return AdaptiveRecalibratingBaseline(alpha=value, recal_gain=0.3, warmup=600).run(y).offset
    return PageHinkleyRecalibrator(delta=0.01, lam=value).run(y).offset


def _dose_baselines(spec: SafetySpec):
    ep = build_safety(spec)
    out = []
    for name, grid in GRID.items():
        for _, v in grid:
            off = baseline_offset(name, v, ep)
            o = safety_outcome(spec, ep, off, np.ones(len(off), bool))
            o.update({"system": name, "param": v})
            out.append(o)
    return out


def _dose_baselines_drift(spec: SafetySpec):
    from pircad.scenarios import make_episode_v2
    ep = make_episode_v2("drift", n=9000, seed=spec.seed, onset=1500, rate=spec.rate)
    out = []
    for name, grid in GRID.items():
        for _, v in grid:
            off = baseline_offset(name, v, ep)
            err = np.abs(ep.obs["T_in"][1500:] - off[1500:] - ep.truth["T_in"][1500:])
            out.append({"system": name, "param": v, "drift_fwd_error": float(err.mean()),
                        "drift_useful": float((err < 0.5).mean())})
    return out


def _dose_duration(args):
    spec, system = args
    if system == "PIRCAD":
        return _run_variant((spec, "PIRCAD"))
    ep = build_safety(spec)
    off = baseline_offset(system, DEFAULTS[system], ep)
    o = safety_outcome(spec, ep, off, np.ones(len(off), bool))
    o["system"] = system
    return o


def part_dose() -> None:
    banner("EXP 14c - dose-response")
    atk = ["stale/white", "fidelity 0.3/matched"]
    specs = [SafetySpec(s, r, a) for a in atk for r in RATES for s in SEEDS]
    dspecs = [SafetySpec(s, r, "stale/white") for r in RATES for s in SEEDS]
    rows = [x for part in pmap(_dose_baselines, specs, "baseline grid") for x in part]
    drows = [x for part in pmap(_dose_baselines_drift, dspecs, "baseline grid drift")
             for x in part]
    tuning = {}
    for name, grid in GRID.items():
        tuning[name] = []
        for _, v in grid:
            a = [r for r in rows if r["system"] == name and r["param"] == v]
            d = [r for r in drows if r["system"] == name and r["param"] == v]
            tuning[name].append({"param": v, "default": v == DEFAULTS[name],
                                 "alarm_34": bootstrap_ci([r["alarm_34"] for r in a]),
                                 "spurious_34": bootstrap_ci([r["spurious_34"] for r in a]),
                                 "err_34": bootstrap_ci([r["err_34"] for r in a]),
                                 "drift_fwd_error": bootstrap_ci([r["drift_fwd_error"] for r in d]),
                                 "drift_useful": bootstrap_ci([r["drift_useful"] for r in d])})
            t = tuning[name][-1]
            print(f"  {name:28s} {v:<8g} alarm {t['alarm_34'][0]:.2f} spurious "
                  f"{t['spurious_34'][0]:.2f} err {t['err_34'][0]:+.2f} drift err "
                  f"{t['drift_fwd_error'][0]:.2f}", flush=True)
    systems = ["PIRCAD", "EWMA recalibration", "Kalman bias estimator",
               "Page-Hinkley recalibration"]
    duration = {}
    for dur in (1500, 3000):
        dsp = [SafetySpec(s, r, a, duration=dur) for a in atk for r in RATES for s in SEEDS]
        res = pmap(_dose_duration, [(s, sy) for s in dsp for sy in systems], f"dur {dur}")
        duration[str(dur)] = {sy: _agg([r for r in res if r["system"] == sy],
                                       ["alarm_34", "spurious_34", "err_34", "poison_forwarded"])
                              for sy in systems}
        for sy in systems:
            dd = duration[str(dur)][sy]
            print(f"  dur {dur} {sy:28s} alarm {dd['alarm_34'][0]:.2f} err {dd['err_34'][0]:+.2f}",
                  flush=True)
    save_json("exp14_dose.json", {"tuning": tuning, "duration": duration})


# ---------------------------------------------------------------------------
# Part 4: reference error on clean data
# ---------------------------------------------------------------------------
def _ref_sim(spec):
    from pircad.simulator import frames
    ep = spec.build()
    res = node_for("PIRCAD").run(frames(ep), truth_labels=ep.labels)
    r = np.array([x.residual for x in res.per_channel["T_in"]])[WARMUP:]
    noise = ep.obs["T_in"][WARMUP:] - ep.truth["T_in"][WARMUP:]
    return {"abs_resid": np.abs(r).tolist()[::5], "abs_eref": np.abs(r - noise).tolist()[::5]}


def _ref_intel(spec):
    import exp10_intel_lab as e10
    from pircad.indoor import WARMUP_INDOOR, indoor_frames
    ep = e10.build(spec)
    res = e10.node_for(intel_clf()).run(indoor_frames(ep), truth_labels=ep.labels)
    r = np.array([x.residual for x in res.per_channel["T"]])[WARMUP_INDOOR:]
    return {"abs_resid": np.abs(r).tolist()[::5], "mote": spec.mote}


def part_reference() -> None:
    banner("EXP 14d - leave-one-out reference error on clean data")
    import json
    import exp10_intel_lab as e10
    from v2_common import ARTIFACTS
    sim = pmap(_ref_sim, eval_specs(["clean"]), "sim clean")
    test_m = json.load(open(ARTIFACTS / "exp10_intel_lab.json"))["test_motes"]
    ispecs = [s for s in e10.specs_for(test_m, 2) if s.scenario == "clean"]
    real = pmap(_ref_intel, ispecs, "intel clean")

    def q(vals):
        v = np.concatenate([np.asarray(x) for x in vals])
        return {"mean": float(v.mean()), "median": float(np.median(v)),
                "p95": float(np.quantile(v, 0.95)), "p99": float(np.quantile(v, 0.99))}

    per_mote = [float(np.median(x["abs_resid"])) for x in real]
    out = {"sim_abs_residual": q([x["abs_resid"] for x in sim]),
           "sim_abs_eref": q([x["abs_eref"] for x in sim]),
           "intel_abs_residual": q([x["abs_resid"] for x in real]),
           "intel_per_mote_median": {"min": min(per_mote), "max": max(per_mote),
                                     "median": float(np.median(per_mote))}}
    print(json.dumps(out, indent=1))
    save_json("exp14_reference.json", out)


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    for name, fn in (("reference", part_reference), ("threshold", part_threshold),
                     ("dose", part_dose), ("safety", part_safety)):
        if what in (name, "all"):
            fn()


# ---------------------------------------------------------------------------
# Part 5: the self-calibrating baselines under every attacker of part 2
# ---------------------------------------------------------------------------
def _baselines_all(spec: SafetySpec):
    ep = build_safety(spec)
    out = []
    for name, v in DEFAULTS.items():
        o = safety_outcome(spec, ep, baseline_offset(name, v, ep), np.ones(spec.n, bool))
        o["system"] = name
        out.append(o)
    return out


def part_baselines() -> None:
    banner("EXP 14e - self-calibrating baselines under A0/A2/A4/A5")
    specs = [SafetySpec(s, r, a) for a in ATTACKERS for r in RATES for s in SEEDS]
    rows = [x for p in pmap(_baselines_all, specs, "baselines") for x in p]
    keys = ([f"alarm_{a:.0f}" for a in ALARMS] + [f"spurious_{a:.0f}" for a in ALARMS]
            + [f"err_{a:.0f}" for a in ALARMS] + ["poison_forwarded", "forwarded_attack_error"])
    out = {name: {a: _agg([r for r in rows if r["system"] == name and r["attacker"] == a], keys)
                  for a in ATTACKERS} for name in DEFAULTS}
    for name in DEFAULTS:
        print(f"{name:28s} " + " ".join(
            f"{a[:6]} alarm {out[name][a]['alarm_34'][0]:.2f} spur {out[name][a]['spurious_34'][0]:.2f} "
            f"err {out[name][a]['err_34'][0]:+.2f} |" for a in ATTACKERS), flush=True)
    save_json("exp14_baselines.json", out)


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "baselines":
    part_baselines()
