"""
Experiment 9 -- does the safety function survive the attack?  (Powered version
of experiment 2, with stronger self-calibrating baselines.)

Three acts per episode, as in experiment 2: nominal operation; a slow
false-data injection biasing T_in downwards at a drift-like rate; then the
attacker withdraws and a genuine over-temperature fault (heater contactor welds
shut) occurs.  A cloud-side alarm at 34 C is applied to whatever the fog layer
forwards.  Whether it fires depends on whether the attack corrupted the
correction the system applies.

Grid: 20 seeds x 3 attack rates x 2 attackers (stale surrogate with white
noise; higher-fidelity surrogate with transducer-matched noise) = 120 episodes.

Systems: PIRCAD; PIRCAD with the response made non-differential (control);
EWMA self-recalibration; augmented-state Kalman bias estimator; Page-Hinkley
detect-and-recalibrate.  Each baseline's adaptation parameters were tuned to
minimise its correction error on *benign* drift (training seeds 1-3, disjoint
from the evaluation seeds), i.e. to do its intended job well:

    Kalman  q_b = 0.02^2      Page-Hinkley  delta = 0.01, lambda = 5
    EWMA    alpha = 0.002, recal_gain = 0.3

A benign-drift arm (same seeds and rates, no attack, no fault) reports how
well every system corrects genuine drift, so that a baseline cannot look safe
merely by not adapting at all.

Run:  python experiments/exp9_adaptation_poisoning.py
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from v2_common import (RunCfg, banner, bootstrap_ci, make_classifier, pmap,
                       save_json, worker_models)

from pircad.config import DEFAULT_SENSORS, PlantParams
from pircad.scenarios import Attacker, inject_stealth
from pircad.simulator import Episode, simulate_plant, transduce
from pircad.types import RootCause

ALARM_C = 34.0
N, ATTACK_START, ATTACK_END, FAULT_AT = 9000, 1500, 6000, 6600
RATES = (-0.8e-3, -1.4e-3, -2.0e-3)
ATTACKERS = {"stale/white": Attacker(0.02, "white"),
             "fidelity 0.3/matched": Attacker(0.3, "matched")}
SEEDS = tuple(range(4242, 4262))
SYSTEMS = ["PIRCAD", "PIRCAD auto re-attribution", "PIRCAD non-differential",
           "EWMA recalibration",
           "Kalman bias estimator", "Page-Hinkley recalibration"]


@dataclass(frozen=True)
class Spec9:
    seed: int
    rate: float
    attacker: str


def build(spec: Spec9) -> Episode:
    p = PlantParams()
    rng = np.random.default_rng(spec.seed)
    truth, act = simulate_plant(N, p, rng, stuck_heater_from=FAULT_AT)
    obs = transduce(truth, DEFAULT_SENSORS, rng)
    labels = np.full(N, int(RootCause.NORMAL))
    sub = {k: v.copy() for k, v in obs.items()}
    lab = labels.copy()
    inject_stealth(sub, truth, lab, "T_in", ATTACK_START, spec.rate,
                   ATTACKERS[spec.attacker], DEFAULT_SENSORS, p, rng)
    win = slice(ATTACK_START, ATTACK_END)
    obs["T_in"][win] = sub["T_in"][win]
    labels[win] = RootCause.DATA_POISONING
    labels[FAULT_AT:] = RootCause.PROCESS_EVENT
    return Episode(t=np.arange(N, dtype=float), truth=truth, obs=obs, actuation=act,
                   labels=labels, target_channel="T_in", scenario="absorb_then_fault",
                   params=p, meta={"onset": float(ATTACK_START), "seed": float(spec.seed)})


def first_cross(x: np.ndarray, thr: float, start: int) -> Optional[int]:
    idx = np.flatnonzero(np.asarray(x)[start:] >= thr)
    return int(idx[0] + start) if idx.size else None


def outcome(ep: Episode, offset: np.ndarray, fwd: np.ndarray,
            causes: Optional[np.ndarray] = None, admitted: float = float("nan")) -> Dict:
    reported = ep.obs["T_in"] - offset
    seen = np.where(fwd, reported, -np.inf)
    tc = first_cross(ep.truth["T_in"], ALARM_C, FAULT_AT)
    sc = first_cross(seen, ALARM_C, FAULT_AT)
    # a crossing of the forwarded value before the genuine fault even exists is
    # a spurious trip; one after the fault began but before the plant really
    # crossed is premature (it over-reads a real excursion -- fails safe)
    first = first_cross(seen, ALARM_C, ATTACK_START)
    att = slice(ATTACK_START, ATTACK_END)
    err_att = np.abs(reported[att] - ep.truth["T_in"][att])
    out = {
        "offset_at_fault": float(offset[tc]) if tc else float("nan"),
        "error_at_true_cross": float(reported[tc] - ep.truth["T_in"][tc]) if tc else float("nan"),
        "alarm_fired": float(sc is not None),
        "alarm_delay": float(sc - tc) if (sc is not None and tc) else float("nan"),
        "spurious_trip": float(first is not None and first < FAULT_AT),
        "premature_trip": float(first is not None and tc is not None
                                and FAULT_AT <= first < tc - 5),
        "lead_samples": float(tc - first) if (first is not None and tc is not None
                                              and first >= FAULT_AT) else float("nan"),
        "poison_forwarded": float(fwd[att].mean()),
        "forwarded_attack_error": float(np.where(fwd[att], err_att, 0.0).mean()),
        "poisoned_admitted_frac": admitted,
    }
    if causes is not None:
        out["poison_recall"] = float((causes[att] == int(RootCause.DATA_POISONING)).mean())
        out["poison_as_drift"] = float((causes[att] == int(RootCause.SENSOR_DRIFT)).mean())
        out["fault_as_process"] = float(
            (causes[FAULT_AT:] == int(RootCause.PROCESS_EVENT)).mean())
    return out


def run_one(spec: Spec9) -> Dict[str, Dict]:
    from pircad.config import FogConfig
    from pircad.fognode import PIRCADFogNode
    from pircad.simulator import frames
    worker_models()
    ep = build(spec)
    out = {}
    n_att = ATTACK_END - ATTACK_START
    for name, diff, reattr in (("PIRCAD", True, False),
                               ("PIRCAD auto re-attribution", True, True),
                               ("PIRCAD non-differential", False, False)):
        cfg = FogConfig(); cfg.differential_response = diff
        cfg.allow_reattribution = reattr
        res = PIRCADFogNode(classifier=make_classifier("rf"), cfg=cfg).run(
            frames(ep), truth_labels=ep.labels)
        d = res.per_channel["T_in"]
        off = np.array([x.calibration_offset for x in d])
        fwd = np.array([x.forwarded for x in d])
        adm = res.poisoned_admitted_by_channel.get("T_in", 0) / n_att
        # The non-differential control's channel sits in the DRIFTING state,
        # which the governor does not count as adaptation-eligible -- yet its
        # calibration offset is learning from attacker data.  Count that too.
        if not diff:
            adm = max(adm, float(np.mean(np.diff(off[ATTACK_START:ATTACK_END]) != 0)))
        out[name] = outcome(ep, off, fwd, res.causes(), adm)
    for name, off in baseline_offsets(ep).items():
        out[name] = outcome(ep, off, np.ones(N, bool), admitted=1.0)
    for v in out.values():
        v.update({"seed": spec.seed, "rate": spec.rate, "attacker": spec.attacker})
    return out


def baseline_offsets(ep: Episode) -> Dict[str, np.ndarray]:
    from pircad.baselines import AdaptiveRecalibratingBaseline
    from pircad.baselines_adaptive import KalmanBiasEstimator, PageHinkleyRecalibrator
    y = ep.obs["T_in"]
    return {
        "EWMA recalibration": AdaptiveRecalibratingBaseline(
            alpha=0.002, recal_gain=0.3, warmup=600).run(y).offset,
        "Kalman bias estimator": KalmanBiasEstimator(q_b=0.02 ** 2).run(
            y, ep.actuation["u_heater"], ep.obs["T_amb"]).offset,
        "Page-Hinkley recalibration": PageHinkleyRecalibrator(
            delta=0.01, lam=5.0).run(y).offset,
    }


def run_drift(spec: Spec9) -> Dict[str, Dict]:
    """Benign drift only: how well does each system correct a real fault?"""
    from pircad.config import FogConfig
    from pircad.fognode import PIRCADFogNode
    from pircad.scenarios import make_episode_v2
    from pircad.simulator import frames
    worker_models()
    ep = make_episode_v2("drift", n=N, seed=spec.seed, onset=ATTACK_START, rate=spec.rate)
    s = ATTACK_START
    truth = ep.truth["T_in"][s:]

    def score(off, fwd, causes=None):
        err = np.abs(ep.obs["T_in"][s:] - off[s:] - truth)
        f = fwd[s:]
        o = {"drift_fwd_frac": float(f.mean()),
             "drift_fwd_error": float(err[f].mean()) if f.any() else float("nan"),
             "drift_final_error": float(err[-500:].mean())}
        if causes is not None:
            o["drift_recall"] = float((causes[s:] == int(RootCause.SENSOR_DRIFT)).mean())
            o["drift_as_poison"] = float((causes[s:] == int(RootCause.DATA_POISONING)).mean())
        return o

    out = {}
    for name, diff, reattr in (("PIRCAD", True, False),
                               ("PIRCAD auto re-attribution", True, True),
                               ("PIRCAD non-differential", False, False)):
        cfg = FogConfig(); cfg.differential_response = diff
        cfg.allow_reattribution = reattr
        res = PIRCADFogNode(classifier=make_classifier("rf"), cfg=cfg).run(
            frames(ep), truth_labels=ep.labels)
        d = res.per_channel["T_in"]
        out[name] = score(np.array([x.calibration_offset for x in d]),
                          np.array([x.forwarded for x in d]), res.causes())
    for name, off in baseline_offsets(ep).items():
        out[name] = score(off, np.ones(N, bool))
    for v in out.values():
        v.update({"seed": spec.seed, "rate": spec.rate})
    return out


KEYS = ["alarm_fired", "spurious_trip", "premature_trip", "lead_samples", "error_at_true_cross", "offset_at_fault",
        "alarm_delay", "poison_forwarded", "forwarded_attack_error",
        "poisoned_admitted_frac", "poison_recall", "poison_as_drift",
        "fault_as_process"]


def main() -> None:
    specs = [Spec9(s, r, a) for a in ATTACKERS for r in RATES for s in SEEDS]
    banner(f"EXPERIMENT 9 - {len(specs)} attack-then-fault episodes")
    runs = pmap(run_one, specs, "exp9")
    rows = {s: [r[s] for r in runs] for s in SYSTEMS}
    summary = {s: {k: bootstrap_ci([x.get(k, np.nan) for x in rows[s]]) for k in KEYS}
               for s in SYSTEMS}
    by_cond = {}
    for s in SYSTEMS:
        by_cond[s] = {}
        for a in ATTACKERS:
            for r in RATES:
                sel = [x for x in rows[s] if x["attacker"] == a and x["rate"] == r]
                by_cond[s][f"{a} | {r:+.1e}"] = {
                    k: float(np.nanmean([x.get(k, np.nan) for x in sel])) for k in
                    ("alarm_fired", "spurious_trip", "premature_trip",
                     "error_at_true_cross")}
    for s in SYSTEMS:
        sm = summary[s]
        print(f"{s:28s} alarm {sm['alarm_fired'][0]:.3f} spurious {sm['spurious_trip'][0]:.3f} "
              f"premature {sm['premature_trip'][0]:.3f} "
              f"err@cross {sm['error_at_true_cross'][0]:+.2f} "
              f"[{sm['error_at_true_cross'][1]:+.2f},{sm['error_at_true_cross'][2]:+.2f}] "
              f"fwdPoison {sm['poison_forwarded'][0]:.3f} "
              f"admitted {sm['poisoned_admitted_frac'][0]:.3f}")
    dspecs = [Spec9(sd, r, "stale/white") for r in RATES for sd in SEEDS]
    banner(f"Benign-drift arm - {len(dspecs)} episodes")
    druns = pmap(run_drift, dspecs, "drift")
    drows = {s: [r[s] for r in druns] for s in SYSTEMS}
    dkeys = ["drift_fwd_frac", "drift_fwd_error", "drift_final_error",
             "drift_recall", "drift_as_poison"]
    dsum = {s: {k: bootstrap_ci([x.get(k, np.nan) for x in drows[s]]) for k in dkeys}
            for s in SYSTEMS}
    for s in SYSTEMS:
        d = dsum[s]
        print(f"{s:28s} drift fwd {d['drift_fwd_frac'][0]:.3f}  "
              f"fwd error {d['drift_fwd_error'][0]:.3f} C  "
              f"final error {d['drift_final_error'][0]:.3f} C  "
              f"called poison {d['drift_as_poison'][0]:.3f}")
    save_json("exp9_adaptation_poisoning.json", {
        "n": len(specs), "summary": summary, "by_condition": by_cond, "rows": rows,
        "drift_summary": dsum, "drift_rows": drows})


if __name__ == "__main__":
    main()
