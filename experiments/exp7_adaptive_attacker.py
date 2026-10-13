"""
Experiment 7 -- how much does the attacker have to know?

The headline benchmark uses one attacker: a stale surrogate of the process plus
white noise.  A reviewer will rightly ask what happens against a stronger one.
This experiment sweeps attacker capability, holding the injected ramp
rate-matched to benign drift throughout, and evaluates PIRCAD and the
data-driven baselines *without retraining* (all were trained on the default
attacker only):

  substitute   surrogate fidelity 0.02 (default) -> 0.05 -> 0.1 -> 0.3 -> 1.0
               (1.0 = the attacker knows the exact live process state),
               crossed with white vs transducer-matched (AR-1) noise
  mitm         an on-path attacker adds the ramp to the genuine reading.
               On the attacked channel this is observationally identical to
               benign drift; no detector can separate the two from that channel
  coordinated  one fake temperature drives T_in, P and RH together, so the
               instrument-side physical constraints stay satisfied

Two kinds of outcome are reported.  Label-level: poisoning recall and the
false-drift rate.  Effect-level: `harm`, the mean absolute error of what is
actually forwarded to the cloud on the attacked channel, against the physical
truth (withheld samples contribute zero).  The second matters because a
misclassified attack whose bias is removed by a physics-anchored correction is
not the same failure as one that is absorbed.

Run:  python experiments/exp7_adaptive_attacker.py [--seeds N]
"""

from __future__ import annotations

import argparse
from typing import Dict, List

import numpy as np

from v2_common import (EVAL_SEEDS, RunCfg, banner, bootstrap_ci, eval_specs,
                       get_ml_baselines, get_v2_models, pmap, run_pircad,
                       save_json, score_baseline, strip_private)

from pircad.scenarios import Attacker

CONDITIONS = [
    ("drift (reference)", None),
    ("stale, white (default)", Attacker(0.02, "white")),
    ("stale, matched noise", Attacker(0.02, "matched")),
    ("fidelity 0.05", Attacker(0.05, "matched")),
    ("fidelity 0.10", Attacker(0.10, "matched")),
    ("fidelity 0.30", Attacker(0.30, "matched")),
    ("exact state, white", Attacker(1.0, "white")),
    ("exact state, matched", Attacker(1.0, "matched")),
    ("on-path (MITM) ramp", Attacker(1.0, "matched", "mitm")),
    ("coordinated, stale", Attacker(0.02, "matched", "coordinated")),
    ("coordinated, exact", Attacker(1.0, "matched", "coordinated")),
]
KEYS = ["poison_recall", "false_drift_rate", "drift_recall", "process_recall",
        "root_cause_accuracy", "poison_containment", "harm", "useful_frac", "fwd_err_mean",
        "final_offset"]


def specs_for(att, seeds):
    if att is None:
        return eval_specs(["drift"], seeds=seeds)
    return eval_specs(["stealth_poison"], seeds=seeds, attacker=att)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    a = ap.parse_args()
    seeds = EVAL_SEEDS[:a.seeds]
    get_v2_models()
    bl = get_ml_baselines()

    banner(f"EXPERIMENT 7 - attacker capability sweep, {len(seeds)} seeds per condition")
    out: Dict[str, Dict] = {}
    for label, att in CONDITIONS:
        specs = specs_for(att, seeds)
        rows = {"PIRCAD": pmap(run_pircad, [(s, RunCfg()) for s in specs], label)}
        for name, model in bl.items():
            rr = []
            for s in specs:
                ep = s.build()
                rr.append(score_baseline(s, ep, model.predict(ep)))
            rows[name] = rr
        out[label] = {}
        line = f"{label:26s}"
        for sysname, rr in rows.items():
            summ = {k: bootstrap_ci([r.get(k, np.nan) for r in rr]) for k in KEYS}
            out[label][sysname] = {"summary": summ, "rows": strip_private(rr)}
            rec = summ["drift_recall" if att is None else "poison_recall"][0]
            fd = summ["false_drift_rate"][0]
            line += f" | {sysname[:10]:10s} rec {rec:.3f} fD {fd:.3f} harm {summ['harm'][0]:.3f}"
        print(line, flush=True)

    save_json("exp7_adaptive_attacker.json",
              {"seeds": list(seeds), "conditions": [c[0] for c in CONDITIONS],
               "results": out})


if __name__ == "__main__":
    main()
