"""
Experiment 8 -- robustness to distribution shift.

Every system is trained once, on the nominal plant, and then evaluated without
retraining on plants and sensors that differ from it.  The hypothesis is that a
model whose inputs are physics residuals transfers better than one that learned
the raw signal's appearance, because the residual is invariant to most of what
changes.  Conditions (all six scenarios each):

  nominal                 reference
  twin mismatch x3, x6    the fog node's process model is 3x / 6x less accurate
                          (PIRCAD only -- the baselines have no model to degrade)
  plant dynamics -15/+20% the true thermal capacitance and loss change
  noise x2, x3            every transducer's noise sigma is scaled
  ambient swing x2.5      a larger daily ambient excursion
  setpoint 35 C           a different operating point
  slow rates              drift/attack rate 0.3-0.8e-3, below the training range
  fast rates              2.5-2.9e-3, above the training range (inside envelope)

Run:  python experiments/exp8_distribution_shift.py [--seeds N]
"""

from __future__ import annotations

import argparse
from typing import Dict

import numpy as np

from v2_common import (EVAL_SEEDS, RunCfg, banner, bootstrap_ci, eval_specs,
                       get_ml_baselines, get_v2_models, pmap, run_pircad,
                       save_json, score_baseline, strip_private)

from pircad.scenarios import Shift

CONDITIONS = [
    ("nominal", Shift(), 1.0, (0.8e-3, 2.5e-3)),
    ("twin mismatch x3", Shift(), 3.0, (0.8e-3, 2.5e-3)),
    ("twin mismatch x6", Shift(), 6.0, (0.8e-3, 2.5e-3)),
    ("plant dynamics -15%", Shift(plant_scale=0.85), 1.0, (0.8e-3, 2.5e-3)),
    ("plant dynamics +20%", Shift(plant_scale=1.20), 1.0, (0.8e-3, 2.5e-3)),
    ("noise x2", Shift(noise_scale=2.0), 1.0, (0.8e-3, 2.5e-3)),
    ("noise x3", Shift(noise_scale=3.0), 1.0, (0.8e-3, 2.5e-3)),
    ("ambient swing x2.5", Shift(amb_amp_scale=2.5), 1.0, (0.8e-3, 2.5e-3)),
    ("setpoint 35 C", Shift(setpoint=35.0), 1.0, (0.8e-3, 2.5e-3)),
    ("slow rates", Shift(), 1.0, (0.3e-3, 0.8e-3)),
    ("fast rates", Shift(), 1.0, (2.5e-3, 2.9e-3)),
]
KEYS = ["macro_f1", "root_cause_accuracy", "drift_recall", "poison_recall",
        "process_recall", "false_drift_rate", "false_poison_rate",
        "process_masking_rate", "false_alarm_rate", "harm", "useful_frac"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    a = ap.parse_args()
    seeds = EVAL_SEEDS[:a.seeds]
    get_v2_models()
    bl = get_ml_baselines()

    banner(f"EXPERIMENT 8 - distribution shift, {len(seeds)} seeds x 6 scenarios")
    out: Dict[str, Dict] = {}
    for label, shift, perr, rr in CONDITIONS:
        specs = eval_specs(seeds=seeds, shift=shift, rate_range=rr)
        rows = {"PIRCAD": pmap(run_pircad, [(s, RunCfg(param_error=perr)) for s in specs],
                               label)}
        if perr == 1.0:
            eps = [s.build() for s in specs]
            for name, model in bl.items():
                rows[name] = [score_baseline(s, ep, model.predict(ep))
                              for s, ep in zip(specs, eps)]
        out[label] = {}
        line = f"{label:22s}"
        for sysname, r in rows.items():
            summ = {k: bootstrap_ci([x.get(k, np.nan) for x in r]) for k in KEYS}
            out[label][sysname] = {"summary": summ, "rows": strip_private(r)}
            line += (f" | {sysname[:10]:10s} mF1 {summ['macro_f1'][0]:.3f} "
                     f"fD {summ['false_drift_rate'][0]:.3f} "
                     f"FAR {summ['false_alarm_rate'][0]:.3f}")
        print(line, flush=True)

    save_json("exp8_distribution_shift.json",
              {"seeds": list(seeds), "conditions": [c[0] for c in CONDITIONS],
               "results": out})


if __name__ == "__main__":
    main()
