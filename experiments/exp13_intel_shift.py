"""
Experiment 13 -- does the robustness pattern of experiment 8 hold on real data?

The models fitted in experiment 10 (PIRCAD's root-cause stage and every
baseline, all trained on the original Intel Lab train motes) are evaluated,
without retraining, on the held-out test motes after a transducer-noise shift:
every mote in the deployment -- targets and neighbours alike -- is given extra
white noise, as if the sensor batch had been replaced by a noisier one.

Run after exp10:  python experiments/exp13_intel_shift.py
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

import exp10_intel_lab as e10
from exp10_intel_lab import (ARTIFACTS, SCENARIOS, _G1, build, score_labels,  # noqa: F401
                             specs_for)
from v2_common import banner, bootstrap_ci, pmap, save_json, strip_private

LEVELS = [("original", 0.0, 0.0), ("noise +0.10 C / +0.3 %RH", 0.10, 0.3),
          ("noise +0.20 C / +0.6 %RH", 0.20, 0.6)]
KEYS = ["macro_f1", "root_cause_accuracy", "drift_recall", "poison_recall",
        "process_recall", "false_drift_rate", "false_poison_rate", "false_alarm_rate"]


def _noisy_data(sT: float, sH: float) -> None:
    """Replace the cached Intel frames in this process by noisier copies."""
    d = e10.data()
    if sT == 0 and sH == 0:
        return
    rng = np.random.default_rng(12345)
    d["T"] = d["T"] + rng.normal(0, sT, d["T"].shape)
    d["H"] = (d["H"] + rng.normal(0, sH, d["H"].shape)).clip(1, 99)


def _run(args):
    spec, sT, sH = args
    _noisy_data(sT, sH) if not getattr(_run, "_done", False) else None
    _run._done = True
    return e10._run((spec, "rf"))


def main() -> None:
    import joblib
    d = e10.data()
    meta = __import__("json").loads((ARTIFACTS / "exp10_intel_lab.json").read_text())
    test_m = meta["test_motes"]
    specs = specs_for(test_m, 2)
    fitted = joblib.load(ARTIFACTS / "exp10_baselines.joblib")
    banner(f"EXPERIMENT 13 - Intel Lab noise shift, {len(test_m)} test motes")
    base_T, base_H = d["T"].copy(), d["H"].copy()
    out: Dict[str, Dict] = {}
    for label, sT, sH in LEVELS:
        rows = {"PIRCAD-rf": pmap(_run, [(s, sT, sH) for s in specs], label)}
        d["T"], d["H"] = base_T.copy(), base_H.copy()
        _noisy_data(sT, sH)
        eps = [build(s) for s in specs]
        for name, model in fitted.items():
            rows[name] = [score_labels(s, ep, model.predict(ep)) for s, ep in zip(specs, eps)]
        out[label] = {}
        line = f"{label:26s}"
        for sysname, r in rows.items():
            summ = {k: bootstrap_ci([x.get(k, np.nan) for x in r]) for k in KEYS}
            out[label][sysname] = {"summary": summ, "rows": strip_private(r)}
            line += (f" | {sysname[:10]:10s} mF1 {summ['macro_f1'][0]:.3f} "
                     f"FAR {summ['false_alarm_rate'][0]:.3f}")
        print(line, flush=True)
        d["T"], d["H"] = base_T.copy(), base_H.copy()
    save_json("exp13_intel_shift.json", {"levels": [l[0] for l in LEVELS], "results": out})


if __name__ == "__main__":
    main()
