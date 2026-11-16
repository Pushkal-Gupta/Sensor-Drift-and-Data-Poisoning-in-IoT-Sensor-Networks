"""
Experiment 17 -- one mitigation of the noise-shift weakness (first-round review).

Section 6.3 found PIRCAD's false-drift rate climbs when the transducers become
noisier than at commissioning, because the root-cause learner only ever saw
features at the nominal noise level.  The cheapest mitigation needs no new
mechanism: collect the training corpus at more than one noise level.

Here the corpus is the usual 72 training episodes at noise x1 plus the same
episodes at x1.5 and x2 (never x3), the random forest is refitted, and the
model is evaluated on the experiment-8 conditions nominal, noise x2 and
noise x3 (x3 is outside the augmented range).  The same augmentation is applied
to the generic-feature forest so the comparison stays fair.

Run:  python experiments/exp17_noise_augmentation.py
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from v2_common import (EVAL_SEEDS, RunCfg, _collect, _make_estimator, _single_thread,
                       accelerate, banner, bootstrap_ci, eval_specs, pmap,
                       run_pircad, save_json, score_baseline, train_specs, wilcoxon)

from pircad.scenarios import Shift

LEVELS = (1.0, 1.5, 2.0)
CONDS = {"nominal": Shift(), "noise x2": Shift(noise_scale=2.0),
         "noise x3": Shift(noise_scale=3.0)}
KEYS = ["macro_f1", "false_drift_rate", "false_poison_rate", "false_alarm_rate",
        "drift_recall", "poison_recall"]

_AUG = {}


def _aug_classifier():
    import joblib
    from pircad.rootcause import HybridRootCauseClassifier
    from v2_common import ARTIFACTS
    if "clf" not in _AUG:
        _AUG["clf"] = HybridRootCauseClassifier(
            accelerate(joblib.load(ARTIFACTS / "exp17_model.joblib")))
    return _AUG["clf"]


def _run_aug(spec):
    import v2_common as V
    orig = V.make_classifier
    V.make_classifier = lambda backend: _aug_classifier()
    try:
        return run_pircad((spec, RunCfg()))
    finally:
        V.make_classifier = orig


def main():
    import joblib
    from pircad.baselines_ml import GenericFeatureRF
    from pircad.rootcause import MLRootCauseClassifier
    from v2_common import ARTIFACTS, get_v2_models
    banner("EXP 17 - noise-augmented training")
    base = get_v2_models()
    specs = [replace(s, shift=Shift(noise_scale=lv)) for lv in LEVELS[1:] for s in train_specs()]
    parts = pmap(_collect, specs, "aug corpus")
    X = np.vstack([base["X"]] + [p[0] for p in parts if len(p[1])])
    y = np.concatenate([base["y"]] + [p[1] for p in parts if len(p[1])])
    clf = MLRootCauseClassifier(model=_make_estimator("rf")).fit(X, y)
    _single_thread(clf.model)
    joblib.dump(clf, ARTIFACTS / "exp17_model.joblib")
    print(f"  corpus {X.shape}")
    gen = GenericFeatureRF().fit([s.build() for s in train_specs()] +
                                 [s.build() for s in specs])

    out = {}
    seeds = EVAL_SEEDS[:10]
    import json
    e8 = json.load(open(ARTIFACTS / "exp8_distribution_shift.json"))
    for name, sh in CONDS.items():
        es = eval_specs(seeds=seeds, shift=sh)
        aug = pmap(_run_aug, es, name)
        eps = [s.build() for s in es]
        g = [score_baseline(s, ep, gen.predict(ep)) for s, ep in zip(es, eps)]
        ref = {(r["seed"], r["scenario"]): r for r in e8["results"][name]["PIRCAD"]["rows"]}
        refg = {(r["seed"], r["scenario"]): r for r in e8["results"][name]["Generic-feature RF"]["rows"]}
        keyz = [(r["seed"], r["scenario"]) for r in aug]
        out[name] = {
            "PIRCAD": {k: bootstrap_ci([ref[q][k] for q in keyz]) for k in KEYS},
            "PIRCAD noise-augmented": {k: bootstrap_ci([r[k] for r in aug]) for k in KEYS},
            "Generic-feature RF": {k: bootstrap_ci([refg[q][k] for q in keyz]) for k in KEYS},
            "Generic-feature RF noise-augmented": {k: bootstrap_ci([r[k] for r in g]) for k in KEYS},
            "test_aug_vs_orig": wilcoxon([r["macro_f1"] for r in aug],
                                         [ref[q]["macro_f1"] for q in keyz]),
            "test_aug_vs_generic_aug": wilcoxon([r["macro_f1"] for r in aug],
                                                [r["macro_f1"] for r in g]),
        }
        o = out[name]
        print(f"  {name:10s} PIRCAD {o['PIRCAD']['macro_f1'][0]:.3f} -> aug "
              f"{o['PIRCAD noise-augmented']['macro_f1'][0]:.3f} (fD "
              f"{o['PIRCAD']['false_drift_rate'][0]:.3f} -> "
              f"{o['PIRCAD noise-augmented']['false_drift_rate'][0]:.3f}) | generic "
              f"{o['Generic-feature RF']['macro_f1'][0]:.3f} -> aug "
              f"{o['Generic-feature RF noise-augmented']['macro_f1'][0]:.3f}", flush=True)
    save_json("exp17_noise_augmentation.json", {"levels": list(LEVELS), "results": out})


if __name__ == "__main__":
    main()
