"""
Experiment 12 -- physics features + generic features in one learner.

Experiment 6 shows that a generic-feature random forest matches or beats
PIRCAD's root-cause labels in distribution, and experiment 8 shows it collapses
under a transducer-noise shift.  The obvious question is whether the two
feature sets are complementary: here PIRCAD's root-cause model receives its 22
physics-anchored features *plus* the generic window features (same governance,
same response).  It is evaluated on the experiment-6 bank and on the shift
conditions of experiment 8 that separated the systems.

Run:  python experiments/exp12_hybrid_features.py
"""

from __future__ import annotations

import time
from typing import Dict

import numpy as np

from v2_common import (ARTIFACTS, EVAL_SEEDS, WARMUP, Spec, _make_estimator, _single_thread,
                       accelerate, banner, bootstrap_ci, detection_f1, effect_metrics,
                       eval_specs, macro_f1_episode, pmap, save_json, strip_private,
                       train_specs)

from pircad.scenarios import Shift

CONDITIONS = [("nominal (exp. 6 bank)", Shift(), 30),
              ("plant dynamics +20%", Shift(plant_scale=1.20), 10),
              ("noise x2", Shift(noise_scale=2.0), 10),
              ("noise x3", Shift(noise_scale=3.0), 10)]
KEYS = ["macro_f1", "root_cause_accuracy", "false_drift_rate", "false_poison_rate",
        "process_masking_rate", "false_alarm_rate", "harm", "useful_frac"]


class Augmented:
    """Root-cause model over [physics features | generic features at sample k]."""

    def __init__(self, model):
        self.model = model
        self.F = None
        self.k = 0

    def classify(self, channel, feats, loc):
        from pircad.rootcause import RootCauseResult, build_vector
        from pircad.types import RootCause
        x = np.concatenate([build_vector(channel, feats, loc), self.F[self.k]])[None]
        p = self.model.predict_proba(x)[0]
        top = int(np.argmax(p))
        return RootCauseResult(RootCause(int(self.model.classes_[top])), float(p[top]),
                               {RootCause(int(c)): float(v) for c, v in
                                zip(self.model.classes_, p)}, "physics+generic")


def _collect(spec: Spec):
    from pircad.baselines_ml import generic_features
    from pircad.fognode import PIRCADFogNode
    from pircad.rootcause import build_vector
    from pircad.simulator import frames
    from pircad.types import RootCause
    ep = spec.build()
    F = generic_features(ep)
    X, y = [], []

    def probe(k, ch, feats, loc, fired):
        lab = int(ep.labels[k])
        if ep.target_channel is not None and ch != ep.target_channel:
            lab = int(RootCause.NORMAL)
        X.append(np.concatenate([build_vector(ch, feats, loc), F[k]])); y.append(lab)

    PIRCADFogNode(classifier=None, feature_probe=probe).run(frames(ep))
    return np.vstack(X), np.asarray(y, int)


_AUG: Dict[str, Augmented] = {}


def _run(spec: Spec):
    import joblib
    from pircad.baselines_ml import generic_features
    from pircad.fognode import PIRCADFogNode
    from pircad.metrics import evaluate_run
    from pircad.rootcause import HybridRootCauseClassifier, MLRootCauseClassifier
    from pircad.simulator import frames
    if "m" not in _AUG:
        clf = joblib.load(ARTIFACTS / "exp12_model.joblib")
        accelerate(clf)
        _AUG["m"] = Augmented(clf.model)
    aug = _AUG["m"]
    ep = spec.build()
    aug.F = generic_features(ep)
    shell = MLRootCauseClassifier(model=aug.model)
    shell.classify = aug.classify
    node = PIRCADFogNode(classifier=HybridRootCauseClassifier(shell))

    def fr():
        for f in frames(ep):
            aug.k = f.k
            yield f

    res = node.run(fr(), truth_labels=ep.labels)
    m = evaluate_run(ep, res, WARMUP)
    m.update(effect_metrics(ep, res))
    m.update(detection_f1(ep.labels, res.causes() != 0))
    m["macro_f1"] = macro_f1_episode(ep.labels[WARMUP:], res.causes()[WARMUP:])
    m["scenario"] = spec.scenario; m["seed"] = spec.seed
    return m


def main() -> None:
    import joblib
    from pircad.rootcause import MLRootCauseClassifier
    banner("EXPERIMENT 12 - physics + generic features")
    parts = pmap(_collect, train_specs(), "corpus")
    X = np.vstack([p[0] for p in parts]); y = np.concatenate([p[1] for p in parts])
    t0 = time.time()
    clf = MLRootCauseClassifier(model=_make_estimator("rf")).fit(X, y)
    _single_thread(clf.model)
    joblib.dump(clf, ARTIFACTS / "exp12_model.joblib")
    print(f"  corpus {X.shape}, fitted in {time.time() - t0:.0f}s", flush=True)

    out = {}
    for label, shift, n_seeds in CONDITIONS:
        specs = eval_specs(seeds=EVAL_SEEDS[:n_seeds], shift=shift)
        rows = pmap(_run, specs, label)
        summ = {k: bootstrap_ci([r.get(k, np.nan) for r in rows]) for k in KEYS}
        out[label] = {"summary": summ, "rows": strip_private(rows)}
        print(f"{label:24s} mF1 {summ['macro_f1'][0]:.3f} acc {summ['root_cause_accuracy'][0]:.3f} "
              f"fD {summ['false_drift_rate'][0]:.3f} FAR {summ['false_alarm_rate'][0]:.3f} "
              f"useful {summ['useful_frac'][0]:.3f}", flush=True)
    save_json("exp12_hybrid_features.json", {"conditions": [c[0] for c in CONDITIONS],
                                             "results": out})


if __name__ == "__main__":
    main()
