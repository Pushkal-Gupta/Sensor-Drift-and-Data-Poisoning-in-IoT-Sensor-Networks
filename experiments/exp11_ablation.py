"""
Experiment 11 -- ablation at 30 seeds, with paired tests.

The 4-seed ablation of experiment 3, re-run on the 180-episode bank, plus two
rows that test claims experiment 3 only argued:

  - dedicated observers     every channel's expectation comes from ONE shared
                            closed-loop estimator that also uses that channel
                            (the conventional state-estimation configuration)
  - adaptation governor     every adaptive element keeps learning whatever the
                            verdict: state re-anchoring, residual statistics

Every row that changes the features refits the root-cause model on a corpus
collected under the same change, so no row is handicapped by a train/test
mismatch.

Run:  python experiments/exp11_ablation.py
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np

from v2_common import (RunCfg, WARMUP, accelerate, banner, bootstrap_ci, effect_metrics,
                       eval_specs, get_v2_models, holm, macro_f1_episode, pmap,
                       save_json, strip_private, train_specs, wilcoxon)

from pircad.rootcause import ALL_FEATURES

COHERENCE = ["dyn_coherence"]
NOISE = ["acf_shift", "var_ratio", "flatness", "hf_ratio", "kurt_incr"]
ONSET = ["corner", "jump_max", "monotone_frac", "slope_ratio"]
CHARACTER = COHERENCE + NOISE + ONSET
LOC = ["loc_score", "loc_margin", "loc_magnitude", "loc_is_self", "loc_is_process"]

ROWS = ["full", "- physics residual (rolling mean)", "- dedicated observers",
        "- cross-sensor consistency", "- excitation coherence",
        "- noise-signature features", "- all character features",
        "- adaptation governor", "- differential response"]
#: rows whose feature distribution differs from the full system's, so they need
#: their own training corpus
STRUCTURAL = {"- physics residual (rolling mean)": "rolling",
              "- dedicated observers": "shared",
              "- cross-sensor consistency": "nocons",
              "- adaptation governor": "nogov"}
MASKS = {"- excitation coherence": COHERENCE, "- noise-signature features": NOISE,
         "- all character features": CHARACTER}
KEYS = ["root_cause_accuracy", "macro_f1", "false_drift_rate", "poison_containment",
        "process_masking_rate", "retraining_protection", "harm", "false_alarm_rate"]


# --------------------------------------------------------------------------
# structural variants
# --------------------------------------------------------------------------
def _shared_twin():
    from pircad.twin import PhysicsVesselTwin

    class SharedObserverTwin(PhysicsVesselTwin):
        """One closed-loop estimator for every channel (no leave-one-out)."""

        def predict(self, frame):
            out = super().predict(frame)
            T = self._T_pred["full"]
            out.update({"T_in": T, "P": self.P_of_T(T), "RH": self.RH_of_T(T)})
            self._pred_cache = dict(out)
            return out

    return SharedObserverTwin()


def _open_governor(gov):
    """Neutralise the adaptation gate: everything always learns."""
    chans = list(gov.trust_weights())
    gov.trust_weights = lambda: {c: 1.0 for c in chans}
    gov.adaptation_allowed = lambda c: True
    gov.training_frozen = lambda: False
    return gov


def make_node(variant: str, classifier, cfg=None):
    from pircad.config import CHANNELS, FogConfig
    from pircad.consistency import NullConsistencyMonitor
    from pircad.fognode import PIRCADFogNode
    from pircad.twin import RollingMeanTwin
    cfg = cfg or FogConfig()
    if variant == "rolling":
        return PIRCADFogNode(twin=RollingMeanTwin(CHANNELS), classifier=classifier,
                             cfg=cfg, consistency=NullConsistencyMonitor())
    if variant == "shared":
        return PIRCADFogNode(twin=_shared_twin(), classifier=classifier, cfg=cfg)
    if variant == "nocons":
        return PIRCADFogNode(classifier=classifier, cfg=cfg,
                             consistency=NullConsistencyMonitor())
    node = PIRCADFogNode(classifier=classifier, cfg=cfg)
    if variant == "nogov":
        _open_governor(node.gov)
    return node


def _collect(args):
    spec, variant = args
    from pircad.rootcause import build_vector
    from pircad.simulator import frames
    from pircad.types import RootCause
    ep = spec.build()
    X, y = [], []

    def probe(k, ch, feats, loc, fired):
        lab = int(ep.labels[k])
        if ep.target_channel is not None and ch != ep.target_channel:
            lab = int(RootCause.NORMAL)
        X.append(build_vector(ch, feats, loc)); y.append(lab)

    node = make_node(variant, None)
    node.feature_probe = probe
    node.run(frames(ep))
    return (np.vstack(X) if X else np.zeros((0, len(ALL_FEATURES)))), np.asarray(y, int)


class Masked:
    def __init__(self, inner, idx):
        self.inner, self.idx = inner, idx

    def classify(self, channel, feats, loc):
        f = dict(feats)
        for i in self.idx:
            if ALL_FEATURES[i] in f:
                f[ALL_FEATURES[i]] = 0.0
        return self.inner.classify(channel, f, loc)


_CLFS: Dict[str, object] = {}


def _classifier(row: str):
    import joblib
    from v2_common import ARTIFACTS
    from pircad.rootcause import HybridRootCauseClassifier
    if row not in _CLFS:
        blob = joblib.load(ARTIFACTS / "exp11_models.joblib")
        clf, idx = blob[row]
        accelerate(clf)
        if row == "- cross-sensor consistency" or row == "- physics residual (rolling mean)":
            # no localisation available: the hybrid's physical override cannot fire
            _CLFS[row] = Masked(clf, idx)
        else:
            _CLFS[row] = HybridRootCauseClassifier(Masked(clf, idx))
    return _CLFS[row]


def _run(args):
    spec, row = args
    from pircad.config import FogConfig
    from pircad.metrics import evaluate_run
    from pircad.simulator import frames
    ep = spec.build()
    cfg = FogConfig()
    if row == "- differential response":
        cfg.differential_response = False
    node = make_node(STRUCTURAL.get(row, "full"), _classifier(row), cfg)
    res = node.run(frames(ep), truth_labels=ep.labels)
    m = evaluate_run(ep, res, WARMUP)
    m.update(effect_metrics(ep, res))
    m["macro_f1"] = macro_f1_episode(ep.labels[WARMUP:], res.causes()[WARMUP:])
    m["scenario"] = spec.scenario; m["seed"] = spec.seed
    return m


def main() -> None:
    import joblib
    from v2_common import ARTIFACTS, _make_estimator, _single_thread
    from pircad.rootcause import MLRootCauseClassifier
    base = get_v2_models()
    banner("EXPERIMENT 11 - ablation, 180 held-out episodes")

    models = {}
    for row in ROWS:
        if row in STRUCTURAL:
            parts = pmap(_collect, [(s, STRUCTURAL[row]) for s in train_specs()], row)
            X = np.vstack([p[0] for p in parts if len(p[1])])
            y = np.concatenate([p[1] for p in parts if len(p[1])])
        else:
            X, y = base["X"].copy(), base["y"]
        drop = MASKS.get(row, [])
        if row in ("- cross-sensor consistency", "- physics residual (rolling mean)"):
            drop = drop + LOC
        idx = [ALL_FEATURES.index(f) for f in drop]
        X[:, idx] = 0.0
        clf = MLRootCauseClassifier(model=_make_estimator("rf")).fit(X, y)
        _single_thread(clf.model)
        models[row] = (clf, idx)
        print(f"  fitted {row}: corpus {X.shape}", flush=True)
    joblib.dump(models, ARTIFACTS / "exp11_models.joblib")

    specs = eval_specs()
    results = {row: pmap(_run, [(s, row) for s in specs], row) for row in ROWS}

    summary = {row: {k: bootstrap_ci([r.get(k, np.nan) for r in rows]) for k in KEYS}
               for row, rows in results.items()}
    raw = {row: wilcoxon([r["macro_f1"] for r in results["full"]],
                         [r["macro_f1"] for r in rows])
           for row, rows in results.items() if row != "full"}
    adj = holm({k: v["p"] for k, v in raw.items()})
    banner("Ablation: mean [95% CI]; paired Wilcoxon on macro-F1 vs full")
    for row in ROWS:
        s = summary[row]
        p = f"p_holm {adj[row]:.1e}" if row in adj else ""
        print(f"{row:36s} acc {s['root_cause_accuracy'][0]:.3f} mF1 {s['macro_f1'][0]:.3f} "
              f"fD {s['false_drift_rate'][0]:.3f} contain {s['poison_containment'][0]:.3f} "
              f"mask {s['process_masking_rate'][0]:.3f} protect "
              f"{s['retraining_protection'][0]:.3f} harm {s['harm'][0]:.3f} {p}")
        if row in raw:
            raw[row]["p_holm"] = adj[row]
    save_json("exp11_ablation.json", {"summary": summary, "tests": raw,
                                      "rows": {k: strip_private(v) for k, v in results.items()}})


if __name__ == "__main__":
    main()
