"""
Experiment 3 -- ablation: what each claimed element is actually worth.

Each row removes exactly one element and leaves everything else in place.  This
is the evidence that the combination is doing the work, rather than one component
carrying the result -- which is the specific objection the search report raised
against Generation 1 ("a predictable integration of known techniques").

  full                    all elements
  -physics residual       expectation from the channel's own rolling mean
                          instead of the process model (the Gen-1 notion)
  -cross-sensor consistency   no constraint graph, so no instrument-versus-process
                          separation and no localisation
  -excitation coherence   the residual/prediction coherence feature removed from
                          the root-cause vector
  -noise-signature feats  the innovation-statistics features removed
  -differential response  one response for every cause (prior-art behaviour)

Run:  python experiments/exp3_ablation.py
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Sequence

import numpy as np

from _common import (EVAL_SEEDS, TRAIN_SEEDS, WARMUP, banner, eval_episodes,
                     get_models, save_json)

from pircad.config import CHANNELS, FogConfig
from pircad.consistency import NullConsistencyMonitor
from pircad.fognode import PIRCADFogNode
from pircad.metrics import aggregate, format_table
from pircad.rootcause import (ALL_FEATURES, HybridRootCauseClassifier,
                              MLRootCauseClassifier)
from pircad.simulator import frames
from pircad.training import build_corpus, training_episodes
from pircad.twin import PhysicsVesselTwin, RollingMeanTwin

KEYS = ["root_cause_accuracy", "drift_recall", "poison_recall", "process_recall",
        "false_drift_rate", "false_poison_rate", "process_masking_rate",
        "false_alarm_rate", "poison_containment", "retraining_protection"]

COHERENCE_FEATS = ["dyn_coherence"]
NOISE_FEATS = ["acf_shift", "var_ratio", "flatness", "hf_ratio", "kurt_incr"]
ONSET_FEATS = ["corner", "jump_max", "monotone_frac", "slope_ratio"]
#: everything that describes the *character* of the deviation rather than its
#: size.  Individually these are redundant with one another; the group is not.
CHARACTER_FEATS = COHERENCE_FEATS + NOISE_FEATS + ONSET_FEATS
LOC_FEATS = ["loc_score", "loc_margin", "loc_magnitude", "loc_is_self",
             "loc_is_process"]


def masked_classifier(corpus, drop: Sequence[str], seed: int = 0):
    """Refit the root-cause model with some feature columns zeroed out."""
    idx = [ALL_FEATURES.index(f) for f in drop]
    X = corpus.X.copy()
    X[:, idx] = 0.0
    return MLRootCauseClassifier().fit(X, corpus.y, seed=seed), idx


class MaskedWrapper:
    """Applies the same column mask at inference time."""

    def __init__(self, inner, idx):
        self.inner, self.idx = inner, idx

    def classify(self, channel, feats, loc):
        f = dict(feats)
        for i in self.idx:
            f[ALL_FEATURES[i]] = 0.0
        return self.inner.classify(channel, f, loc)


def evaluate(name: str, episodes, make_node) -> Dict[str, float]:
    from pircad.metrics import evaluate_run
    rows: List[dict] = []
    for ep in episodes:
        node = make_node()
        res = node.run(frames(ep), truth_labels=ep.labels)
        rows.append(evaluate_run(ep, res, WARMUP))
    out = aggregate(rows)
    print(f"  {name:34s} rc={out['root_cause_accuracy']:.3f}  "
          f"fdr={out.get('false_drift_rate', float('nan')):.3f}  "
          f"pmask={out.get('process_masking_rate', float('nan')):.3f}")
    return out


def main() -> None:
    models = get_models()
    corpus = models["corpus"]
    base_clf = models["root_cause"]
    episodes = eval_episodes()

    banner("EXPERIMENT 3 - ablation")
    print(f"{len(episodes)} held-out episodes; one element removed per row\n")

    results: Dict[str, Dict[str, float]] = {}

    results["full"] = evaluate(
        "full", episodes,
        lambda: PIRCADFogNode(classifier=HybridRootCauseClassifier(base_clf)))

    # --- remove the physics residual --------------------------------------
    print("  (refitting the root-cause model on rolling-mean residuals...)")
    rm_eps = training_episodes(TRAIN_SEEDS, 4000)

    def rm_node_factory(clf=None):
        def make():
            twin = RollingMeanTwin(CHANNELS)
            return PIRCADFogNode(twin=twin, classifier=clf,
                                 consistency=NullConsistencyMonitor())
        return make

    rm_corpus = build_corpus_with(rm_eps, rm_node_factory(None))
    rm_clf = MLRootCauseClassifier().fit(rm_corpus[0], rm_corpus[1])
    results["-physics residual"] = evaluate(
        "-physics residual", episodes, rm_node_factory(rm_clf))

    # --- remove the cross-sensor consistency graph -------------------------
    print("  (refitting without localisation features...)")
    nc_clf, nc_idx = masked_classifier(corpus, LOC_FEATS)
    results["-cross-sensor consistency"] = evaluate(
        "-cross-sensor consistency", episodes,
        lambda: PIRCADFogNode(classifier=MaskedWrapper(nc_clf, nc_idx),
                              consistency=NullConsistencyMonitor()))

    # --- remove single feature groups -------------------------------------
    for label, feats in (("-excitation coherence", COHERENCE_FEATS),
                         ("-noise-signature feats", NOISE_FEATS),
                         ("-all character feats", CHARACTER_FEATS)):
        clf, idx = masked_classifier(corpus, feats)
        results[label] = evaluate(
            label, episodes,
            lambda c=clf, i=idx: PIRCADFogNode(
                classifier=HybridRootCauseClassifier(MaskedWrapper(c, i))))

    # --- remove the differential response ---------------------------------
    def nodiff():
        cfg = FogConfig()
        cfg.differential_response = False
        return PIRCADFogNode(classifier=HybridRootCauseClassifier(base_clf), cfg=cfg)

    results["-differential response"] = evaluate("-differential response",
                                                 episodes, nodiff)

    banner("Ablation table")
    print(format_table(results, KEYS, width=28))
    save_json("exp3_ablation.json", results)


def build_corpus_with(episodes, make_node):
    """Collect a training corpus using a non-standard node factory."""
    from pircad.rootcause import build_vector
    X, y = [], []
    for ep in episodes:
        rows_x, rows_y = [], []

        def probe(k, channel, feats, loc, fired, _x=rows_x, _y=rows_y, _ep=ep):
            label = int(_ep.labels[k])
            if _ep.target_channel is not None and channel != _ep.target_channel:
                label = 0
            _x.append(build_vector(channel, feats, loc))
            _y.append(label)

        node = make_node()
        node.feature_probe = probe
        node.run(frames(ep))
        X.extend(rows_x); y.extend(rows_y)
    return np.vstack(X), np.asarray(y, dtype=int)


if __name__ == "__main__":
    main()
