"""
Experiment 10 -- real sensor data: the Intel Berkeley Research Lab deployment.

The clean signal is real: 51 usable Mica2Dot motes, 11 days, 2-minute samples,
with all the nonstationarity, packet loss and local micro-climate a real office
has.  Ground truth for drift and poisoning can only exist if the events are
injected, so they are injected into the real streams (semi-synthetic), and the
genuine process event is a physically consistent local heat source (T rises, RH
falls at constant vapour pressure).

Protocol: motes are split at random into disjoint train and test halves; every
model -- PIRCAD's root-cause stage and every data-driven baseline -- is fitted
on train motes only and scored on test motes it has never seen.  Clean test
episodes measure the false-alarm rate on genuinely unmodified real data.

Run:  python experiments/exp10_intel_lab.py
"""

from __future__ import annotations

import time
from typing import Dict, List

import numpy as np

from v2_common import (ARTIFACTS, N_WORKERS, banner, bootstrap_ci, detection_f1,
                       holm, macro_f1_episode, pmap, save_json, strip_private,
                       wilcoxon)

from pircad.indoor import (INDOOR_CHANNELS, WARMUP_INDOOR, IndoorConsistencyMonitor,
                           IndoorSpec, IndoorTwin, indoor_config, indoor_frames,
                           load_intel, make_indoor_episode, usable_motes)
from pircad.types import RootCause

SCENARIOS = ["clean", "drift", "stealth_poison", "spike_poison", "replay_poison",
             "process_event"]
REPS = 2
KEYS = ["root_cause_accuracy", "macro_f1", "drift_recall", "poison_recall",
        "process_recall", "false_drift_rate", "false_poison_rate",
        "process_masking_rate", "false_alarm_rate", "detection_f1", "harm", "useful_frac"]

_DATA = {}


def data():
    if not _DATA:
        T, H, L = load_intel()
        _DATA.update(T=T, H=H, L=L, pool=usable_motes(T, H))
    return _DATA


def build(spec: IndoorSpec):
    d = data()
    return make_indoor_episode(spec, d["T"], d["H"], d["L"], d["pool"])


def specs_for(motes, base_seed: int, attack: str = "substitute") -> List[IndoorSpec]:
    out = []
    for m in motes:
        for j, sc in enumerate(SCENARIOS):
            for r in range(REPS):
                seed = base_seed + 1000 * m + 10 * j + r
                rng = np.random.default_rng(seed)
                out.append(IndoorSpec(int(m), sc, seed, int(rng.integers(1800, 3600)),
                                      float(rng.uniform(0.6e-3, 2.0e-3)), attack))
    return out


def node_for(clf):
    from pircad.fognode import PIRCADFogNode
    tw = IndoorTwin()
    return PIRCADFogNode(twin=tw, classifier=clf, cfg=indoor_config(),
                         channels=INDOOR_CHANNELS,
                         consistency=IndoorConsistencyMonitor(tw))


def _collect(spec: IndoorSpec):
    from pircad.rootcause import build_vector
    ep = build(spec)
    X, y = [], []

    def probe(k, ch, feats, loc, fired):
        lab = int(ep.labels[k])
        if ep.target_channel is not None and ch != ep.target_channel:
            lab = int(RootCause.NORMAL)
        X.append(build_vector(ch, feats, loc)); y.append(lab)

    from pircad.fognode import PIRCADFogNode
    tw = IndoorTwin()
    PIRCADFogNode(twin=tw, classifier=None, cfg=indoor_config(),
                  channels=INDOOR_CHANNELS, consistency=IndoorConsistencyMonitor(tw),
                  feature_probe=probe).run(indoor_frames(ep))
    return (np.vstack(X) if X else np.zeros((0, 22))), np.asarray(y, int)


_CLF = {}


def _run(args):
    spec, backend = args
    import joblib
    from pircad.metrics import evaluate_run
    from pircad.rootcause import HybridRootCauseClassifier, RuleRootCauseClassifier
    from v2_common import accelerate, effect_metrics
    if backend not in _CLF:
        _CLF[backend] = (RuleRootCauseClassifier() if backend == "rules" else
                         HybridRootCauseClassifier(accelerate(
                             joblib.load(ARTIFACTS / "exp10_models.joblib")[backend])))
    ep = build(spec)
    res = node_for(_CLF[backend]).run(indoor_frames(ep), truth_labels=ep.labels)
    m = evaluate_run(ep, res, WARMUP_INDOOR)
    m.update(effect_metrics(ep, res))
    m.update(detection_f1_w(ep.labels, res.causes() != 0))
    m["macro_f1"] = macro_f1_episode(ep.labels[WARMUP_INDOOR:], res.causes()[WARMUP_INDOOR:])
    m["scenario"] = spec.scenario; m["mote"] = spec.mote; m["seed"] = spec.seed
    return m


def detection_f1_w(labels, anom):
    import v2_common
    old = v2_common.WARMUP
    v2_common.WARMUP = WARMUP_INDOOR
    try:
        return detection_f1(labels, anom)
    finally:
        v2_common.WARMUP = old


def score_labels(spec, ep, y):
    from pircad.metrics import evaluate_labels
    m = evaluate_labels(ep, y, WARMUP_INDOOR)
    m.update(detection_f1_w(ep.labels, np.asarray(y) != 0))
    m["macro_f1"] = macro_f1_episode(ep.labels[WARMUP_INDOOR:], np.asarray(y)[WARMUP_INDOOR:])
    if ep.target_channel:
        s = int(ep.meta["onset"]); f = np.asarray(y)[s:] == 0
        err = np.abs(ep.obs["T"][s:] - ep.truth["T"][s:])
        m["harm"] = float(np.where(f, err, 0.0).mean())
        m["useful_frac"] = float((f & (err < 0.5)).mean())
    m["scenario"] = spec.scenario; m["mote"] = spec.mote; m["seed"] = spec.seed
    return m


class _G1:
    name = "Gen-1 FHAD"

    def fit(self, eps):
        from pircad.baselines import Gen1FHAD
        self.m = Gen1FHAD().fit(eps, channel="T")
        return self

    def predict(self, ep):
        return self.m.run(ep, channel="T")


def main() -> None:
    import joblib
    d = data()
    pool = d["pool"]
    rng = np.random.default_rng(0)
    perm = rng.permutation(pool)
    train_m, test_m = sorted(perm[: len(pool) // 2]), sorted(perm[len(pool) // 2:])
    banner(f"EXPERIMENT 10 - Intel Lab, {len(pool)} motes: "
           f"{len(train_m)} train / {len(test_m)} test (disjoint)")
    tr_specs = specs_for(train_m, 1)
    te_specs = specs_for(test_m, 2)
    mitm_specs = [s for s in specs_for(test_m, 3, "mitm") if s.scenario == "stealth_poison"]

    # ---- PIRCAD root-cause stage, trained on train motes -----------------
    parts = pmap(_collect, tr_specs, "corpus")
    X = np.vstack([p[0] for p in parts if len(p[1])])
    y = np.concatenate([p[1] for p in parts if len(p[1])])
    print(f"  corpus {X.shape} classes {np.bincount(y, minlength=4)}")
    from v2_common import _make_estimator, _single_thread
    from pircad.rootcause import MLRootCauseClassifier
    models = {}
    for b in ("rf", "hgb"):
        clf = MLRootCauseClassifier(model=_make_estimator(b)).fit(X, y)
        _single_thread(clf.model)
        models[b] = clf
    joblib.dump(models, ARTIFACTS / "exp10_models.joblib")

    results: Dict[str, List[Dict]] = {}
    for b in ("rf", "hgb", "rules"):
        results[f"PIRCAD-{b}"] = pmap(_run, [(s, b) for s in te_specs], f"PIRCAD-{b}")
    mitm = {"PIRCAD-rf": pmap(_run, [(s, "rf") for s in mitm_specs], "mitm")}

    # ---- baselines, same train motes --------------------------------------
    from pircad.baselines import Gen1FHAD
    from pircad.baselines_ml import GenericFeatureRF, MultiScaleCNN
    tr_eps = [build(s) for s in tr_specs]
    te_eps = [build(s) for s in te_specs]
    mi_eps = [build(s) for s in mitm_specs]

    # baselines do not depend on the fog node, so they are fitted once and cached
    bl_cache = ARTIFACTS / "exp10_baselines.joblib"
    fitted = joblib.load(bl_cache) if bl_cache.exists() else {}
    for cls, kw in ((_G1, {}), (GenericFeatureRF, {"stride": 4}),
                    (MultiScaleCNN, {"stride": 8})):
        t0 = time.time()
        if cls.name not in fitted:
            fitted[cls.name] = cls(**kw).fit(tr_eps)
            joblib.dump(fitted, bl_cache)
        model = fitted[cls.name]
        results[cls.name] = [score_labels(s, ep, model.predict(ep))
                             for s, ep in zip(te_specs, te_eps)]
        mitm[cls.name] = [score_labels(s, ep, model.predict(ep))
                          for s, ep in zip(mitm_specs, mi_eps)]
        print(f"  {cls.name}: {time.time() - t0:.0f}s", flush=True)

    # ---- report -------------------------------------------------------------
    banner("Intel Lab, held-out motes: mean [95% CI]")
    summary = {}
    for name, rows in results.items():
        summary[name] = {k: bootstrap_ci([r.get(k, np.nan) for r in rows]) for k in KEYS}
        s = summary[name]
        print(f"{name:20s} acc {s['root_cause_accuracy'][0]:.3f} mF1 {s['macro_f1'][0]:.3f} "
              f"drift {s['drift_recall'][0]:.3f} poison {s['poison_recall'][0]:.3f} "
              f"proc {s['process_recall'][0]:.3f} fD {s['false_drift_rate'][0]:.3f} "
              f"fP {s['false_poison_rate'][0]:.3f} FAR {s['false_alarm_rate'][0]:.3f} "
              f"harm {s['harm'][0]:.3f}")
    per_sc = {name: {sc: bootstrap_ci([r["root_cause_accuracy"] for r in rows
                                       if r["scenario"] == sc]) for sc in SCENARIOS}
              for name, rows in results.items()}
    banner("Per-scenario accuracy")
    for name, d_ in per_sc.items():
        print(f"{name:20s}" + "".join(f"{sc[:10]:>11s} {v[0]:.3f}" for sc, v in d_.items()))
    mitm_sum = {name: {k: bootstrap_ci([r.get(k, np.nan) for r in rows])
                       for k in ("poison_recall", "false_drift_rate", "harm", "useful_frac")}
                for name, rows in mitm.items()}
    banner("On-path (MITM) ramp on held-out motes")
    for name, s in mitm_sum.items():
        print(f"{name:20s} poison {s['poison_recall'][0]:.3f} fD {s['false_drift_rate'][0]:.3f} "
              f"harm {s['harm'][0]:.3f}")

    ref = results["PIRCAD-rf"]
    raw = {n: wilcoxon([r["macro_f1"] for r in ref], [r["macro_f1"] for r in rows])
           for n, rows in results.items() if n != "PIRCAD-rf"}
    adj = holm({k: v["p"] for k, v in raw.items()})
    for k in raw:
        raw[k]["p_holm"] = adj[k]
        print(f"  macro-F1 PIRCAD-rf vs {k:20s} median diff {raw[k]['median_diff']:+.3f} "
              f"p_holm {adj[k]:.2e}")

    save_json("exp10_intel_lab.json", {
        "train_motes": [int(m) for m in train_m], "test_motes": [int(m) for m in test_m],
        "n_test_episodes": len(te_specs), "summary": summary, "per_scenario": per_sc,
        "mitm": mitm_sum, "tests": raw,
        "rows": {k: strip_private(v) for k, v in results.items()},
    })


if __name__ == "__main__":
    main()
