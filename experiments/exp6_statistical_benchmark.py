"""
Experiment 6 -- statistically powered benchmark against data-driven baselines.

180 held-out episodes (6 scenarios x 30 seeds, disjoint from the 6 training
seeds).  Compared systems:

  PIRCAD with five interchangeable root-cause back-ends
      rf (reported default), hgb, logreg, mlp, rules (no training data)
  supervised root-cause baselines, same training episodes and labels
      Gen-1 FHAD, generic-feature RF, multi-scale 1D-CNN
  unsupervised detection baselines (no cause vocabulary)
      Isolation Forest, PCA-SPE, autoencoder, Kalman chi-square

Statistics: per-episode metrics, bootstrap 95% CIs of the mean, paired
Wilcoxon signed-rank tests (PIRCAD-rf vs every other system, matched by
episode) with Holm correction, and rank-biserial effect sizes.

Run:  python experiments/exp6_statistical_benchmark.py [--quick]
"""

from __future__ import annotations

import argparse
import time
from collections import defaultdict
from typing import Dict, List

import numpy as np

from v2_common import (BACKENDS, detection_f1, EVAL_SEEDS, SCENARIOS, WARMUP, RunCfg, Spec,
                       banner, bootstrap_ci, eval_specs, get_v2_models, holm,
                       macro_f1_episode, pmap, run_pircad, save_json,
                       strip_private, train_specs, wilcoxon)

from pircad.metrics import confusion, evaluate_labels
from pircad.types import RootCause

KEYS = ["root_cause_accuracy", "macro_f1", "drift_recall", "poison_recall",
        "process_recall", "false_drift_rate", "false_poison_rate",
        "process_masking_rate", "false_alarm_rate", "detection_f1"]


def score_labels(spec: Spec, ep, y: np.ndarray) -> Dict[str, float]:
    m = evaluate_labels(ep, y, WARMUP)
    m.update(detection_f1(ep.labels, y != int(RootCause.NORMAL)))
    m["macro_f1"] = macro_f1_episode(ep.labels[WARMUP:], np.asarray(y)[WARMUP:])
    m["scenario"] = spec.scenario
    m["seed"] = spec.seed
    return m


def _gen1(spec: Spec):
    from v2_common import worker_models
    ep = spec.build()
    y = worker_models()["gen1"].run(ep)
    m = score_labels(spec, ep, y)
    m["_pred"] = np.asarray(y, dtype=np.int8)
    return m


def _kalman(args):
    spec, det = args
    ep = spec.build()
    y = det.predict(ep)
    m = detection_f1(ep.labels, y)
    m["scenario"] = spec.scenario; m["seed"] = spec.seed
    return m


def summarise(rows: List[Dict], keys=KEYS) -> Dict[str, Dict[str, float]]:
    out = {}
    for k in keys:
        vals = [r.get(k, np.nan) for r in rows]
        mean, lo, hi = bootstrap_ci(vals)
        out[k] = {"mean": mean, "lo": lo, "hi": hi,
                  "std": float(np.nanstd(np.asarray(vals, float)))}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="5 seeds only")
    args = ap.parse_args()
    seeds = EVAL_SEEDS[:5] if args.quick else EVAL_SEEDS
    specs = eval_specs(seeds=seeds)
    get_v2_models()
    results: Dict[str, List[Dict]] = {}
    preds: Dict[str, List[np.ndarray]] = {}

    banner(f"EXPERIMENT 6 - {len(specs)} held-out episodes, {len(seeds)} seeds")

    # ---------------- PIRCAD, five back-ends -------------------------------
    for b in BACKENDS:
        results[f"PIRCAD-{b}"] = pmap(run_pircad, [(s, RunCfg(backend=b)) for s in specs],
                                      f"PIRCAD-{b}")

    # ---------------- Gen-1 ------------------------------------------------
    results["Gen-1 FHAD"] = pmap(_gen1, specs, "gen1")

    # ---------------- data-driven supervised baselines ---------------------
    # the same fitted models experiments 7 and 8 use (artifacts/v2_ml_baselines)
    from v2_common import get_ml_baselines
    eval_eps = [s.build() for s in specs]
    for name, model in get_ml_baselines().items():
        t0 = time.time()
        rows = []
        for s, ep in zip(specs, eval_eps):
            y = model.predict(ep)
            m = score_labels(s, ep, y)
            m["_pred"] = y.astype(np.int8)
            rows.append(m)
        print(f"  scored {name} in {time.time() - t0:.0f}s", flush=True)
        results[name] = rows

    # ---------------- unsupervised detectors -------------------------------
    from pircad.baselines_ml import (DenseAEDetector, IsolationForestDetector,
                                     KalmanChi2Detector, PCASPEDetector)
    from pircad.scenarios import make_episode_v2
    clean_train = [make_episode_v2("clean", seed=50 + i, onset=1000) for i in range(12)]
    det_rows: Dict[str, List[Dict]] = {}
    for cls in (IsolationForestDetector, PCASPEDetector, DenseAEDetector):
        det = cls().fit(clean_train)
        rows = []
        for s, ep in zip(specs, eval_eps):
            m = detection_f1(ep.labels, det.predict(ep))
            m["scenario"] = s.scenario; m["seed"] = s.seed
            rows.append(m)
        det_rows[cls.name] = rows
        print(f"  scored {cls.name}", flush=True)
    kal = KalmanChi2Detector().fit(clean_train)
    det_rows[kal.name] = pmap(_kalman, [(s, kal) for s in specs], "kalman")

    # ---------------- report -----------------------------------------------
    banner("Root-cause metrics: mean [95% bootstrap CI] over episodes")
    summary = {name: summarise(rows) for name, rows in results.items()}
    for name, sm in summary.items():
        print(f"{name:22s} acc {sm['root_cause_accuracy']['mean']:.3f} "
              f"[{sm['root_cause_accuracy']['lo']:.3f},{sm['root_cause_accuracy']['hi']:.3f}]  "
              f"macroF1 {sm['macro_f1']['mean']:.3f}  "
              f"falseDrift {sm['false_drift_rate']['mean']:.3f}  "
              f"falsePoison {sm['false_poison_rate']['mean']:.3f}  "
              f"mask {sm['process_masking_rate']['mean']:.3f}  "
              f"FAR {sm['false_alarm_rate']['mean']:.3f}  "
              f"detF1 {sm['detection_f1']['mean']:.3f}")

    per_scenario = {}
    banner("Per-scenario root-cause accuracy (mean over seeds)")
    print(f"{'system':22s}" + "".join(f"{s[:12]:>13s}" for s in SCENARIOS))
    for name, rows in results.items():
        per_scenario[name] = {}
        line = f"{name:22s}"
        for sc in SCENARIOS:
            v = [r["root_cause_accuracy"] for r in rows if r["scenario"] == sc]
            mean, lo, hi = bootstrap_ci(v)
            per_scenario[name][sc] = {"mean": mean, "lo": lo, "hi": hi}
            line += f"{mean:13.3f}"
        print(line)

    det_summary = {name: summarise(rows, ["detection_f1", "detection_recall",
                                          "false_alarm_rate"])
                   for name, rows in det_rows.items()}
    banner("Detection-only baselines (anomalous vs not)")
    for name, sm in det_summary.items():
        print(f"{name:22s} detF1 {sm['detection_f1']['mean']:.3f}  "
              f"recall {sm['detection_recall']['mean']:.3f}  "
              f"FAR {sm['false_alarm_rate']['mean']:.3f}")
    det_per_scenario = {name: {sc: float(np.nanmean([r["detection_recall"] for r in rows
                                                     if r["scenario"] == sc]))
                               for sc in SCENARIOS if sc != "clean"}
                        for name, rows in det_rows.items()}

    banner("Paired Wilcoxon, PIRCAD-rf vs each system (Holm-corrected)")
    ref = results["PIRCAD-rf"]
    tests = {}
    for metric in ("macro_f1", "root_cause_accuracy"):
        raw = {}
        for name, rows in results.items():
            if name == "PIRCAD-rf":
                continue
            raw[name] = wilcoxon([r[metric] for r in ref], [r[metric] for r in rows])
        adj = holm({k: v["p"] for k, v in raw.items()})
        for k in raw:
            raw[k]["p_holm"] = adj[k]
            print(f"  {metric:20s} vs {k:22s} median diff {raw[k]['median_diff']:+.3f} "
                  f"r_rb {raw[k]['r_rb']:+.2f}  p_holm {adj[k]:.2e}")
        tests[metric] = raw

    cms = {}
    eval_labels = [ep.labels for ep in eval_eps]
    for name, rows in results.items():
        cm = np.zeros((4, 4), dtype=int)
        for lab, r in zip(eval_labels, rows):
            cm += confusion(lab[WARMUP:], r["_pred"].astype(int)[WARMUP:])
        cms[name] = cm.tolist()

    save_json("exp6_statistical_benchmark.json", {
        "n_episodes": len(specs), "seeds": list(seeds),
        "summary": summary, "per_scenario": per_scenario,
        "detection_summary": det_summary,
        "detection_per_scenario_recall": det_per_scenario,
        "tests": tests, "confusion": cms,
        "rows": {k: strip_private(v) for k, v in results.items()},
    })



if __name__ == "__main__":
    main()
