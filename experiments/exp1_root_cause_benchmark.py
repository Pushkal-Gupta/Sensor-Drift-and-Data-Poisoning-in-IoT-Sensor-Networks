"""
Experiment 1 -- root-cause benchmark: Gen-1 FHAD vs Gen-2 PIRCAD.

The decisive condition is `stealth_poison`: a false-data injection whose
magnitude AND growth rate are matched to benign drift by construction.  Any
detector that reasons about the size or the persistence of a deviation is at
chance there; separating the two requires evidence about *how* the deviation was
produced, which is what the physics residual supplies.

Run:  python experiments/exp1_root_cause_benchmark.py
"""

from __future__ import annotations

import time
from collections import defaultdict
from typing import Dict, List

import numpy as np

from _common import (EVAL_SEEDS, SCENARIOS, WARMUP, banner, eval_episodes,
                     get_models, print_confusion, save_json)

from pircad.fognode import PIRCADFogNode
from pircad.metrics import (aggregate, confusion, evaluate_labels, evaluate_run,
                            format_table)
from pircad.rootcause import HybridRootCauseClassifier
from pircad.simulator import frames
from pircad.types import RootCause

HEADLINE = [
    "root_cause_accuracy", "detection_accuracy", "drift_recall", "poison_recall",
    "process_recall", "false_drift_rate", "false_poison_rate",
    "process_masking_rate", "false_alarm_rate", "detection_delay",
    "poison_containment", "drift_kept_flowing", "retraining_protection",
    "bandwidth_saved", "conditional_activation_rate", "avg_latency_s",
]


def main() -> None:
    models = get_models()
    clf = HybridRootCauseClassifier(models["root_cause"])
    gen1 = models["gen1"]

    episodes = eval_episodes()
    rows_p: Dict[str, List[dict]] = defaultdict(list)
    rows_g: Dict[str, List[dict]] = defaultdict(list)
    cm_p = np.zeros((4, 4), dtype=int)
    cm_g = np.zeros((4, 4), dtype=int)

    banner("EXPERIMENT 1 - root-cause benchmark (held-out seeds)")
    print(f"episodes={len(episodes)}  seeds={EVAL_SEEDS}  warmup={WARMUP} samples\n")

    for ep in episodes:
        node = PIRCADFogNode(classifier=clf)
        res = node.run(frames(ep), truth_labels=ep.labels)
        m = evaluate_run(ep, res, WARMUP)
        rows_p[ep.scenario].append(m)
        cm_p += confusion(ep.labels[WARMUP:], res.causes()[WARMUP:])

        t0 = time.perf_counter()
        y1 = gen1.run(ep)
        t1 = time.perf_counter()
        rows_g[ep.scenario].append(
            evaluate_labels(ep, y1, WARMUP, runtime_s=t1 - t0,
                            invocations=gen1.invocations))
        cm_g += confusion(ep.labels[WARMUP:], np.asarray(y1)[WARMUP:])

        print(f"  {ep.scenario:16s} seed={int(ep.meta['seed']):6d} "
              f"PIRCAD rc={m['root_cause_accuracy']:.3f}  "
              f"Gen1 rc={rows_g[ep.scenario][-1]['root_cause_accuracy']:.3f}")

    # ---------------- per-scenario ----------------
    banner("Per-scenario root-cause accuracy")
    print(f"{'scenario':18s}{'PIRCAD':>12s}{'Gen-1 FHAD':>14s}{'delta':>10s}")
    print("-" * 54)
    per_scenario = {}
    for sc in SCENARIOS:
        a = aggregate(rows_p[sc])["root_cause_accuracy"]
        b = aggregate(rows_g[sc])["root_cause_accuracy"]
        per_scenario[sc] = {"pircad": a, "gen1": b}
        print(f"{sc:18s}{a:12.3f}{b:14.3f}{a - b:+10.3f}")

    # ---------------- overall ----------------
    all_p = aggregate([r for rs in rows_p.values() for r in rs])
    all_g = aggregate([r for rs in rows_g.values() for r in rs])
    banner("Overall (mean over all held-out episodes)")
    print(format_table({"PIRCAD (Gen 2)": all_p, "FHAD (Gen 1)": all_g}, HEADLINE))

    labels = [c.name for c in (RootCause.NORMAL, RootCause.SENSOR_DRIFT,
                               RootCause.DATA_POISONING, RootCause.PROCESS_EVENT)]
    print_confusion(cm_p, labels, "\nPIRCAD confusion (pooled)")
    print_confusion(cm_g, labels, "\nGen-1 FHAD confusion (pooled)")

    # ---------------- the decisive cell ----------------
    banner("The decisive comparison: rate-matched injection vs benign drift")
    sub_p = aggregate(rows_p["stealth_poison"] + rows_p["drift"])
    sub_g = aggregate(rows_g["stealth_poison"] + rows_g["drift"])
    print(format_table({"PIRCAD (Gen 2)": sub_p, "FHAD (Gen 1)": sub_g},
                       ["root_cause_accuracy", "drift_recall", "poison_recall",
                        "false_drift_rate", "false_poison_rate",
                        "poison_containment", "retraining_protection"]))
    print("\nfalse_drift_rate is the operationally dangerous error: an injection\n"
          "classified as drift is answered with recalibration, i.e. the system\n"
          "absorbs the attacker's bias into its own reference.")

    save_json("exp1_root_cause_benchmark.json", {
        "per_scenario": per_scenario,
        "overall": {"pircad": all_p, "gen1": all_g},
        "decisive": {"pircad": sub_p, "gen1": sub_g},
        "confusion_pircad": cm_p.tolist(),
        "confusion_gen1": cm_g.tolist(),
        "labels": labels,
    })


if __name__ == "__main__":
    main()
