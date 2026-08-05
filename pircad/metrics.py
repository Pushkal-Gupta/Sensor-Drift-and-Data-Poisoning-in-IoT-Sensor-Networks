"""
Evaluation metrics.

Generation 1 was scored on accuracy / precision / recall of an anomaly label.
Those numbers cannot express what this system is for, so the suite is extended
with the quantities the differential-response architecture actually changes:

    root-cause accuracy        did we identify *why*, not just *that*
    false-drift rate           poisoning mistaken for drift -- the dangerous one,
                               because it triggers recalibration on attacker data
    false-poison rate          drift mistaken for poisoning -- costly, because it
                               quarantines a usable sensor
    process-masking rate       a genuine process excursion mistaken for a sensor
                               fault -- the safety-critical error
    retraining protection      fraction of poisoned samples kept out of every
                               adaptive element
    poison containment         fraction of poisoned samples not forwarded
    detection delay            samples from fault onset to correct classification
    bandwidth saved, conditional-activation rate, latency, throughput
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from .fognode import RunResult
from .simulator import Episode
from .types import Action, RootCause

CAUSE_ORDER = [RootCause.NORMAL, RootCause.SENSOR_DRIFT,
               RootCause.DATA_POISONING, RootCause.PROCESS_EVENT]


def confusion(y_true: np.ndarray, y_pred: np.ndarray,
              classes: Sequence[RootCause] = CAUSE_ORDER) -> np.ndarray:
    idx = {int(c): i for i, c in enumerate(classes)}
    m = np.zeros((len(classes), len(classes)), dtype=int)
    for t, p in zip(y_true, y_pred):
        ti, pi = idx.get(int(t)), idx.get(int(p))
        if ti is None or pi is None:
            continue
        m[ti, pi] += 1
    return m


def per_class_recall(cm: np.ndarray) -> np.ndarray:
    """NaN (not 0) for classes absent from the episode, so that averaging over a
    mixed bank does not silently dilute a class that was never present."""
    denom = cm.sum(axis=1)
    return np.divide(np.diag(cm), denom, out=np.full(len(cm), np.nan),
                     where=denom > 0)


def per_class_precision(cm: np.ndarray) -> np.ndarray:
    denom = cm.sum(axis=0)
    return np.divide(np.diag(cm), denom, out=np.full(len(cm), np.nan),
                     where=denom > 0)


def f1_from_cm(cm: np.ndarray) -> np.ndarray:
    p, r = per_class_precision(cm), per_class_recall(cm)
    s = p + r
    ok = np.isfinite(s) & (s > 0)
    return np.divide(2 * p * r, s, out=np.full(len(cm), np.nan), where=ok)


@dataclass
class EvalRow:
    system: str
    scenario: str
    seed: int
    values: Dict[str, float] = field(default_factory=dict)


def detection_delay(y_true: np.ndarray, y_pred: np.ndarray, onset: int,
                    settle: int = 25) -> float:
    """Samples from onset until the correct cause has been asserted `settle` times."""
    target = int(y_true[min(len(y_true) - 1, onset + 5)])
    if target == int(RootCause.NORMAL):
        return float("nan")
    hits = 0
    for k in range(onset, len(y_pred)):
        if int(y_pred[k]) == target:
            hits += 1
            if hits >= settle:
                return float(k - onset)
        else:
            hits = 0
    return float("inf")


def evaluate_run(ep: Episode, res: RunResult, warmup: int,
                 system: str = "PIRCAD") -> Dict[str, float]:
    """Score one episode.  Everything before the warmup boundary is excluded."""
    y_true = np.asarray(ep.labels, dtype=int)[warmup:]
    y_pred = res.causes()[warmup:]
    n = len(y_true)
    onset = max(0, int(ep.meta.get("onset", 0)) - warmup)

    cm = confusion(y_true, y_pred)
    rec, prec, f1 = per_class_recall(cm), per_class_precision(cm), f1_from_cm(cm)

    is_poison = y_true == int(RootCause.DATA_POISONING)
    is_drift = y_true == int(RootCause.SENSOR_DRIFT)
    is_proc = y_true == int(RootCause.PROCESS_EVENT)
    is_normal = y_true == int(RootCause.NORMAL)
    anom_true = ~is_normal
    anom_pred = y_pred != int(RootCause.NORMAL)

    out: Dict[str, float] = {
        "root_cause_accuracy": float((y_true == y_pred).mean()) if n else float("nan"),
        "detection_accuracy": float((anom_true == anom_pred).mean()) if n else float("nan"),
        "normal_recall": float(rec[0]),
        "drift_recall": float(rec[1]),
        "poison_recall": float(rec[2]),
        "process_recall": float(rec[3]),
        "drift_precision": float(prec[1]),
        "poison_precision": float(prec[2]),
        "drift_f1": float(f1[1]),
        "poison_f1": float(f1[2]),
        # the two directional confusions that matter operationally
        "false_drift_rate": float((y_pred[is_poison] == int(RootCause.SENSOR_DRIFT)).mean())
        if is_poison.any() else float("nan"),
        "false_poison_rate": float((y_pred[is_drift] == int(RootCause.DATA_POISONING)).mean())
        if is_drift.any() else float("nan"),
        "process_masking_rate": float(
            np.isin(y_pred[is_proc], [int(RootCause.SENSOR_DRIFT),
                                      int(RootCause.DATA_POISONING)]).mean())
        if is_proc.any() else float("nan"),
        "false_alarm_rate": float(anom_pred[is_normal].mean()) if is_normal.any() else float("nan"),
        "detection_delay": detection_delay(y_true, y_pred, onset),
        "n_samples": float(n),
    }

    # ---- action-level effects (PIRCAD only reports these meaningfully) ----
    actions = np.asarray(res.frame_action, dtype=int)[warmup:]
    fwd = np.asarray(res.forwarded, dtype=bool)[warmup:]
    if is_poison.any():
        out["poison_containment"] = float(1.0 - fwd[is_poison].mean())
        out["poison_recalibrated"] = float(
            (actions[is_poison] == int(Action.RECALIBRATE_AND_FORWARD)).mean())
    if is_drift.any():
        out["drift_kept_flowing"] = float(fwd[is_drift].mean())
        out["drift_quarantined"] = float(
            (actions[is_drift] == int(Action.QUARANTINE)).mean())
    if is_proc.any():
        out["process_forwarded"] = float(fwd[is_proc].mean())
        out["process_recalibrated"] = float(
            (actions[is_proc] == int(Action.RECALIBRATE_AND_FORWARD)).mean())

    out["bandwidth_saved"] = float(1.0 - fwd.mean()) if n else 0.0
    # invocations are per channel per sample, so normalise by both
    n_ch = max(1, len(res.channels))
    out["conditional_activation_rate"] = (
        float(res.classifier_invocations) / max(1, res.n_samples * n_ch))
    out["classifier_calls_per_sample"] = (
        float(res.classifier_invocations) / max(1, res.n_samples))
    out["avg_latency_s"] = float(res.avg_latency_s)
    out["runtime_s"] = float(res.runtime_s)
    out["escalations"] = float(res.escalation.get("escalations_submitted", 0.0))

    # Only meaningful for a system that actually has an adaptation gate; a
    # detector with no governor is reported as NaN rather than as perfect.
    if res.has_governor:
        # Count admissions on the channel the poisoning was actually injected
        # into.  Summing over channels would compare a per-channel counter with
        # a per-frame ground truth and can exceed 1 (or go negative after the
        # 1 - x), which is meaningless.
        if ep.target_channel and ep.target_channel in res.poisoned_admitted_by_channel:
            poisoned_admitted = res.poisoned_admitted_by_channel[ep.target_channel]
        else:
            poisoned_admitted = res.poisoned_admitted_to_training
        n_poison = int(is_poison.sum())
        out["poisoned_samples_admitted"] = float(poisoned_admitted)
        out["retraining_protection"] = (
            float(1.0 - poisoned_admitted / n_poison) if n_poison else float("nan"))
        out["samples_admitted_for_training"] = float(res.admitted_to_training)
    return out


def evaluate_labels(ep: Episode, y_pred: np.ndarray, warmup: int,
                    runtime_s: float = float("nan"),
                    invocations: int = 0) -> Dict[str, float]:
    """Score a bare label sequence (used for the Gen-1 comparison).

    The comparison system filters anything it flags, so `forwarded` is taken to
    be "predicted normal" -- the most favourable reading of its bandwidth effect.
    """
    from .fognode import RunResult as _RR
    res = _RR(channels=[])
    res.frame_cause = list(np.asarray(y_pred, dtype=int))
    res.frame_action = [int(Action.FORWARD)] * len(y_pred)
    res.forwarded = [int(c) == int(RootCause.NORMAL) for c in y_pred]
    res.n_samples = len(y_pred)
    res.runtime_s = runtime_s
    res.classifier_invocations = invocations
    res.has_governor = False
    return evaluate_run(ep, res, warmup, system="Gen1")


def aggregate(rows: List[Dict[str, float]]) -> Dict[str, float]:
    keys = sorted({k for r in rows for k in r})
    out = {}
    for k in keys:
        vals = np.asarray([r[k] for r in rows if k in r and np.isfinite(r[k])],
                          dtype=float)
        out[k] = float(vals.mean()) if vals.size else float("nan")
        out[k + "_std"] = float(vals.std()) if vals.size else float("nan")
    return out


def format_table(rows: Dict[str, Dict[str, float]], keys: Sequence[str],
                 title: str = "", width: int = 26) -> str:
    """Small text table: rows = {system: metrics}."""
    systems = list(rows)
    lines = []
    if title:
        lines += [title, "=" * len(title)]
    header = "metric".ljust(width) + "".join(s.rjust(18) for s in systems)
    lines += [header, "-" * len(header)]
    for k in keys:
        cells = []
        for s in systems:
            v = rows[s].get(k, float("nan"))
            cells.append(("--" if not np.isfinite(v) else
                          (f"{v:.3e}" if 0 < abs(v) < 1e-3 else f"{v:.3f}")).rjust(18))
        lines.append(k.ljust(width) + "".join(cells))
    return "\n".join(lines)
