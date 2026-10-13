"""Load experiment artifacts and turn them into manuscript tables and numbers.

Nothing in the results section is typed by hand: every figure quoted comes from
artifacts/exp{6..11}_*.json through the helpers below.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"


def load(name: str) -> Optional[dict]:
    p = ART / name
    return json.loads(p.read_text()) if p.exists() else None


def t3(v) -> tuple:
    if isinstance(v, dict):
        return v["mean"], v["lo"], v["hi"]
    return tuple(v)


def f(v, nd=3) -> str:
    """mean only."""
    m = t3(v)[0] if not isinstance(v, (int, float)) else v
    return "–" if m is None or not np.isfinite(m) else f"{m:.{nd}f}"


def fci(v, nd=3) -> str:
    m, lo, hi = t3(v)
    if not np.isfinite(m):
        return "–"
    return f"{m:.{nd}f} [{lo:.{nd}f}, {hi:.{nd}f}]"


def fsigned(v, nd=2) -> str:
    m, lo, hi = t3(v)
    return f"{m:+.{nd}f} [{lo:+.{nd}f}, {hi:+.{nd}f}]"


def pfmt(p: float) -> str:
    if p < 1e-4:
        return "<10^{−4}"
    if p < 0.001:
        return f"{p:.1e}".replace("e-0", "×10^{−").replace("e-", "×10^{−") + "}"
    return f"{p:.3f}"


SYS6 = ["PIRCAD-rf", "PIRCAD-hgb", "PIRCAD-mlp", "PIRCAD-logreg", "PIRCAD-rules",
        "Multi-scale 1D-CNN", "Generic-feature RF", "Gen-1 FHAD"]
SYS6_LABEL = {"PIRCAD-rf": "PIRCAD (random forest)", "PIRCAD-hgb": "PIRCAD (gradient boosting)",
              "PIRCAD-mlp": "PIRCAD (MLP)", "PIRCAD-logreg": "PIRCAD (logistic regression)",
              "PIRCAD-rules": "PIRCAD (rules, no training)"}
SCEN = ["clean", "drift", "stealth_poison", "spike_poison", "replay_poison", "process_event"]
SCEN_LABEL = {"clean": "Clean", "drift": "Benign drift", "stealth_poison": "Rate-matched poisoning",
              "spike_poison": "Spike poisoning", "replay_poison": "Replay poisoning",
              "process_event": "Process event"}


def bench_table(d) -> List[List[str]]:
    S, T = d["summary"], d["tests"]["macro_f1"]
    rows = []
    for s in SYS6:
        sm = S[s]
        p = "ref." if s == "PIRCAD-rf" else pfmt(T[s]["p_holm"])
        rows.append([SYS6_LABEL.get(s, s), fci(sm["macro_f1"]), fci(sm["root_cause_accuracy"]),
                     f(sm["false_drift_rate"]), f(sm["false_poison_rate"]),
                     f(sm["process_masking_rate"]), f(sm["false_alarm_rate"]), p])
    return rows


def scenario_table(d, systems, key="per_scenario") -> List[List[str]]:
    ps = d[key]
    rows = []
    for sc in SCEN:
        rows.append([SCEN_LABEL[sc]] + [f(ps[s][sc]) for s in systems])
    return rows


def detection_table(d) -> List[List[str]]:
    rows = []
    for s in ("PIRCAD-rf", "Multi-scale 1D-CNN"):
        sm = d["summary"][s]
        rows.append([SYS6_LABEL.get(s, s), f(sm["detection_f1"]), "–", f(sm["false_alarm_rate"])])
    for name, sm in d["detection_summary"].items():
        rows.append([name, f(sm["detection_f1"]), f(sm["detection_recall"]),
                     f(sm["false_alarm_rate"])])
    return rows


def det_recall_by_scenario(d):
    return d["detection_per_scenario_recall"]


ATT_LABEL = {
    "drift (reference)": "Benign drift (reference)",
    "stale, white (default)": "A0 stale surrogate",
    "stale, matched noise": "A1 noise-matched",
    "fidelity 0.05": "A2 fidelity 0.05",
    "fidelity 0.10": "A2 fidelity 0.10",
    "fidelity 0.30": "A2 fidelity 0.30",
    "exact state, white": "A3 exact state",
    "exact state, matched": "A3 exact state + matched noise",
    "on-path (MITM) ramp": "A4 on-path (MITM)",
    "coordinated, stale": "A5 coordinated, stale",
    "coordinated, exact": "A5 coordinated, exact",
}
ATT_SYS = ("PIRCAD", "Multi-scale 1D-CNN", "Generic-feature RF")


def attacker_table(d) -> List[List[str]]:
    R = d["results"]
    rows = []
    for c in d["conditions"]:
        r = R[c]
        key = "drift_recall" if c.startswith("drift") else "poison_recall"
        row = [ATT_LABEL.get(c, c)]
        row += [f(r[s]["summary"][key]) for s in ATT_SYS]
        row.append(f(r["PIRCAD"]["summary"]["false_drift_rate"]) if key == "poison_recall" else "–")
        row += [f(r[s]["summary"]["useful_frac"], 2) for s in ATT_SYS]
        row.append(f(r["PIRCAD"]["summary"]["fwd_err_mean"], 2))
        rows.append(row)
    return rows


SHIFT_LABEL = {"nominal": "Nominal", "twin mismatch x3": "Process-model mismatch ×3",
               "twin mismatch x6": "Process-model mismatch ×6",
               "plant dynamics -15%": "Plant dynamics −15 %",
               "plant dynamics +20%": "Plant dynamics +20 %", "noise x2": "Sensor noise ×2",
               "noise x3": "Sensor noise ×3", "ambient swing x2.5": "Ambient swing ×2.5",
               "setpoint 35 C": "Set-point 35 °C", "slow rates": "Rates 0.3–0.8 ×10^{−3}",
               "fast rates": "Rates 2.5–2.9 ×10^{−3}"}


def shift_table(d) -> List[List[str]]:
    R = d["results"]
    rows = []
    for c in d["conditions"]:
        r = R[c]
        row = [SHIFT_LABEL.get(c, c)]
        for s in ("PIRCAD", "Multi-scale 1D-CNN", "Generic-feature RF"):
            row.append(f(r[s]["summary"]["macro_f1"]) if s in r else "n/a")
        row.append(f(r["PIRCAD"]["summary"]["false_drift_rate"]))
        row.append(f(r["PIRCAD"]["summary"]["false_alarm_rate"]))
        rows.append(row)
    return rows


def safety_table(d) -> List[List[str]]:
    S, D = d["summary"], d["drift_summary"]
    rows = []
    for s in S:
        sm = S[s]
        rows.append([s, f(sm["alarm_fired"], 2), f(sm["spurious_trip"], 2),
                     f(sm["premature_trip"], 2), fsigned(sm["error_at_true_cross"]),
                     f(sm["poison_forwarded"], 3), f(D[s]["drift_fwd_error"], 2)])
    return rows


def intel_table(d) -> List[List[str]]:
    S, T = d["summary"], d["tests"]
    rows = []
    for s in ("PIRCAD-rf", "PIRCAD-hgb", "PIRCAD-rules", "Multi-scale 1D-CNN",
              "Generic-feature RF", "Gen-1 FHAD"):
        if s not in S:
            continue
        sm = S[s]
        p = "ref." if s == "PIRCAD-rf" else pfmt(T[s]["p_holm"])
        rows.append([SYS6_LABEL.get(s, s), fci(sm["macro_f1"]), f(sm["drift_recall"]),
                     f(sm["poison_recall"]), f(sm["process_recall"]),
                     f(sm["false_drift_rate"]), f(sm["false_alarm_rate"]), f(sm["harm"], 2), p])
    return rows


def ablation_table(d) -> List[List[str]]:
    S, T = d["summary"], d["tests"]
    rows = []
    for r, sm in S.items():
        p = "ref." if r == "full" else pfmt(T[r]["p_holm"])
        rows.append(["Full system" if r == "full" else r, fci(sm["macro_f1"]),
                     f(sm["false_drift_rate"]), f(sm["poison_containment"]),
                     f(sm["process_masking_rate"]), f(sm["retraining_protection"]),
                     f(sm["harm"], 2), p])
    return rows


def latency(d) -> Dict[str, float]:
    """Per-frame latency of PIRCAD-rf from the experiment-6 rows (ms)."""
    rows = d["rows"]["PIRCAD-rf"]
    lat = np.array([r["avg_latency_s"] for r in rows]) * 1e3
    act = np.array([r["conditional_activation_rate"] for r in rows])
    return {"median_ms": float(np.median(lat)), "p95_ms": float(np.quantile(lat, 0.95)),
            "activation": float(np.mean(act))}


def per_scenario_metric(rows: List[dict], scenario: str, key: str) -> tuple:
    """Bootstrap mean [CI] of `key` over the episodes of one scenario."""
    x = np.array([r.get(key, np.nan) for r in rows if r.get("scenario") == scenario], float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return (float("nan"),) * 3
    rng = np.random.default_rng(0)
    bs = rng.choice(x, size=(4000, x.size)).mean(1)
    return float(x.mean()), float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))
