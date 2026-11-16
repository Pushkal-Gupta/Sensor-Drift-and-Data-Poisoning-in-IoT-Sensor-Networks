"""
Experiment 15 -- post-hoc statistics on existing per-episode artifacts.

No episode is re-run.  Added in response to the first-round review:

  polarity    per-polarity breakdown of the 180-episode held-out bank (exp. 6):
              does any system still behave differently on rising and falling
              events (the "polarity shortcut" of Section 7.2)?
  attacker    paired Wilcoxon tests for the attacker sweep (exp. 7), matched
              by seed, PIRCAD vs. each learned baseline, Holm-corrected.
  shift       95 % CIs and paired tests for the distribution-shift table
              (exp. 8), PIRCAD vs. each learned baseline per condition.
  intel_shift CIs for the Intel noise shift (exp. 13), and a paired test of
              whether PIRCAD's drop in macro-F1 differs from each baseline's.

Run:  python experiments/exp15_posthoc.py
"""

from __future__ import annotations

import json

import numpy as np
from scipy.stats import mannwhitneyu

from v2_common import ARTIFACTS, bootstrap_ci, holm, save_json, wilcoxon


def load(n):
    return json.load(open(ARTIFACTS / n))


def polarity():
    d = load("exp6_statistical_benchmark.json")
    out = {}
    for sysname in ("PIRCAD-rf", "PIRCAD-hgb", "Multi-scale 1D-CNN", "Generic-feature RF"):
        rows = [r for r in d["rows"][sysname] if r["scenario"] != "clean"]
        for r in rows:
            r["_sign"] = 1 if ((r["seed"] // 7919) - 900) % 2 == 0 else -1
        res = {}
        for key in ("macro_f1", "false_drift_rate", "false_poison_rate",
                    "process_masking_rate", "drift_recall"):
            up = [r[key] for r in rows if r["_sign"] == 1 and np.isfinite(r.get(key, np.nan))]
            dn = [r[key] for r in rows if r["_sign"] == -1 and np.isfinite(r.get(key, np.nan))]
            p = float(mannwhitneyu(up, dn).pvalue) if len(up) and len(dn) and \
                not np.allclose(np.r_[up, dn], np.r_[up, dn][0]) else 1.0
            res[key] = {"rising": bootstrap_ci(up), "falling": bootstrap_ci(dn), "p": p,
                        "n": [len(up), len(dn)]}
        out[sysname] = res
    return out


def attacker():
    d = load("exp7_adaptive_attacker.json")
    out, raw = {}, {}
    for c in d["conditions"]:
        R = d["results"][c]
        key = "drift_recall" if c.startswith("drift") else "poison_recall"
        by = {s: {r["seed"]: r for r in R[s]["rows"]} for s in R}
        seeds = sorted(by["PIRCAD"])
        out[c] = {}
        for b in ("Multi-scale 1D-CNN", "Generic-feature RF"):
            for k in (key, "useful_frac"):
                t = wilcoxon([by["PIRCAD"][s].get(k, np.nan) for s in seeds],
                             [by[b][s].get(k, np.nan) for s in seeds])
                out[c][f"{b}|{k}"] = t
                raw[f"{c}|{b}|{k}"] = t["p"]
    adj = holm(raw)
    for name, p in adj.items():
        c, b, k = name.split("|")
        out[c][f"{b}|{k}"]["p_holm"] = p
    return out


def shift():
    d = load("exp8_distribution_shift.json")
    out, raw = {}, {}
    for c in d["conditions"]:
        R = d["results"][c]
        by = {s: {(r["seed"], r["scenario"]): r for r in R[s]["rows"]} for s in R}
        keys = sorted(by["PIRCAD"])
        out[c] = {s: {"macro_f1": bootstrap_ci([r["macro_f1"] for r in R[s]["rows"]])}
                  for s in R}
        for b in ("Multi-scale 1D-CNN", "Generic-feature RF"):
            if b not in by:
                continue
            t = wilcoxon([by["PIRCAD"][k]["macro_f1"] for k in keys],
                         [by[b][k]["macro_f1"] for k in keys])
            out[c][b]["test"] = t
            raw[f"{c}|{b}"] = t["p"]
    for name, p in holm(raw).items():
        c, b = name.split("|")
        out[c][b]["test"]["p_holm"] = p
    return out


def intel_shift():
    d = load("exp13_intel_shift.json")
    L = d["levels"]
    out = {"levels": L, "ci": {}, "drop_tests": {}}
    names = ("PIRCAD-rf", "Multi-scale 1D-CNN", "Generic-feature RF")
    for lv in L:
        out["ci"][lv] = {n: {k: bootstrap_ci([r[k] for r in d["results"][lv][n]["rows"]])
                             for k in ("macro_f1", "false_alarm_rate")} for n in names}
    by = {lv: {n: {r["seed"]: r["macro_f1"] for r in d["results"][lv][n]["rows"]}
               for n in names} for lv in L}
    seeds = sorted(by[L[0]]["PIRCAD-rf"])
    drop = {n: np.array([by[L[0]][n][s] - by[L[-1]][n][s] for s in seeds]) for n in names}
    raw = {}
    for n in names:
        t = wilcoxon(drop[n], np.zeros_like(drop[n]))     # is the drop non-zero?
        t["drop"] = bootstrap_ci(drop[n])
        out["drop_tests"][n] = t
        raw[n] = t["p"]
    for b in names[1:]:                                  # PIRCAD's drop vs. b's drop
        t = wilcoxon(drop["PIRCAD-rf"], drop[b])
        out["drop_tests"][f"PIRCAD-rf vs {b}"] = t
        raw[f"PIRCAD-rf vs {b}"] = t["p"]
    for k, p in holm(raw).items():
        out["drop_tests"][k]["p_holm"] = p
    return out


def main():
    res = {"polarity": polarity(), "attacker": attacker(), "shift": shift(),
           "intel_shift": intel_shift()}
    for s, r in res["polarity"].items():
        m = r["macro_f1"]
        print(f"{s:22s} mF1 rising {m['rising'][0]:.3f} falling {m['falling'][0]:.3f} p {m['p']:.3f} | "
              f"fD {r['false_drift_rate']['rising'][0]:.3f}/{r['false_drift_rate']['falling'][0]:.3f} "
              f"fP {r['false_poison_rate']['rising'][0]:.3f}/{r['false_poison_rate']['falling'][0]:.3f}")
    for c, r in res["attacker"].items():
        print(c, {k: f"{v['r_rb']:+.2f} p{v['p_holm']:.1e}" for k, v in r.items()})
    for c, r in res["shift"].items():
        print(c, {k: (f"{v['macro_f1'][0]:.3f} [{v['macro_f1'][1]:.3f},{v['macro_f1'][2]:.3f}]"
                      + (f" p{v['test']['p_holm']:.1e} r{v['test']['r_rb']:+.2f}" if 'test' in v else ""))
                  for k, v in r.items()})
    for k, v in res["intel_shift"]["drop_tests"].items():
        print(k, f"p_holm {v['p_holm']:.2e} r {v['r_rb']:+.2f}", v.get("drop", ""))
    save_json("exp15_posthoc.json", res)


if __name__ == "__main__":
    main()
