"""Figures for the journal evaluation (experiments 6-11).

Reads artifacts/exp{6..11}_*.json and writes figures/v2_*.png (300 dpi) and
.pdf.  Every point carries its 95% bootstrap CI.  PIRCAD is always blue; the
data-driven baselines take the next categorical slots and also differ by marker
shape, so identity never rests on colour alone.

Run:  python experiments/make_figures_v2.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ART, FIG = ROOT / "artifacts", ROOT / "figures"
FIG.mkdir(exist_ok=True)

INK, INK2, GRID, NEUTRAL = "#0b0b0b", "#52514e", "#e5e4df", "#a8a69f"
STYLE = {   # system -> (colour, marker)
    "PIRCAD": ("#2a78d6", "o"),
    "Multi-scale 1D-CNN": ("#eb6834", "s"),
    "Generic-feature RF": ("#1baf7a", "D"),
    "Gen-1 FHAD": ("#eda100", "^"),
}
SCEN_LABEL = {"clean": "Clean", "drift": "Benign drift",
              "stealth_poison": "Rate-matched\npoisoning", "spike_poison": "Spike\npoisoning",
              "replay_poison": "Replay\npoisoning", "process_event": "Process\nevent"}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": INK2,
    "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "legend.frameon": False, "figure.dpi": 150, "savefig.dpi": 300,
    "savefig.bbox": "tight", "lines.linewidth": 2,
})


def load(name):
    p = ART / name
    return json.loads(p.read_text()) if p.exists() else None


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"{name}.{ext}")
    plt.close(fig)
    print(f"[fig] {name}")


def ci(v):
    """Accept {'mean','lo','hi'} or [mean, lo, hi]."""
    if isinstance(v, dict):
        return v["mean"], v["lo"], v["hi"]
    return v[0], v[1], v[2]


def dotplot(ax, groups, systems, getter, ylabel, ylim=None, dodge=0.18):
    x = np.arange(len(groups))
    k = len(systems)
    for i, s in enumerate(systems):
        col, mk = STYLE.get(s, (NEUTRAL, "o"))
        vals = [getter(s, g) for g in groups]
        m = np.array([v[0] if v else np.nan for v in vals])
        lo = np.array([v[1] if v else np.nan for v in vals])
        hi = np.array([v[2] if v else np.nan for v in vals])
        xs = x + (i - (k - 1) / 2) * dodge
        ax.errorbar(xs, m, yerr=[m - lo, hi - m], fmt=mk, color=col, ms=6,
                    elinewidth=1.4, capsize=0, label=s,
                    markeredgecolor="white", markeredgewidth=0.8, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(groups, fontsize=8)
    ax.set_ylabel(ylabel)
    if ylim:
        ax.set_ylim(*ylim)
    ax.grid(axis="x", visible=False)


# --------------------------------------------------------------------------
def fig_benchmark():
    d = load("exp6_statistical_benchmark.json")
    if not d:
        return
    ps = d["per_scenario"]
    systems = ["PIRCAD", "Multi-scale 1D-CNN", "Generic-feature RF", "Gen-1 FHAD"]
    key = {"PIRCAD": "PIRCAD-rf"}
    scen = list(SCEN_LABEL)
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    dotplot(ax, [SCEN_LABEL[s] for s in scen], systems,
            lambda s, g: ci(ps[key.get(s, s)][scen[[SCEN_LABEL[x] for x in scen].index(g)]]),
            "Root-cause accuracy", (0, 1.04))
    ax.legend(ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=8)
    save(fig, "v2_fig_benchmark")

    # confusion matrices, PIRCAD vs the strongest learned baseline
    labels = ["Normal", "Drift", "Poisoning", "Process"]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.1))
    for ax, (name, title) in zip(axes, (("PIRCAD-rf", "(a) PIRCAD"),
                                        ("Multi-scale 1D-CNN", "(b) Multi-scale 1D-CNN"))):
        cm = np.asarray(d["confusion"][name], float)
        frac = cm / np.maximum(cm.sum(1, keepdims=True), 1)
        ax.imshow(frac, cmap="Blues", vmin=0, vmax=1)
        for i in range(4):
            for j in range(4):
                ax.text(j, i, f"{frac[i, j]:.2f}", ha="center", va="center", fontsize=8,
                        color="white" if frac[i, j] > 0.55 else INK)
        ax.set_xticks(range(4)); ax.set_xticklabels(labels, fontsize=8)
        ax.set_yticks(range(4)); ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel("Predicted cause"); ax.set_title(title, fontsize=9, loc="left")
        ax.grid(False)
    axes[0].set_ylabel("True cause")
    save(fig, "v2_fig_confusion")


def fig_attacker():
    d = load("exp7_adaptive_attacker.json")
    if not d:
        return
    conds = d["conditions"][1:]          # drop the drift reference
    short = ["A0\nstale", "A1\nmatched", "A2\n0.05", "A2\n0.10", "A2\n0.30",
             "A3\nexact", "A3 exact\n+matched", "A4\non-path", "A5\nstale",
             "A5\nexact"][:len(conds)]
    R = d["results"]
    systems = ["PIRCAD", "Multi-scale 1D-CNN", "Generic-feature RF"]
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 5.0), sharex=True)
    for ax, k, lab in ((axes[0], "poison_recall", "Poisoning recall"),
                       (axes[1], "useful_frac", "Useful-data rate")):
        dotplot(ax, short, systems,
                lambda s, g, k=k: ci(R[conds[short.index(g)]][s]["summary"][k]), lab)
    axes[0].set_ylim(-0.03, 1.05)
    axes[1].set_ylim(-0.03, 1.05)
    axes[0].legend(ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=8)
    axes[1].set_xlabel("Attacker capability (left to right: more knowledge of the live process)")
    save(fig, "v2_fig_attacker")


def fig_shift():
    d = load("exp8_distribution_shift.json")
    if not d:
        return
    conds = d["conditions"]; R = d["results"]
    systems = ["PIRCAD", "Multi-scale 1D-CNN", "Generic-feature RF"]
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    dotplot(ax, conds, systems,
            lambda s, g: ci(R[g][s]["summary"]["macro_f1"]) if s in R[g] else None,
            "Macro-F1 (no retraining)", (0, 1.04), dodge=0.22)
    short = {"nominal": "nominal", "twin mismatch x3": "model\nerr ×3",
             "twin mismatch x6": "model\nerr ×6", "plant dynamics -15%": "plant\n−15 %",
             "plant dynamics +20%": "plant\n+20 %", "noise x2": "noise\n×2",
             "noise x3": "noise\n×3", "ambient swing x2.5": "ambient\n×2.5",
             "setpoint 35 C": "set-point\n35 °C", "slow rates": "slow\nrates",
             "fast rates": "fast\nrates"}
    ax.set_xticklabels([short.get(c, c) for c in conds], fontsize=7.5)
    ax.legend(ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=8)
    save(fig, "v2_fig_shift")


def fig_safety():
    d = load("exp9_adaptation_poisoning.json")
    if not d:
        return
    S = d["summary"]; D = d["drift_summary"]
    systems = list(S)
    names = [s.replace(" recalibration", "\nrecalibration").replace(" estimator", "\nestimator")
             .replace(" non-differential", "\nnon-differential")
             .replace(" auto re-attribution", "\nauto re-attribution") for s in systems]
    cols = ["#2a78d6" if s == "PIRCAD" else NEUTRAL for s in systems]
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 3.1), sharey=True)
    y = np.arange(len(systems))[::-1]
    panels = ((S, "alarm_fired", "(a) Safety alarm fired", (0, 1.3)),
              (S, "error_at_true_cross", "(b) Error at true crossing (°C)", None),
              (D, "drift_fwd_error", "(c) Benign-drift correction error (°C)", None))
    for ax, (src, k, title, xl) in zip(axes, panels):
        m = np.array([src[s][k][0] for s in systems])
        lo = np.array([src[s][k][1] for s in systems]); hi = np.array([src[s][k][2] for s in systems])
        ax.barh(y, m, color=cols, height=0.6, zorder=2)
        ax.errorbar(m, y, xerr=[m - lo, hi - m], fmt="none", ecolor=INK2, elinewidth=1, zorder=3)
        span = max(hi.max(), 0) - min(lo.min(), 0)
        pad = 0.03 * span
        for yi, v, l, h in zip(y, m, lo, hi):
            txt = f"{v:+.2f}" if k != "alarm_fired" else f"{v:.2f}"
            if v >= 0:
                ax.text(h + pad, yi, txt, va="center", ha="left", fontsize=7.5, color=INK)
            else:
                ax.text(l - pad, yi, txt, va="center", ha="right", fontsize=7.5, color=INK)
        ax.axvline(0, color=INK2, lw=0.8)
        ax.set_title(title, fontsize=8.5, loc="left")
        ax.grid(axis="y", visible=False)
        lo_lim = min(lo.min(), 0) - (0.28 * span if lo.min() < 0 else 0)
        hi_lim = max(hi.max(), 0) + 0.28 * span
        ax.set_xlim(*(xl if xl else (lo_lim, hi_lim)))
    axes[0].set_yticks(y); axes[0].set_yticklabels(names, fontsize=7.5)
    save(fig, "v2_fig_safety")


def fig_intel():
    d = load("exp10_intel_lab.json")
    if not d:
        return
    ps = d["per_scenario"]
    systems = ["PIRCAD", "Multi-scale 1D-CNN", "Generic-feature RF", "Gen-1 FHAD"]
    key = {"PIRCAD": "PIRCAD-rf"}
    scen = list(SCEN_LABEL)
    labs = [SCEN_LABEL[s] for s in scen]
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    dotplot(ax, labs, systems,
            lambda s, g: ci(ps[key.get(s, s)][scen[labs.index(g)]]),
            "Root-cause accuracy", (0, 1.04))
    ax.legend(ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=8)
    save(fig, "v2_fig_intel")


def fig_ablation():
    d = load("exp11_ablation.json")
    if not d:
        return
    S = d["summary"]; rows = list(S)
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.3), sharey=True)
    y = np.arange(len(rows))[::-1]
    for ax, (k, title) in zip(axes, (("macro_f1", "(a) Macro-F1"),
                                     ("harm", "(b) Forwarded error on target (°C)"))):
        m = np.array([S[r][k][0] for r in rows]); lo = np.array([S[r][k][1] for r in rows])
        hi = np.array([S[r][k][2] for r in rows])
        cols = ["#2a78d6" if r == "full" else NEUTRAL for r in rows]
        ax.barh(y, m, color=cols, height=0.6, zorder=2)
        ax.errorbar(m, y, xerr=[m - lo, hi - m], fmt="none", ecolor=INK2, elinewidth=1, zorder=3)
        ax.axvline(m[0], color="#2a78d6", lw=0.8, ls=":")
        ax.set_title(title, fontsize=8.5, loc="left"); ax.grid(axis="y", visible=False)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(["Full system" if r == "full" else r for r in rows], fontsize=7.5)
    save(fig, "v2_fig_ablation")


if __name__ == "__main__":
    for f in (fig_benchmark, fig_attacker, fig_shift, fig_safety, fig_intel, fig_ablation):
        f()
