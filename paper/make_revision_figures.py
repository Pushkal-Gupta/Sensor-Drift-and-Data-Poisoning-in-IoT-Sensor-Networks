"""Figures added in the first-round revision.

    python paper/make_revision_figures.py

figures/rev_fig_tradeoff.png -- safety vs. calibration trade-off of the
self-calibrating baselines swept over their adaptation parameter (exp. 14c),
with PIRCAD at its default setting (exp. 9).
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ART, FIG = ROOT / "artifacts", ROOT / "figures"
COL = {"Kalman bias estimator": "#3b6ea8", "EWMA recalibration": "#c9822b",
       "Page-Hinkley recalibration": "#6b8e23", "PIRCAD": "#b3261e"}
MARK = {"Kalman bias estimator": "o", "EWMA recalibration": "s",
        "Page-Hinkley recalibration": "^", "PIRCAD": "*"}


def main():
    dose = json.load(open(ART / "exp14_dose.json"))["tuning"]
    e9 = json.load(open(ART / "exp9_adaptation_poisoning.json"))
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for name, pts in dose.items():
        x = [p["drift_fwd_error"][0] for p in pts]
        y = [p["alarm_34"][0] for p in pts]
        sz = [30 + 250 * p["spurious_34"][0] for p in pts]
        ax.plot(x, y, "-", color=COL[name], lw=1, alpha=0.6)
        ax.scatter(x, y, s=sz, color=COL[name], marker=MARK[name], alpha=0.85,
                   edgecolor="white", lw=0.6, label=name, zorder=3)
        for p, xi, yi in zip(pts, x, y):
            if p["default"]:
                ax.annotate("tuned", (xi, yi), xytext=(6, 8), textcoords="offset points",
                            fontsize=7.5, color=COL[name])
    px = e9["drift_summary"]["PIRCAD"]["drift_fwd_error"][0]
    py = e9["summary"]["PIRCAD"]["alarm_fired"][0]
    ps = e9["summary"]["PIRCAD"]["spurious_trip"][0]
    ax.scatter([px], [py], s=260 + 250 * ps, color=COL["PIRCAD"], marker="*",
               edgecolor="white", lw=0.6, label="PIRCAD (default)", zorder=4)
    ax.set_xscale("log")
    ax.set_xlabel("Correction error on benign drift (°C, log scale)")
    ax.set_ylabel("Share of episodes in which the alarm fired")
    ax.set_ylim(-0.06, 1.08)
    ax.grid(alpha=0.25, which="both")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(fontsize=8, frameon=False, loc="center right")
    fig.tight_layout()
    out = FIG / "rev_fig_tradeoff.png"
    fig.savefig(out, dpi=220, facecolor="white")
    print("wrote", out)


if __name__ == "__main__":
    main()
