"""Draw Fig. 1 (paper/figures/architecture.png): the stages E1-E8 in order.

    python paper/make_architecture.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

OUT = Path(__file__).resolve().parent / "figures" / "architecture.png"
TEAL, GREY, RED, INK = "#127c7c", "#4a4f57", "#b3261e", "#222222"

STAGES = [("E1", "process\nmodel", TEAL), ("E2", "residual &\nfeatures", TEAL),
          ("E3", "trigger", GREY), ("E4", "physical\nlocalisation", TEAL),
          ("E5", "root-cause\nlearner", TEAL), ("E6", "confidence\ngate", TEAL),
          ("E8", "differential\nresponse", TEAL)]


def main():
    fig, ax = plt.subplots(figsize=(13.8, 4.3))
    ax.set_xlim(0, 13.8); ax.set_ylim(0, 4.3); ax.axis("off")
    w, h, y, gap, x0 = 1.42, 1.05, 2.45, 0.32, 1.2
    xs = [x0 + i * (w + gap) for i in range(len(STAGES))]
    for x, (tag, label, col) in zip(xs, STAGES):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                    fc="white", ec=col, lw=2.4))
        ax.text(x + w / 2, y + h - 0.22, tag, ha="center", va="center", color=col,
                fontsize=13, fontweight="bold")
        ax.text(x + w / 2, y + 0.38, label, ha="center", va="center", color=INK, fontsize=10.5)
    arrow = dict(arrowstyle="-|>", mutation_scale=14, lw=1.6, color=INK)
    ax.add_patch(FancyArrowPatch((0.05, y + h / 2), (xs[0], y + h / 2), **arrow))
    ax.text(0.05, y + h / 2 + 0.2, "sensor data", fontsize=9.5, color=GREY)
    for a, b in zip(xs[:-1], xs[1:]):
        ax.add_patch(FancyArrowPatch((a + w, y + h / 2), (b, y + h / 2), **arrow))
    ax.add_patch(FancyArrowPatch((xs[-1] + w, y + h / 2), (13.75, y + h / 2), **arrow))
    ax.text(xs[-1] + w + 0.06, y + h / 2 + 0.2, "to cloud", fontsize=9.5, color=GREY)
    # bracket over the stages that run only when E3 fires
    bx0, bx1, by = xs[3], xs[6] + w, y + h + 0.3
    ax.plot([bx0, bx0, bx1, bx1], [by - 0.1, by, by, by - 0.1], color=GREY, lw=1.4)
    ax.text((bx0 + bx1) / 2, by + 0.15, "executed only when E3 fires", ha="center",
            fontsize=10.5, color=GREY)
    # governor
    gx0, gx1, gy0, gy1 = xs[1] - 0.2, xs[5] + w, 0.15, 1.35
    ax.add_patch(FancyBboxPatch((gx0, gy0), gx1 - gx0, gy1 - gy0,
                                boxstyle="round,pad=0.02,rounding_size=0.12",
                                fc="white", ec=RED, lw=2.0, ls=(0, (5, 3))))
    ax.text((gx0 + gx1) / 2, gy1 - 0.35, "E7  ADAPTATION GOVERNOR", ha="center",
            color=RED, fontsize=12.5, fontweight="bold")
    ax.text((gx0 + gx1) / 2, gy0 + 0.33,
            "on a suspect channel it freezes every element that learns:\n"
            "process-model anchoring (E1), residual statistics (E2), calibration offset (E8)",
            ha="center", va="center", fontsize=9.8, color=INK)
    dashed = dict(arrowstyle="-|>", mutation_scale=13, lw=1.6, color=RED, ls=(0, (4, 3)))
    for tx, sx in ((xs[0] + w / 2, gx0), (xs[1] + w / 2, xs[1] + w / 2), (xs[6] + w / 2, gx1)):
        if sx == gx0 or sx == gx1:
            ax.plot([sx, tx], [0.75, 0.75], color=RED, lw=1.6, ls=(0, (4, 3)))
            ax.add_patch(FancyArrowPatch((tx, 0.75), (tx, y), **dashed))
        else:
            ax.add_patch(FancyArrowPatch((tx, gy1), (tx, y), **dashed))
    # verdicts feed the governor
    ax.add_patch(FancyArrowPatch((xs[5] + w / 2, y), (xs[5] + w / 2, gy1),
                                 arrowstyle="-|>", mutation_scale=13, lw=1.4, color=GREY))
    ax.text(xs[5] + w / 2 + 0.08, (y + gy1) / 2, "verdict", fontsize=9.5, color=GREY, va="center")
    fig.savefig(OUT, dpi=220, bbox_inches="tight", facecolor="white")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
