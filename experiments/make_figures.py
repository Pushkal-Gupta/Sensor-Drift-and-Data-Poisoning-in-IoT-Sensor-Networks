"""
Figures for the Generation-2 disclosure.

  fig1_architecture_residuals.png  the pipeline's signals on one episode
  fig2_drift_vs_poison.png         why rate-matched drift and injection separate
  fig3_consistency_signatures.png  the constraint-violation signature space
  fig4_adaptation_poisoning.png    Experiment 2: absorption and the missed alarm
  fig5_confusion.png               Gen-1 vs Gen-2 confusion matrices

Run:  python experiments/make_figures.py
"""

from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from _common import ARTIFACTS, FIGURES, WARMUP, banner, get_models

from pircad.fognode import PIRCADFogNode
from pircad.rootcause import HybridRootCauseClassifier
from pircad.simulator import frames, make_episode
from pircad.types import RootCause

CAUSE_COLOURS = {0: "#94a3b8", 1: "#2563eb", 2: "#dc2626", 3: "#ea580c"}
CAUSE_NAMES = {0: "normal", 1: "sensor drift", 2: "data poisoning",
               3: "process event"}


def _run(scenario, clf, seed=900, n=4000):
    ep = make_episode(scenario, n=n, seed=seed)
    node = PIRCADFogNode(classifier=clf)
    res = node.run(frames(ep), truth_labels=ep.labels)
    return ep, node, res


def fig1(clf):
    ep, node, res = _run("stealth_poison", clf)
    d = res.per_channel["T_in"]
    z = np.array([x.residual_norm for x in d])
    trg = np.array([x.triggered for x in d])
    cause = res.causes()
    onset = int(ep.meta["onset"])

    fig, ax = plt.subplots(4, 1, figsize=(11, 9), sharex=True,
                           gridspec_kw={"height_ratios": [2, 2, 1, 1]})
    ax[0].plot(ep.truth["T_in"], lw=0.7, color="#0f766e", label="true temperature")
    ax[0].plot(ep.obs["T_in"], lw=0.5, color="#dc2626", alpha=0.8,
               label="reported (injected from onset)")
    ax[0].axvline(onset, color="k", ls="--", lw=0.8)
    ax[0].set_ylabel("degC"); ax[0].legend(loc="upper left", fontsize=8)
    ax[0].set_title("Stage 1 - raw stream: a slow ramp, indistinguishable by "
                    "magnitude or rate from transducer ageing")

    ax[1].plot(z, lw=0.6, color="#1d4ed8")
    ax[1].axhline(0, color="k", lw=0.5)
    ax[1].axvline(onset, color="k", ls="--", lw=0.8)
    ax[1].set_ylabel("residual (sigma)")
    ax[1].set_title("Stage 2 - physics residual: measurement minus model expectation")

    ax[2].fill_between(np.arange(len(trg)), 0, trg.astype(float), step="mid",
                       color="#a855f7", alpha=0.8)
    ax[2].set_ylabel("trigger"); ax[2].set_yticks([0, 1])
    ax[2].set_title("Stage 3 - statistical trigger (conditional activation)")

    for c, col in CAUSE_COLOURS.items():
        m = cause == c
        ax[3].fill_between(np.arange(len(cause)), 0, m.astype(float), step="mid",
                           color=col, alpha=0.85, label=CAUSE_NAMES[c])
    ax[3].set_ylabel("verdict"); ax[3].set_yticks([])
    ax[3].legend(loc="upper left", ncol=4, fontsize=8)
    ax[3].set_title("Stages 4-6 - root cause and differential response")
    ax[3].set_xlabel("sample")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig1_architecture_residuals.png", dpi=150)
    plt.close(fig)


def fig2(clf):
    """The two episodes are matched in magnitude and rate; only character differs.

    The response stage is DISABLED for this figure.  With it enabled the drift
    residual is driven back towards zero by its own calibration correction while
    the injection is quarantined and left uncorrected, so the two panels would
    differ because of what the system did, not because of what the deviations are.
    The point of the figure is the latter.
    """
    fig, ax = plt.subplots(2, 3, figsize=(13, 6))
    for row, sc in enumerate(("drift", "stealth_poison")):
        ep, node, res = _run(sc, None)
        d = res.per_channel["T_in"]
        z = np.array([x.residual_norm for x in d])
        onset = int(ep.meta["onset"])
        eng = node.res["T_in"]

        ax[row, 0].plot(z, lw=0.6, color="#1d4ed8")
        ax[row, 0].axvline(onset, color="k", ls="--", lw=0.8)
        ax[row, 0].set_title(f"{sc}: residual")
        ax[row, 0].set_ylabel("sigma")

        seg = slice(onset + 600, onset + 900)
        r = np.array([x.residual for x in d])[seg]
        ax[row, 1].plot(np.diff(r), lw=0.7, color="#dc2626")
        ax[row, 1].set_title("residual increments (noise signature)")

        # coherence with the modelled dynamics
        pred = np.array([x.residual for x in d])
        ax[row, 2].axis("off")
        f = eng.features()
        txt = "\n".join([
            f"slope_ratio     {f['slope_ratio']:.2f}",
            f"monotone_frac   {f['monotone_frac']:.2f}",
            f"corner          {f['corner']:.1f}",
            f"acf_shift       {f['acf_shift']:+.3f}",
            f"var_ratio       {f['var_ratio']:+.3f}",
            f"dyn_coherence   {f['dyn_coherence']:+.3f}",
        ])
        ax[row, 2].text(0.02, 0.95, f"end-of-run features ({sc})\n\n" + txt,
                        va="top", family="monospace", fontsize=9)
    fig.suptitle("Rate-matched benign drift vs false-data injection, response "
                 "stage disabled:\nidentical amplitude and growth rate, "
                 "different character", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig2_drift_vs_poison.png", dpi=150)
    plt.close(fig)


def fig3(clf):
    """Constraint-violation directions for each hypothesis, measured."""
    scenarios = ["clean", "drift", "stealth_poison", "process_event"]
    obs = {}
    for sc in scenarios:
        ep, node, res = _run(sc, clf)
        g = node.consistency.normalised_violation()
        obs[sc] = g / (np.linalg.norm(g) + 1e-9)
    names = list(node.consistency.CONSTRAINTS)
    sig = node.consistency.signatures

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    w = 0.2
    xs = np.arange(len(names))
    for i, (k, v) in enumerate(sig.items()):
        ax[0].bar(xs + i * w - 0.4, v, width=w, label=k)
    ax[0].set_xticks(xs); ax[0].set_xticklabels(names, rotation=20, fontsize=8)
    ax[0].set_title("hypothesis signatures (from the model, not from data)")
    ax[0].legend(fontsize=7, ncol=2); ax[0].axhline(0, color="k", lw=0.5)

    for i, (k, v) in enumerate(obs.items()):
        ax[1].bar(xs + i * w - 0.3, v, width=w, label=k)
    ax[1].set_xticks(xs); ax[1].set_xticklabels(names, rotation=20, fontsize=8)
    ax[1].set_title("observed violation directions (end of episode)")
    ax[1].legend(fontsize=7); ax[1].axhline(0, color="k", lw=0.5)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig3_consistency_signatures.png", dpi=150)
    plt.close(fig)


def fig4():
    p = ARTIFACTS / "exp2_traces.npz"
    if not p.exists():
        print("  (skipping fig4: run exp2 first)")
        return
    z = np.load(p, allow_pickle=True)
    onset, attack_end, fault_at, alarm = z["meta"]
    onset, attack_end, fault_at = int(onset), int(attack_end), int(fault_at)

    # PIRCAD withholds quarantined samples, so its line must be broken there --
    # drawing through them would show it reporting data it never forwarded.
    pircad_seen = np.array(z["pircad_reported"], dtype=float)
    if "pircad_forwarded" in z:
        pircad_seen[~np.asarray(z["pircad_forwarded"], dtype=bool)] = np.nan

    fig, ax = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    ax[0].plot(z["truth"], color="#0f766e", lw=0.8, label="true temperature")
    ax[0].plot(z["adaptive_reported"], color="#dc2626", lw=0.8,
               label="reported - adaptive recalibrating baseline")
    ax[0].plot(pircad_seen, color="#1d4ed8", lw=0.8,
               label="forwarded by PIRCAD (gap = quarantined)")
    ax[0].axhline(alarm, color="k", ls=":", lw=1.0, label=f"cloud alarm {alarm} C")
    ax[0].axvspan(onset, attack_end, color="#fecaca", alpha=0.5)
    ax[0].axvline(fault_at, color="#ea580c", ls="--", lw=1.0)
    ax[0].text(onset + 40, ax[0].get_ylim()[1] * 0.995, " injection window",
               fontsize=8, va="top")
    ax[0].set_ylabel("degC"); ax[0].legend(fontsize=8, loc="lower left")
    ax[0].set_title("Adaptation poisoning: the attack is absorbed, then the real "
                    "fault is under-reported")

    ax[1].plot(z["adaptive_offset"], color="#dc2626", lw=1.0,
               label="baseline: absorbed bias")
    ax[1].plot(z["pircad_offset"], color="#1d4ed8", lw=1.0,
               label="PIRCAD: calibration offset")
    ax[1].plot(z["nodiff_offset"], color="#a855f7", lw=1.0, ls="--",
               label="PIRCAD without differential response")
    ax[1].axvspan(onset, attack_end, color="#fecaca", alpha=0.5)
    ax[1].axhline(0, color="k", lw=0.5)
    ax[1].set_ylabel("degC"); ax[1].set_xlabel("sample")
    ax[1].legend(fontsize=8); ax[1].set_title("What each system adopted as its new zero")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig4_adaptation_poisoning.png", dpi=150)
    plt.close(fig)


def fig5():
    p = ARTIFACTS / "exp1_root_cause_benchmark.json"
    if not p.exists():
        print("  (skipping fig5: run exp1 first)")
        return
    blob = json.loads(p.read_text())
    labels = blob["labels"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
    for a, key, title in ((ax[0], "confusion_pircad", "PIRCAD (Gen 2)"),
                          (ax[1], "confusion_gen1", "FHAD (Gen 1)")):
        cm = np.asarray(blob[key], dtype=float)
        norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        im = a.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        a.set_xticks(range(len(labels)))
        a.set_xticklabels([l.replace("_", "\n") for l in labels], fontsize=8)
        a.set_yticks(range(len(labels)))
        a.set_yticklabels([l.replace("_", "\n") for l in labels], fontsize=8)
        for i in range(len(labels)):
            for j in range(len(labels)):
                a.text(j, i, f"{norm[i, j]:.2f}", ha="center", va="center",
                       fontsize=8, color="white" if norm[i, j] > 0.5 else "black")
        a.set_title(title); a.set_xlabel("predicted"); a.set_ylabel("true cause")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.savefig(FIGURES / "fig5_confusion.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    banner("Generating figures")
    models = get_models()
    clf = HybridRootCauseClassifier(models["root_cause"])
    for name, fn in (("fig1", lambda: fig1(clf)), ("fig2", lambda: fig2(clf)),
                     ("fig3", lambda: fig3(clf)), ("fig4", fig4), ("fig5", fig5)):
        print(f"  {name} ...")
        fn()
    print(f"\nfigures written to {FIGURES}")


if __name__ == "__main__":
    main()
