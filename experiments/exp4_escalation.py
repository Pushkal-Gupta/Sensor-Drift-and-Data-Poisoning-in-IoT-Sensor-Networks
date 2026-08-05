"""
Experiment 4 -- confidence-gated escalation, with a quantum-kernel backend.

Two questions, kept strictly separate.

1. Does the confidence gate select the right samples?  If escalation is worth
   anything, the samples it holds back must be the ones the fog node was about to
   get wrong.  Measured as the local accuracy on escalated versus non-escalated
   samples.

2. What does escalation cost and buy?  Measured as the fraction of samples that
   leave the fog node, the accuracy on those samples for a conventional heavy
   backend versus a gate-model quantum-kernel backend, and the end-to-end effect
   once the (deliberately large) round-trip latency is paid.

The quantum backend here is a ZZ-feature-map fidelity kernel evaluated by exact
statevector simulation.  No quantum advantage is claimed: the point of the
experiment is that the architecture *tolerates* a slow, modest, optional backend
because the backend is off the real-time path and the local verdict stands if it
never answers.

Run:  python experiments/exp4_escalation.py       (add --quantum to fit the QKE)
"""

from __future__ import annotations

import argparse
import time
from typing import Dict, List

import numpy as np

from _common import WARMUP, banner, eval_episodes, get_models, save_json

from pircad.escalation import EscalationService
from pircad.fognode import PIRCADFogNode
from pircad.metrics import aggregate, format_table
from pircad.rootcause import HybridRootCauseClassifier, build_vector
from pircad.simulator import frames
from pircad.types import RootCause


def gate_quality(models, episodes) -> Dict[str, float]:
    """Is the confidence gate selecting the samples the fog node gets wrong?"""
    clf = HybridRootCauseClassifier(models["root_cause"])
    lo_ok = lo_n = hi_ok = hi_n = 0
    thresh = 0.62
    for ep in episodes:
        rows: List[tuple] = []

        def probe(k, channel, feats, loc, fired, _r=rows, _ep=ep):
            if feats is None:
                return
            r = clf.classify(channel, feats, loc)
            truth = int(_ep.labels[k])
            if _ep.target_channel is not None and channel != _ep.target_channel:
                truth = int(RootCause.NORMAL)
            _r.append((r.confidence, int(r.cause) == truth))

        node = PIRCADFogNode(classifier=clf, feature_probe=probe)
        node.run(frames(ep), truth_labels=ep.labels)
        for conf, ok in rows:
            if conf < thresh:
                lo_n += 1; lo_ok += int(ok)
            else:
                hi_n += 1; hi_ok += int(ok)
    return {
        "threshold": thresh,
        "escalated_fraction": lo_n / max(1, lo_n + hi_n),
        "local_accuracy_when_confident": hi_ok / max(1, hi_n),
        "local_accuracy_when_unsure": lo_ok / max(1, lo_n),
        "n_confident": hi_n, "n_unsure": lo_n,
    }


def backend_accuracy(models, episodes, backend, name: str) -> Dict[str, float]:
    """Accuracy of a backend on exactly the samples the gate would escalate."""
    clf = HybridRootCauseClassifier(models["root_cause"])
    X, y, local_ok = [], [], []
    for ep in episodes:
        def probe(k, channel, feats, loc, fired, _ep=ep):
            if feats is None:
                return
            r = clf.classify(channel, feats, loc)
            if r.confidence >= 0.62:
                return
            truth = int(_ep.labels[k])
            if _ep.target_channel is not None and channel != _ep.target_channel:
                truth = int(RootCause.NORMAL)
            X.append(build_vector(channel, feats, loc))
            y.append(truth)
            local_ok.append(int(r.cause) == truth)

        node = PIRCADFogNode(classifier=clf, feature_probe=probe)
        node.run(frames(ep), truth_labels=ep.labels)

    if not X:
        return {"backend": name, "n": 0}
    X = np.vstack(X); y = np.asarray(y)
    t0 = time.perf_counter()
    labels, confs = backend.classify(X)
    dt = time.perf_counter() - t0
    return {
        "backend": name,
        "n_escalated_samples": float(len(y)),
        "local_accuracy": float(np.mean(local_ok)),
        "backend_accuracy": float(np.mean(labels == y)),
        "accuracy_gain": float(np.mean(labels == y) - np.mean(local_ok)),
        "inference_ms_per_sample": float(1000 * dt / len(y)),
    }


def end_to_end(models, episodes, backend, latency: int, name: str) -> Dict[str, float]:
    from pircad.metrics import evaluate_run
    clf = HybridRootCauseClassifier(models["root_cause"])
    rows = []
    for ep in episodes:
        esc = EscalationService(backend=backend, latency_samples=latency)
        node = PIRCADFogNode(classifier=clf, escalation=esc)
        res = node.run(frames(ep), truth_labels=ep.labels)
        rows.append(evaluate_run(ep, res, WARMUP))
    out = aggregate(rows)
    out["_name"] = name
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quantum", action="store_true",
                    help="fit and evaluate the quantum-kernel backend")
    ap.add_argument("--episodes", type=int, default=12)
    args = ap.parse_args()

    models = get_models(quantum=args.quantum)
    episodes = eval_episodes()[:args.episodes]

    banner("EXPERIMENT 4 - confidence-gated escalation")

    print("\n(1) Is the gate selecting the right samples?")
    gq = gate_quality(models, episodes)
    for k, v in gq.items():
        print(f"    {k:34s} {v:.4f}" if isinstance(v, float) else f"    {k:34s} {v}")
    print("\n    If the accuracy when unsure is materially below the accuracy when\n"
          "    confident, the gate is separating the hard cases from the easy ones,\n"
          "    which is the whole justification for spending anything on them.")

    print("\n(2) What does a backend add on those samples?")
    results = [backend_accuracy(models, episodes, models["heavy"], "gradient boosting")]
    if args.quantum and models.get("quantum") is not None:
        qb = models["quantum"]
        results.append(backend_accuracy(models, episodes, qb, "quantum kernel (simulated)"))
        print("\n    Quantum circuit resources per kernel entry:")
        for k, v in qb.resource_estimate().items():
            print(f"      {k:28s} {v}")

    banner("Backend comparison on escalated samples only")
    keys = ["n_escalated_samples", "local_accuracy", "backend_accuracy",
            "accuracy_gain", "inference_ms_per_sample"]
    table = {r["backend"]: r for r in results if r.get("n_escalated_samples")}
    if table:
        print(format_table(table, keys, width=26))

    print("\n(3) End-to-end, with the round trip actually paid:")
    e2e = {"no escalation": end_to_end(models, episodes, None, 0, "none"),
           "heavy backend (60-sample RTT)":
               end_to_end(models, episodes, models["heavy"], 60, "heavy")}
    if args.quantum and models.get("quantum") is not None:
        e2e["quantum backend (600-sample RTT)"] = end_to_end(
            models, episodes, models["quantum"], 600, "quantum")
    print(format_table(e2e, ["root_cause_accuracy", "poison_containment",
                             "false_alarm_rate", "escalations",
                             "avg_latency_s"], width=26))
    print("\n    The quantum column carries a 600-sample round trip on purpose:\n"
          "    published gate-model kernel pipelines report cloud round trips in\n"
          "    the minutes.  It stays usable only because escalated samples are\n"
          "    held rather than forwarded, and the local verdict is never blocked.")

    save_json("exp4_escalation.json",
              {"gate_quality": gq, "backends": results,
               "end_to_end": {k: {kk: vv for kk, vv in v.items()
                                  if not kk.startswith("_")} for k, v in e2e.items()}})


if __name__ == "__main__":
    main()
