"""Shared setup for the PIRCAD experiments: paths, model cache, printing."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ARTIFACTS = ROOT / "artifacts"
FIGURES = ROOT / "figures"
ARTIFACTS.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)

#: seeds used to fit the root-cause model
TRAIN_SEEDS = (0, 1, 2)
#: disjoint seeds used for every number that gets reported
EVAL_SEEDS = (900, 901, 902, 903)
EPISODE_LEN = 4000
WARMUP = 600

SCENARIOS = ["clean", "drift", "stealth_poison", "spike_poison",
             "replay_poison", "process_event"]


def get_models(force: bool = False, quantum: bool = False):
    """Fit (or load) the root-cause model and the Gen-1 comparison model."""
    import joblib
    from pircad.baselines import Gen1FHAD
    from pircad.training import train_root_cause, training_episodes

    cache = ARTIFACTS / "models.joblib"
    if cache.exists() and not force:
        blob = joblib.load(cache)
        if not quantum or blob.get("quantum") is not None:
            return blob

    print("[setup] fitting models (this takes a few minutes)...")
    clf, corpus = train_root_cause(seeds=TRAIN_SEEDS, n=EPISODE_LEN)
    gen1 = Gen1FHAD().fit(training_episodes(TRAIN_SEEDS, EPISODE_LEN))
    blob = {"root_cause": clf, "corpus": corpus, "gen1": gen1,
            "heavy": None, "quantum": None}

    from pircad.training import train_escalation_backends
    heavy, qb = train_escalation_backends(corpus, quantum=quantum)
    blob["heavy"] = heavy
    blob["quantum"] = qb

    joblib.dump(blob, cache)
    print(f"[setup] models cached at {cache}")
    return blob


def eval_episodes(scenarios: Sequence[str] = SCENARIOS,
                  seeds: Sequence[int] = EVAL_SEEDS,
                  n: int = EPISODE_LEN):
    """Held-out evaluation bank, with varied onsets and rates."""
    from pircad.simulator import make_episode
    eps = []
    for s in seeds:
        for j, sc in enumerate(scenarios):
            rng = np.random.default_rng(7919 * s + j)
            onset = int(rng.integers(n // 4, n // 2))
            rate = float(rng.uniform(0.8e-3, 2.5e-3))
            eps.append(make_episode(sc, n=n, seed=int(7919 * s + j),
                                    onset=onset, rate=rate))
    return eps


def save_json(name: str, obj) -> Path:
    p = ARTIFACTS / name
    p.write_text(json.dumps(obj, indent=2, default=float))
    print(f"[saved] {p}")
    return p


def banner(text: str) -> None:
    print()
    print("=" * 78)
    print(text)
    print("=" * 78)


def print_confusion(cm: np.ndarray, labels: Sequence[str], title: str = "") -> None:
    if title:
        print(f"\n{title}")
    w = max(12, max(len(l) for l in labels) + 2)
    print(" " * w + "".join(l[:11].rjust(12) for l in labels) + "   (rows = truth)")
    for i, l in enumerate(labels):
        print(l.ljust(w) + "".join(str(int(v)).rjust(12) for v in cm[i]))
