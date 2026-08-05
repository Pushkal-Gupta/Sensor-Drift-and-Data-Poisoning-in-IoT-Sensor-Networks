"""
Training of the root-cause stage (and of the escalation backends).

Important property of the design: feature extraction does not depend on the
classifier.  Whether a channel is engaged, and how much weight the process model
gives it, are decided by the *trigger* and the governor, never by the root-cause
verdict.  So a training corpus can be collected by running the full pipeline with
no classifier at all, and the features seen at training time are the ones seen at
inference time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .config import FogConfig
from .fognode import PIRCADFogNode
from .rootcause import (ALL_FEATURES, MLRootCauseClassifier, build_vector)
from .simulator import Episode, frames, make_episode
from .twin import PhysicsVesselTwin
from .types import RootCause


@dataclass
class Corpus:
    X: np.ndarray
    y: np.ndarray
    groups: np.ndarray          # episode index, for group-wise splitting
    channels: List[str]
    feature_names: List[str]

    def __len__(self) -> int:
        return len(self.y)

    def summary(self) -> Dict[str, int]:
        return {RootCause(int(c)).name: int((self.y == c).sum())
                for c in np.unique(self.y)}


def collect_episode(ep: Episode, cfg: Optional[FogConfig] = None,
                    group: int = 0) -> Tuple[List[np.ndarray], List[int], List[int]]:
    """Run one episode with no classifier and harvest (features, label)."""
    X: List[np.ndarray] = []
    y: List[int] = []
    grp: List[int] = []

    def probe(k, channel, feats, loc, fired):
        label = int(ep.labels[k])
        # a channel that is not the attacked one is, by construction, healthy;
        # this is what teaches the model to say NORMAL on collateral triggers
        if ep.target_channel is not None and channel != ep.target_channel:
            label = int(RootCause.NORMAL)
        if ep.scenario == "process_event":
            label = int(RootCause.PROCESS_EVENT) if ep.labels[k] else int(RootCause.NORMAL)
        X.append(build_vector(channel, feats, loc))
        y.append(label)
        grp.append(group)

    node = PIRCADFogNode(twin=PhysicsVesselTwin(), classifier=None,
                         cfg=cfg or FogConfig(), feature_probe=probe)
    node.run(frames(ep))
    return X, y, grp


def build_corpus(episodes: Sequence[Episode],
                 cfg: Optional[FogConfig] = None) -> Corpus:
    X: List[np.ndarray] = []
    y: List[int] = []
    g: List[int] = []
    for i, ep in enumerate(episodes):
        xi, yi, gi = collect_episode(ep, cfg, group=i)
        X.extend(xi); y.extend(yi); g.extend(gi)
    if not X:
        raise RuntimeError("no features collected -- did any trigger fire?")
    return Corpus(np.vstack(X), np.asarray(y, dtype=int), np.asarray(g, dtype=int),
                  channels=[], feature_names=list(ALL_FEATURES))


def training_episodes(seeds: Sequence[int], n: int = 4000) -> List[Episode]:
    """Balanced training bank.  Kept disjoint from the evaluation seeds."""
    scenarios = ["clean", "drift", "stealth_poison", "spike_poison",
                 "replay_poison", "process_event"]
    eps: List[Episode] = []
    for s in seeds:
        for j, sc in enumerate(scenarios):
            # vary onset and rate so the model cannot key on a fixed geometry
            rng = np.random.default_rng(1000 * s + j)
            onset = int(rng.integers(n // 4, n // 2))
            rate = float(rng.uniform(0.8e-3, 2.5e-3))
            eps.append(make_episode(sc, n=n, seed=int(1000 * s + j),
                                    onset=onset, rate=rate))
    return eps


def train_root_cause(seeds: Sequence[int] = (0, 1, 2, 3),
                     n: int = 4000, model_seed: int = 0,
                     cfg: Optional[FogConfig] = None,
                     verbose: bool = True) -> Tuple[MLRootCauseClassifier, Corpus]:
    eps = training_episodes(seeds, n)
    corpus = build_corpus(eps, cfg)
    if verbose:
        print(f"[training] corpus {len(corpus)} rows  {corpus.summary()}")
    clf = MLRootCauseClassifier().fit(corpus.X, corpus.y, seed=model_seed)
    return clf, corpus


def train_escalation_backends(corpus: Corpus, seed: int = 0,
                              quantum: bool = True, q_subsample: int = 900):
    """Fit the advanced classifiers used by the escalation service."""
    from .escalation import LocalHeavyBackend, QuantumEscalationBackend

    heavy = LocalHeavyBackend(seed=seed).fit(corpus.X, corpus.y)
    qb = None
    if quantum:
        rng = np.random.default_rng(seed)
        # the quantum kernel is O(n^2) in training-set size; subsample, balanced
        idx: List[int] = []
        per = max(50, q_subsample // max(1, len(np.unique(corpus.y))))
        for c in np.unique(corpus.y):
            ci = np.flatnonzero(corpus.y == c)
            idx.extend(rng.choice(ci, size=min(per, ci.size), replace=False))
        idx = np.asarray(idx)
        qb = QuantumEscalationBackend(n_qubits=4, reps=2, seed=seed)
        qb.fit(corpus.X[idx], corpus.y[idx])
    return heavy, qb
