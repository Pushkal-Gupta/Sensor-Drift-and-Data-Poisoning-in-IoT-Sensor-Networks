"""Fast single-sample inference for a fitted random forest (fog-node use)."""

from __future__ import annotations

from typing import Optional

import numpy as np


class FastForest:
    """Numerically identical, vectorised `predict_proba` for a fitted forest.

    scikit-learn's forest inference dispatches one joblib task per tree, which
    costs ~16 ms per *single* sample regardless of n_jobs -- 97% of a fog-node
    run.  A fog node classifies one sample at a time, so the forest is
    flattened into arrays and all trees are traversed together.  Agreement with
    sklearn is asserted on construction.
    """

    def __init__(self, forest, check: Optional[np.ndarray] = None):
        trees = [e.tree_ for e in forest.estimators_]
        self.classes_ = forest.classes_
        offs = np.cumsum([0] + [t.node_count for t in trees[:-1]])
        self.roots = offs.astype(np.int64)
        L = np.concatenate([np.where(t.children_left >= 0, t.children_left + o, -1)
                            for t, o in zip(trees, offs)])
        R = np.concatenate([np.where(t.children_right >= 0, t.children_right + o, -1)
                            for t, o in zip(trees, offs)])
        self.left, self.right = L, R
        self.feat = np.concatenate([t.feature for t in trees]).astype(np.int64)
        self.thr = np.concatenate([t.threshold for t in trees])
        v = np.concatenate([t.value[:, 0, :] for t in trees])
        self.val = v / np.maximum(v.sum(axis=1, keepdims=True), 1e-300)
        self.depth = max(e.get_depth() for e in forest.estimators_)
        self.n_trees = len(trees)
        if check is not None:
            ref = forest.predict_proba(check)
            got = np.vstack([self.predict_proba(x[None]) for x in check])
            assert np.allclose(ref, got, atol=1e-12), "FastForest disagrees with sklearn"

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        x = np.asarray(X, dtype=np.float32).astype(np.float64)[0]
        node = self.roots.copy()
        for _ in range(self.depth + 1):
            leaf = self.left[node] < 0
            if leaf.all():
                break
            f = self.feat[node]
            go_left = x[np.where(leaf, 0, f)] <= self.thr[node]
            node = np.where(leaf, node, np.where(go_left, self.left[node], self.right[node]))
        return self.val[node].mean(axis=0, keepdims=True)
