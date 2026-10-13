"""
Data-driven comparison systems for the journal evaluation.

Root-cause baselines (supervised, same training episodes and the same four
labels as PIRCAD; they see every raw channel plus the heater command, i.e. at
least the information PIRCAD sees, but no physics):

GenericFeatureRF   a random forest over generic multi-scale window statistics
                   of every channel, including rolling cross-channel
                   correlations -- so it *could* learn the physical coupling
                   from data if the data supports it.
MultiScaleCNN      a dilated 1-D CNN over the raw standardised channels at two
                   time scales (last 128 samples at full rate, last 1024 samples
                   mean-pooled by 8).

Detection baselines (unsupervised, fitted on clean data only, thresholded at the
99th percentile of their clean-training score):

IsolationForest, PCA-SPE   on the generic features
DenseAE                    reconstruction error on the short raw window
KalmanChi2                 windowed normalised-innovation-squared of a
                           single closed-loop observer over all channels -- the
                           classical bad-data detector of state estimation.

All of these output per-sample predictions for an `Episode` and are scored by
the same metric code as PIRCAD.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .config import CHANNELS
from .types import RootCause

SIG = list(CHANNELS) + ["u_heater"]
PAIRS = [("T_in", "P"), ("T_in", "RH"), ("P", "RH"), ("T_in", "u_heater"),
         ("T_in", "T_amb")]
WARM = 600


def signals(ep) -> List[str]:
    """Signals a data-driven baseline may see.  Ground truth (u_phys) never."""
    return list(ep.meta.get("signals", SIG))


def _warm(ep) -> int:
    return int(ep.meta.get("warm", WARM))


def _series(ep, name: str) -> np.ndarray:
    src = ep.obs if name in ep.obs else ep.actuation
    return np.asarray(src[name], float)


# --------------------------------------------------------------------------
# inputs
# --------------------------------------------------------------------------
def standardised(ep) -> np.ndarray:
    """(n, S) array, continuous signals standardised on the clean warm-up."""
    cols, w = [], _warm(ep)
    for c in signals(ep):
        x = _series(ep, c)
        if c.startswith("u_"):          # binary command, left as is
            cols.append(x)
            continue
        mu, sd = x[:w].mean(), x[:w].std() + 1e-9
        cols.append((x - mu) / sd)
    return np.stack(cols, axis=1)


def generic_features(ep) -> np.ndarray:
    """(n, F) causal multi-scale window statistics."""
    X = standardised(ep)
    df = pd.DataFrame(X, columns=signals(ep))
    feats = []
    idx = pd.Series(np.arange(len(df), dtype=float))
    for w in (120, 960):
        r = df.rolling(w, min_periods=16)
        mean, std = r.mean(), r.std()
        feats += [mean, std, r.min(), r.max(), df - mean]
        # least-squares slope = cov(x, t) / var(t)
        cov = df.apply(lambda s: s.rolling(w, min_periods=16).cov(idx))
        feats.append(cov.div(idx.rolling(w, min_periods=16).var(), axis=0))
        d = df.diff()
        feats.append(d.rolling(w, min_periods=16).std())
        feats.append(d.apply(lambda s: s.rolling(w, min_periods=16).corr(s.shift(1))))
        # cross-channel coupling, the generic stand-in for physics
        for a, b in [tuple(p) for p in ep.meta.get("pairs", PAIRS)]:
            feats.append(df[a].rolling(w, min_periods=16).corr(df[b]).to_frame(f"c_{a}_{b}_{w}"))
            feats.append(d[a].rolling(w, min_periods=16).corr(d[b]).to_frame(f"dc_{a}_{b}_{w}"))
    F = pd.concat(feats, axis=1).to_numpy(dtype=float)
    return np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0)


def multiscale_windows(ep, idx: np.ndarray, X: Optional[np.ndarray] = None) -> np.ndarray:
    """(len(idx), 2S, 128) tensor: short window + 8x pooled long window."""
    X = standardised(ep) if X is None else X
    S = X.shape[1]
    pad = np.vstack([np.repeat(X[:1], 1024, axis=0), X])
    out = np.empty((len(idx), 2 * S, 128), dtype=np.float32)
    for j, t in enumerate(idx):
        tp = t + 1024
        out[j, :S] = pad[tp - 127: tp + 1].T
        out[j, S:] = pad[tp - 1023: tp + 1].reshape(128, 8, S).mean(axis=1).T
    return out


def frame_labels(ep) -> np.ndarray:
    return np.asarray(ep.labels, dtype=int)


# --------------------------------------------------------------------------
# supervised root-cause baselines
# --------------------------------------------------------------------------
class GenericFeatureRF:
    name = "Generic-feature RF"

    def __init__(self, seed: int = 0, stride: int = 2):
        self.seed, self.stride = seed, stride

    def fit(self, episodes: Sequence) -> "GenericFeatureRF":
        from sklearn.ensemble import RandomForestClassifier
        Xs, ys = [], []
        for ep in episodes:
            F = generic_features(ep)
            sl = slice(_warm(ep), None, self.stride)
            Xs.append(F[sl]); ys.append(frame_labels(ep)[sl])
        self.model = RandomForestClassifier(
            n_estimators=300, max_depth=16, min_samples_leaf=8,
            class_weight="balanced_subsample", n_jobs=-1,
            random_state=self.seed).fit(np.vstack(Xs), np.concatenate(ys))
        return self

    def predict(self, ep) -> np.ndarray:
        F = generic_features(ep)
        y = np.full(len(F), int(RootCause.NORMAL))
        y[_warm(ep):] = self.model.predict(F[_warm(ep):])
        return y


class MultiScaleCNN:
    name = "Multi-scale 1D-CNN"

    def __init__(self, seed: int = 0, epochs: int = 10, stride: int = 3):
        self.seed, self.epochs, self.stride = seed, epochs, stride

    def _net(self, n_in: int = 10):
        import torch.nn as nn
        def block(i, o, k, d):
            return nn.Sequential(nn.Conv1d(i, o, k, dilation=d, padding=d * (k // 2)),
                                 nn.BatchNorm1d(o), nn.ReLU())
        return nn.Sequential(block(n_in, 32, 7, 1), block(32, 64, 5, 2),
                             block(64, 64, 3, 4), block(64, 64, 3, 8),
                             nn.AdaptiveAvgPool1d(1), nn.Flatten(),
                             nn.Dropout(0.2), nn.Linear(64, 4))

    def fit(self, episodes: Sequence) -> "MultiScaleCNN":
        import torch
        torch.manual_seed(self.seed)
        torch.set_num_threads(8)
        Xs, ys = [], []
        for ep in episodes:
            idx = np.arange(_warm(ep), len(ep.labels), self.stride)
            Xs.append(multiscale_windows(ep, idx)); ys.append(frame_labels(ep)[idx])
        X = torch.from_numpy(np.concatenate(Xs)); y = torch.from_numpy(np.concatenate(ys)).long()
        counts = torch.bincount(y, minlength=4).float()
        w = (counts.sum() / (4 * counts.clamp(min=1)))
        self.net = self._net(X.shape[1])
        opt = torch.optim.AdamW(self.net.parameters(), lr=2e-3, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=3e-3, total_steps=self.epochs * (len(X) // 256 + 1))
        lossf = torch.nn.CrossEntropyLoss(weight=w)
        g = torch.Generator().manual_seed(self.seed)
        self.net.train()
        for e in range(self.epochs):
            perm = torch.randperm(len(X), generator=g)
            tot = 0.0
            for i in range(0, len(X), 256):
                b = perm[i:i + 256]
                opt.zero_grad()
                loss = lossf(self.net(X[b]), y[b])
                loss.backward(); opt.step(); sched.step()
                tot += float(loss) * len(b)
            print(f"    [cnn] epoch {e + 1}/{self.epochs} loss {tot / len(X):.4f}", flush=True)
        self.net.eval()
        return self

    def predict(self, ep) -> np.ndarray:
        import torch
        torch.set_num_threads(1)
        idx = np.arange(_warm(ep), len(ep.labels))
        y = np.full(len(ep.labels), int(RootCause.NORMAL))
        Xs = standardised(ep)
        with torch.no_grad():
            for i in range(0, len(idx), 1024):
                xb = torch.from_numpy(multiscale_windows(ep, idx[i:i + 1024], Xs))
                y[idx[i:i + 1024]] = self.net(xb).argmax(1).numpy()
        return y


# --------------------------------------------------------------------------
# unsupervised detection baselines
# --------------------------------------------------------------------------
class _ScoreDetector:
    name = "detector"
    quantile = 0.99

    def scores(self, ep) -> np.ndarray:
        raise NotImplementedError

    def calibrate(self, clean_eps: Sequence) -> None:
        s = np.concatenate([self.scores(ep)[WARM:] for ep in clean_eps])
        self.threshold = float(np.quantile(s, self.quantile))

    def predict(self, ep) -> np.ndarray:
        """1 = anomalous, 0 = normal (no cause vocabulary)."""
        s = self.scores(ep)
        y = (s > self.threshold).astype(int)
        y[:WARM] = 0
        return y


class IsolationForestDetector(_ScoreDetector):
    name = "Isolation Forest"

    def fit(self, clean_eps: Sequence):
        from sklearn.ensemble import IsolationForest
        X = np.vstack([generic_features(ep)[WARM::2] for ep in clean_eps])
        self.m = IsolationForest(n_estimators=200, random_state=0, n_jobs=-1).fit(X)
        self.m.n_jobs = 1
        self.calibrate(clean_eps)
        return self

    def scores(self, ep):
        return -self.m.score_samples(generic_features(ep))


class PCASPEDetector(_ScoreDetector):
    name = "PCA-SPE"

    def fit(self, clean_eps: Sequence, var: float = 0.95):
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        X = np.vstack([generic_features(ep)[WARM::2] for ep in clean_eps])
        self.sc = StandardScaler().fit(X)
        self.pca = PCA(n_components=var, random_state=0).fit(self.sc.transform(X))
        self.calibrate(clean_eps)
        return self

    def scores(self, ep):
        Z = self.sc.transform(generic_features(ep))
        R = Z - self.pca.inverse_transform(self.pca.transform(Z))
        return (R ** 2).sum(axis=1)


class DenseAEDetector(_ScoreDetector):
    name = "Autoencoder"

    def fit(self, clean_eps: Sequence, epochs: int = 15):
        import torch, torch.nn as nn
        torch.manual_seed(0)
        X = np.concatenate([multiscale_windows(ep, np.arange(WARM, len(ep.labels), 4))
                            for ep in clean_eps]).reshape(-1, 1280)
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-6
        Xt = torch.from_numpy(((X - self.mu) / self.sd).astype(np.float32))
        self.net = nn.Sequential(nn.Linear(1280, 256), nn.ReLU(), nn.Linear(256, 32),
                                 nn.ReLU(), nn.Linear(32, 256), nn.ReLU(),
                                 nn.Linear(256, 1280))
        opt = torch.optim.Adam(self.net.parameters(), lr=1e-3)
        for _ in range(epochs):
            perm = torch.randperm(len(Xt))
            for i in range(0, len(Xt), 256):
                b = Xt[perm[i:i + 256]]
                opt.zero_grad(); loss = ((self.net(b) - b) ** 2).mean()
                loss.backward(); opt.step()
        self.net.eval()
        self.calibrate(clean_eps)
        return self

    def scores(self, ep):
        import torch
        torch.set_num_threads(1)
        idx = np.arange(len(ep.labels))
        out = np.zeros(len(idx))
        with torch.no_grad():
            for i in range(0, len(idx), 1024):
                X = multiscale_windows(ep, idx[i:i + 1024]).reshape(-1, 1280)
                Xt = torch.from_numpy(((X - self.mu) / self.sd).astype(np.float32))
                out[i:i + 1024] = ((self.net(Xt) - Xt) ** 2).mean(1).numpy()
        return out


class KalmanChi2Detector(_ScoreDetector):
    """Single closed-loop observer over all channels + windowed NIS test.

    Uses the same physical model as PIRCAD but in the classical configuration:
    one estimator re-anchored on every channel, every sample, unconditionally.
    """

    name = "Kalman chi-square"

    def __init__(self, window: int = 30):
        self.window = window

    def fit(self, clean_eps: Sequence):
        self.calibrate(clean_eps)
        return self

    def scores(self, ep):
        from .simulator import frames
        from .twin import PhysicsVesselTwin
        tw = PhysicsVesselTwin()
        sig = {"T_in": 0.05, "P": 25.0, "RH": 0.30}
        nis = np.zeros(len(ep.labels))
        for fr in frames(ep):
            e = tw.predict(fr)
            # innovation against the *full* observer, which uses every channel
            T = tw._T_pred["full"]
            pred = {"T_in": T, "P": tw.P_of_T(T), "RH": tw.RH_of_T(T)}
            nis[fr.k] = sum(((fr.values[c] - pred[c]) / sig[c]) ** 2 for c in sig)
            tw.update(fr, {c: 1.0 for c in CHANNELS})
        return pd.Series(nis).rolling(self.window, min_periods=1).mean().to_numpy()
