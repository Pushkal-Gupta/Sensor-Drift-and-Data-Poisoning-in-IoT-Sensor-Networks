"""Shared harness for the journal evaluation (experiments 6-10).

Differences from `_common.py`, which experiments 1-5 keep using unchanged:

* models are refitted under the installed scikit-learn and cached separately
  (`artifacts/v2_models.joblib`), with six training seeds instead of three;
* single-sample inference runs with n_jobs=1 -- a thread pool per prediction
  dominated runtime by ~5x without changing a single output;
* episodes are described by small picklable specs and evaluated in a process
  pool, so 30-seed banks are affordable;
* several root-cause back-ends are available, to show the result is a property
  of the architecture rather than of the random forest.
"""

from __future__ import annotations

import os

# One BLAS/OpenMP thread per worker process: the pool supplies the parallelism,
# and per-call OpenMP pools (HistGradientBoosting) otherwise oversubscribe the
# machine by an order of magnitude.  Must precede the numpy/sklearn imports.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import warnings  # noqa: E402

warnings.filterwarnings("ignore")

from pircad.scenarios import (DEFAULT_ATTACKER, NO_SHIFT, Attacker,  # noqa: E402
                              Shift, make_episode_v2)

ARTIFACTS = ROOT / "artifacts"
FIGURES = ROOT / "figures"
ARTIFACTS.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)

TRAIN_SEEDS = (0, 1, 2, 3, 4, 5)
EVAL_SEEDS = tuple(range(900, 930))          # 30 held-out seeds
EPISODE_LEN = 4000
WARMUP = 600
SCENARIOS = ["clean", "drift", "stealth_poison", "spike_poison",
             "replay_poison", "process_event"]
N_WORKERS = max(1, min(16, (os.cpu_count() or 2) - 2))
BACKENDS = ("rf", "hgb", "logreg", "mlp", "rules")
#: a forwarded sample counts as useful if within this many units of the truth
USEFUL_TOL = 0.5


# --------------------------------------------------------------------------
# episode specs
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Spec:
    scenario: str
    seed: int
    onset: int
    rate: float
    attacker: Attacker = DEFAULT_ATTACKER
    shift: Shift = NO_SHIFT
    n: int = EPISODE_LEN
    sign: int = 1

    def build(self):
        return make_episode_v2(self.scenario, n=self.n, seed=self.seed,
                               onset=self.onset, rate=self.rate,
                               attacker=self.attacker, shift=self.shift,
                               sign=self.sign)


def _sign(seed_index: int) -> int:
    """Event polarity, alternating with the seed so the bank is exactly balanced.

    Drawing it at random looked equivalent and was not: with six training seeds
    every drift episode came out downward by chance, and the model learned
    "upward drift is normal".
    """
    return 1 if seed_index % 2 == 0 else -1


def eval_specs(scenarios: Sequence[str] = SCENARIOS,
               seeds: Sequence[int] = EVAL_SEEDS,
               rate_range=(0.8e-3, 2.5e-3),
               attacker: Attacker = DEFAULT_ATTACKER,
               shift: Shift = NO_SHIFT, n: int = EPISODE_LEN) -> List[Spec]:
    """Held-out bank: onset/rate drawn as in experiments 1-5, polarity balanced."""
    out = []
    for s in seeds:
        for j, sc in enumerate(scenarios):
            j0 = SCENARIOS.index(sc)
            rng = np.random.default_rng(7919 * s + j0)
            onset = int(rng.integers(n // 4, n // 2))
            rate = float(rng.uniform(0.8e-3, 2.5e-3))
            if tuple(rate_range) != (0.8e-3, 2.5e-3):
                rate = float(np.random.default_rng(31 * s + j0)
                             .uniform(*rate_range))
            out.append(Spec(sc, int(7919 * s + j0), onset, rate, attacker,
                            shift, n, _sign(s)))
    return out


def train_specs(seeds: Sequence[int] = TRAIN_SEEDS, n: int = EPISODE_LEN) -> List[Spec]:
    """`pircad.training.training_episodes`, each in both event polarities."""
    out = []
    for s in seeds:
        for j, sc in enumerate(SCENARIOS):
            rng = np.random.default_rng(1000 * s + j)
            onset = int(rng.integers(n // 4, n // 2))
            rate = float(rng.uniform(0.8e-3, 2.5e-3))
            # both polarities of every training episode
            for sg in (1, -1):
                out.append(Spec(sc, int(1000 * s + j), onset, rate, n=n, sign=sg))
    return out


# --------------------------------------------------------------------------
# parallel map
# --------------------------------------------------------------------------
def pmap(fn: Callable, items: Sequence, desc: str = "", workers: int = N_WORKERS):
    from concurrent.futures import as_completed
    t0 = time.time()
    out = [None] * len(items)
    if workers <= 1:
        for i, it in enumerate(items):
            out[i] = fn(it)
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(fn, it): i for i, it in enumerate(items)}
            done = 0
            for f in as_completed(futs):
                out[futs[f]] = f.result()
                done += 1
                if done % max(1, len(items) // 10) == 0 or done == len(items):
                    print(f"  [{desc}] {done}/{len(items)}  "
                          f"{time.time() - t0:6.0f}s", flush=True)
    return out


# --------------------------------------------------------------------------
# models
# --------------------------------------------------------------------------
def _collect(spec: Spec):
    from pircad.training import collect_episode
    X, y, _ = collect_episode(spec.build())
    return np.vstack(X) if X else np.zeros((0, 22)), np.asarray(y, dtype=int)


def _make_estimator(name: str, seed: int = 0):
    from sklearn.ensemble import (HistGradientBoostingClassifier,
                                  RandomForestClassifier)
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    if name == "rf":
        return RandomForestClassifier(n_estimators=150, max_depth=12,
                                      min_samples_leaf=8,
                                      class_weight="balanced_subsample",
                                      n_jobs=-1, random_state=seed)
    if name == "hgb":
        return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08,
                                              class_weight="balanced",
                                              random_state=seed)
    if name == "logreg":
        return make_pipeline(StandardScaler(),
                             LogisticRegression(max_iter=3000, C=1.0,
                                                class_weight="balanced"))
    if name == "mlp":
        return make_pipeline(StandardScaler(),
                             MLPClassifier(hidden_layer_sizes=(64, 32),
                                           max_iter=300, early_stopping=True,
                                           random_state=seed))
    raise ValueError(name)


def _single_thread(est) -> None:
    if hasattr(est, "n_jobs"):
        est.n_jobs = 1
    for step in getattr(est, "steps", []):
        _single_thread(step[1])


def get_v2_models(force: bool = False) -> Dict:
    import joblib
    cache = ARTIFACTS / "v2_models.joblib"
    if cache.exists() and not force:
        return joblib.load(cache)

    print(f"[v2] collecting training corpus from {len(train_specs())} episodes")
    parts = pmap(_collect, train_specs(), "corpus")
    X = np.vstack([p[0] for p in parts if len(p[1])])
    y = np.concatenate([p[1] for p in parts if len(p[1])])
    print(f"[v2] corpus {X.shape}  classes {np.bincount(y)}")

    from pircad.rootcause import MLRootCauseClassifier
    blob = {"X": X, "y": y, "backends": {}}
    for name in ("rf", "hgb", "logreg", "mlp"):
        t0 = time.time()
        est = _make_estimator(name)
        clf = MLRootCauseClassifier(model=est).fit(X, y)
        _single_thread(clf.model)
        blob["backends"][name] = clf
        print(f"[v2] fitted {name:7s} {time.time() - t0:5.1f}s")

    from pircad.baselines import Gen1FHAD
    t0 = time.time()
    gen1 = Gen1FHAD().fit([s.build() for s in train_specs()])
    _single_thread(gen1.model)
    blob["gen1"] = gen1
    print(f"[v2] fitted gen1    {time.time() - t0:5.1f}s")

    joblib.dump(blob, cache)
    print(f"[v2] cached at {cache}")
    return blob


_WORKER_MODELS: Dict = {}


def worker_models() -> Dict:
    if not _WORKER_MODELS:
        blob = get_v2_models()
        idx = np.random.default_rng(0).choice(len(blob["X"]), 500, replace=False)
        accelerate(blob["backends"]["rf"], blob["X"][idx])
        _WORKER_MODELS.update(blob)
    return _WORKER_MODELS


def accelerate(clf, check: Optional[np.ndarray] = None):
    """Swap a fitted RF back-end's model for the verified FastForest."""
    from sklearn.ensemble import RandomForestClassifier
    if isinstance(clf.model, RandomForestClassifier):
        clf.model = FastForest(clf.model, check=check)
    return clf


def make_classifier(backend: str):
    from pircad.rootcause import (HybridRootCauseClassifier,
                                  RuleRootCauseClassifier)
    if backend == "rules":
        return RuleRootCauseClassifier()
    return HybridRootCauseClassifier(worker_models()["backends"][backend])


# --------------------------------------------------------------------------
# PIRCAD run with effect-level metrics
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class RunCfg:
    backend: str = "rf"
    param_error: float = 1.0        # twin mismatch multiplier
    differential: bool = True
    twin: str = "physics"           # "physics" | "rolling"
    character: bool = True          # residual-character features available
    consistency: bool = True
    reattribution: bool = False     # governor policy (see experiment 9)


def run_pircad(args) -> Dict[str, float]:
    spec, rc = args
    from pircad.config import CHANNELS, FogConfig
    from pircad.consistency import NullConsistencyMonitor
    from pircad.fognode import PIRCADFogNode
    from pircad.metrics import evaluate_run
    from pircad.simulator import frames
    from pircad.twin import PhysicsVesselTwin, RollingMeanTwin

    ep = spec.build()
    cfg = FogConfig()
    cfg.differential_response = rc.differential
    cfg.allow_reattribution = rc.reattribution
    twin = (PhysicsVesselTwin(param_error=rc.param_error) if rc.twin == "physics"
            else RollingMeanTwin(CHANNELS))
    kw = {}
    if not rc.consistency:
        kw["consistency"] = NullConsistencyMonitor()
    node = PIRCADFogNode(twin=twin, classifier=make_classifier(rc.backend),
                         cfg=cfg, **kw)
    res = node.run(frames(ep), truth_labels=ep.labels)
    m = evaluate_run(ep, res, WARMUP)
    m.update(effect_metrics(ep, res))
    m["scenario"] = spec.scenario
    m["seed"] = spec.seed
    m["macro_f1"] = macro_f1_episode(ep.labels[WARMUP:], res.causes()[WARMUP:])
    m.update(detection_f1(ep.labels, res.causes() != 0))
    m["_pred"] = res.causes().astype(np.int8)
    return m


def effect_metrics(ep, res) -> Dict[str, float]:
    """What reaches the cloud, compared with the physical truth.

    `fwd_err` is the mean absolute error of the *forwarded, corrected* value of
    the attacked/drifting channel against its true physical value, over the
    event window.  It is the quantity downstream analytics actually consume,
    and it lets an attack that is misclassified but harmlessly corrected be told
    apart from one that is misclassified and absorbed.
    """
    out: Dict[str, float] = {}
    ch = ep.target_channel
    if ch is None:
        return out
    d = res.per_channel[ch]
    off = np.array([x.calibration_offset for x in d])
    fwd = np.array([x.forwarded for x in d])
    corrected = ep.obs[ch] - off
    s = int(ep.meta["onset"])
    win = slice(s, len(ep.labels))
    err = np.abs(corrected[win] - ep.truth[ch][win])
    f = fwd[win]
    out["fwd_frac"] = float(f.mean())
    out["fwd_err_mean"] = float(err[f].mean()) if f.any() else 0.0
    # integrated harm: error of what was forwarded, zero for withheld samples
    out["harm"] = float(np.where(f, err, 0.0).mean())
    # availability: forwarded AND within tolerance of the truth
    out["useful_frac"] = float((f & (err < USEFUL_TOL)).mean())
    out["final_offset"] = float(off[-1])
    return out


def detection_f1(y_true: np.ndarray, anom_pred: np.ndarray) -> Dict[str, float]:
    from pircad.types import RootCause
    t = np.asarray(y_true)[WARMUP:] != int(RootCause.NORMAL)
    p = np.asarray(anom_pred)[WARMUP:].astype(bool)
    tp = float((t & p).sum()); fp = float((~t & p).sum()); fn = float((t & ~p).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) and np.isfinite(rec) else (
        float("nan") if not np.isfinite(rec) else 0.0)
    far = float(p[~t].mean()) if (~t).any() else float("nan")
    return {"detection_f1": f1, "detection_recall": rec, "false_alarm_rate": far}


def macro_f1_episode(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    from sklearn.metrics import f1_score
    labels = sorted(set(int(v) for v in np.unique(y_true)))
    return float(f1_score(y_true, y_pred, labels=labels, average="macro",
                          zero_division=0))


# --------------------------------------------------------------------------
# statistics
# --------------------------------------------------------------------------
def bootstrap_ci(x: Sequence[float], n_boot: int = 4000, alpha: float = 0.05,
                 seed: int = 0):
    x = np.asarray([v for v in x if np.isfinite(v)], dtype=float)
    if x.size == 0:
        return (float("nan"),) * 3
    rng = np.random.default_rng(seed)
    bs = rng.choice(x, size=(n_boot, x.size), replace=True).mean(axis=1)
    return float(x.mean()), float(np.quantile(bs, alpha / 2)), float(np.quantile(bs, 1 - alpha / 2))


def wilcoxon(a: Sequence[float], b: Sequence[float]) -> Dict[str, float]:
    """Paired two-sided Wilcoxon signed-rank test plus rank-biserial effect size."""
    from scipy.stats import wilcoxon as _w
    a = np.asarray(a, float); b = np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    d = a - b
    if d.size == 0 or np.allclose(d, 0):
        return {"p": 1.0, "n": int(d.size), "r_rb": 0.0, "median_diff": 0.0}
    st = _w(a, b, zero_method="wilcox", alternative="two-sided")
    nz = d[d != 0]
    ranks = np.argsort(np.argsort(np.abs(nz))) + 1
    rpos = ranks[nz > 0].sum(); rneg = ranks[nz < 0].sum()
    return {"p": float(st.pvalue), "n": int(d.size),
            "r_rb": float((rpos - rneg) / (rpos + rneg)),
            "median_diff": float(np.median(d))}


def holm(pvals: Dict[str, float]) -> Dict[str, float]:
    keys = sorted(pvals, key=pvals.get)
    m = len(keys); out = {}; running = 0.0
    for i, k in enumerate(keys):
        running = max(running, min(1.0, (m - i) * pvals[k]))
        out[k] = running
    return out


def strip_private(rows: List[Dict]) -> List[Dict]:
    return [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows]


def save_json(name: str, obj) -> Path:
    p = ARTIFACTS / name
    p.write_text(json.dumps(obj, indent=2, default=lambda o: (
        asdict(o) if hasattr(o, "__dataclass_fields__") else float(o))))
    print(f"[saved] {p}")
    return p


def banner(text: str) -> None:
    print("\n" + "=" * 78 + "\n" + text + "\n" + "=" * 78, flush=True)


# --------------------------------------------------------------------------
# data-driven baselines, fitted once and shared by experiments 6-8
# --------------------------------------------------------------------------
def get_ml_baselines(force: bool = False) -> Dict:
    import joblib
    cache = ARTIFACTS / "v2_ml_baselines.joblib"
    if cache.exists() and not force:
        return joblib.load(cache)
    from pircad.baselines_ml import GenericFeatureRF, MultiScaleCNN
    train_eps = [s.build() for s in train_specs()]
    out = {}
    for cls in (GenericFeatureRF, MultiScaleCNN):
        t0 = time.time()
        out[cls.name] = cls().fit(train_eps)
        print(f"[v2] fitted {cls.name} in {time.time() - t0:.0f}s", flush=True)
    joblib.dump(out, cache)
    return out


def score_baseline(spec: Spec, ep, y: np.ndarray) -> Dict[str, float]:
    """Score a label-only system with the same metrics as PIRCAD.

    A label-only system has no recalibration, so what it forwards is the raw
    reading, and (most favourably to it) it withholds every flagged sample.
    """
    from pircad.metrics import evaluate_labels
    from pircad.types import RootCause
    m = evaluate_labels(ep, y, WARMUP)
    m.update(detection_f1(ep.labels, np.asarray(y) != int(RootCause.NORMAL)))
    m["macro_f1"] = macro_f1_episode(ep.labels[WARMUP:], np.asarray(y)[WARMUP:])
    ch = ep.target_channel
    if ch is not None:
        s = int(ep.meta["onset"])
        f = np.asarray(y)[s:] == int(RootCause.NORMAL)
        err = np.abs(ep.obs[ch][s:] - ep.truth[ch][s:])
        m["fwd_frac"] = float(f.mean())
        m["fwd_err_mean"] = float(err[f].mean()) if f.any() else 0.0
        m["harm"] = float(np.where(f, err, 0.0).mean())
        m["useful_frac"] = float((f & (err < USEFUL_TOL)).mean())
    m["scenario"] = spec.scenario
    m["seed"] = spec.seed
    return m


# --------------------------------------------------------------------------
# fast single-sample random-forest inference (lives in the package; re-exported)
# --------------------------------------------------------------------------
from pircad.fastforest import FastForest  # noqa: E402,F401
