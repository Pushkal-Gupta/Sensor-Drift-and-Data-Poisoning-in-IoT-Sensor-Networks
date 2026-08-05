"""
E5 -- Root-cause disambiguation and confidence.

Input: the residual-character features (E2) for the triggering channel, plus the
consistency localisation (E4).  Output: one of
{NORMAL, SENSOR_DRIFT, DATA_POISONING, PROCESS_EVENT, UNKNOWN} together with a
calibrated confidence that drives escalation (E6) and action (E8).

Two interchangeable embodiments:

RuleRootCauseClassifier   transparent, needs no training data, and states the
                          physical argument explicitly.  Suitable for a fog node
                          with no historical fault corpus.
MLRootCauseClassifier     the primary embodiment: any supervised classifier over
                          the same feature vector.  The invention does not depend
                          on which -- random forest, gradient boosting or a TinyML
                          model are all embodiments.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .consistency import PROCESS_HYPOTHESIS, Localisation
from .residual import FEATURE_NAMES
from .types import RootCause

#: features handed to the classifier: residual character + localisation context
LOC_FEATURES: List[str] = ["loc_score", "loc_margin", "loc_magnitude",
                           "loc_is_self", "loc_is_process"]
ALL_FEATURES: List[str] = FEATURE_NAMES + LOC_FEATURES

CLASSES: Tuple[RootCause, ...] = (RootCause.NORMAL, RootCause.SENSOR_DRIFT,
                                  RootCause.DATA_POISONING, RootCause.PROCESS_EVENT)


def build_vector(channel: str, feats: Dict[str, float],
                 loc: Localisation) -> np.ndarray:
    """Assemble the classifier input for one channel of one frame."""
    v = [float(feats.get(name, 0.0)) for name in FEATURE_NAMES]
    v.append(float(loc.score))
    v.append(float(loc.margin))
    v.append(float(np.log1p(max(0.0, loc.magnitude))))
    v.append(1.0 if loc.hypothesis == channel else 0.0)
    v.append(1.0 if loc.hypothesis == PROCESS_HYPOTHESIS else 0.0)
    return np.asarray(v, dtype=float)


@dataclass
class RootCauseResult:
    cause: RootCause
    confidence: float
    scores: Dict[RootCause, float]
    rationale: str = ""


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


class RuleRootCauseClassifier:
    """Interpretable embodiment.  Every term is a physical statement.

    poisoning evidence
        the reported stream stopped following the modelled micro-dynamics
        (dyn_coherence), the deviation began with a discontinuity a physical
        ageing process cannot produce (corner, jump_max), it grows faster than the
        transducer's ageing envelope (slope_ratio), or the channel's noise
        signature changed (acf_shift, var_ratio, flatness).
    drift evidence
        smooth, one-directional, inside the ageing envelope, and the transducer's
        own noise signature and dynamic response are intact.
    """

    def __init__(self, process_score: float = 0.55, trigger_floor: float = 2.5):
        self.process_score = process_score
        self.trigger_floor = trigger_floor

    def classify(self, channel: str, feats: Dict[str, float],
                 loc: Localisation) -> RootCauseResult:
        f = feats

        # 0) is anything actually there?
        strength = max(abs(f.get("r_mean_norm", 0.0)),
                       abs(f.get("z_now", 0.0)) / 2.0,
                       abs(f.get("slope_norm", 0.0)))
        if strength < self.trigger_floor and loc.magnitude < 6.0:
            return RootCauseResult(RootCause.NORMAL, 0.75,
                                   {RootCause.NORMAL: 1.0}, "below evidence floor")

        # 1) instrument side or process side?
        if (loc.hypothesis == PROCESS_HYPOTHESIS
                and loc.score >= self.process_score and loc.margin > 0.05):
            conf = float(np.clip(0.55 + 0.8 * loc.margin, 0.5, 0.97))
            return RootCauseResult(
                RootCause.PROCESS_EVENT, conf,
                {RootCause.PROCESS_EVENT: conf},
                "constraint violations confined to the dynamic/energy balance; "
                "instrument constraints intact")

        # 2) drift versus poisoning, on residual character
        poison = (
            3.0 * float(np.clip(f.get("dyn_coherence", 0.0), 0.0, 1.0))
            + 1.0 * float(np.clip(f.get("corner", 0.0) / 8.0 - 1.0, -1.0, 2.0))
            + 1.0 * float(np.clip(f.get("jump_max", 0.0) / 6.0 - 1.0, -1.0, 3.0))
            + 1.2 * float(np.clip(f.get("slope_ratio", 0.0) - 1.0, -1.0, 2.0))
            + 1.2 * float(np.clip(abs(f.get("acf_shift", 0.0)) / 0.25 - 1.0, -1.0, 2.0))
            + 0.8 * float(np.clip(abs(f.get("var_ratio", 0.0)) / 0.4 - 1.0, -1.0, 2.0))
            + 1.5 * float(np.clip(f.get("exceed_frac", 0.0) * 4.0 - 0.5, -0.5, 2.0))
            + 0.5 * float(np.clip(f.get("kurt_incr", 0.0) / 3.0, -1.0, 2.0))
        )
        drift = (
            2.5 * float(np.clip((f.get("monotone_frac", 0.5) - 0.52) * 6.0, -1.0, 2.0))
            + 1.5 * float(np.clip(1.0 - f.get("slope_ratio", 0.0), -1.0, 1.0))
            + 1.5 * float(np.clip(1.0 - 3.0 * abs(f.get("dyn_coherence", 0.0)), -1.0, 1.0))
            + 1.0 * float(np.clip(1.0 - abs(f.get("acf_shift", 0.0)) / 0.25, -1.0, 1.0))
            + 0.8 * float(np.clip(1.0 - abs(f.get("var_ratio", 0.0)) / 0.4, -1.0, 1.0))
        )
        p = _sigmoid(0.9 * (poison - drift))
        cause = RootCause.DATA_POISONING if p >= 0.5 else RootCause.SENSOR_DRIFT
        conf = float(max(p, 1.0 - p))
        return RootCauseResult(
            cause, conf,
            {RootCause.DATA_POISONING: p, RootCause.SENSOR_DRIFT: 1.0 - p},
            f"poison_evidence={poison:.2f} drift_evidence={drift:.2f}")


class MLRootCauseClassifier:
    """Supervised embodiment over the same feature vector."""

    def __init__(self, model=None, classes: Optional[List[RootCause]] = None):
        self.model = model
        self.classes = classes or list(CLASSES)

    @staticmethod
    def default_model(seed: int = 0):
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(
            n_estimators=150, max_depth=12, min_samples_leaf=8,
            class_weight="balanced_subsample", n_jobs=-1, random_state=seed,
        )

    def fit(self, X: np.ndarray, y: np.ndarray, seed: int = 0) -> "MLRootCauseClassifier":
        if self.model is None:
            self.model = self.default_model(seed)
        self.model.fit(X, y)
        self.classes = [RootCause(int(c)) for c in self.model.classes_]
        return self

    def classify(self, channel: str, feats: Dict[str, float],
                 loc: Localisation) -> RootCauseResult:
        if self.model is None:
            raise RuntimeError("MLRootCauseClassifier used before fit()")
        x = build_vector(channel, feats, loc).reshape(1, -1)
        proba = self.model.predict_proba(x)[0]
        order = np.argsort(proba)[::-1]
        top = int(order[0])
        cause = RootCause(int(self.model.classes_[top]))
        conf = float(proba[top])
        scores = {RootCause(int(c)): float(pr)
                  for c, pr in zip(self.model.classes_, proba)}
        return RootCauseResult(cause, conf, scores, "supervised root-cause model")


class HybridRootCauseClassifier:
    """Rule stage arbitrates instrument-vs-process; ML stage arbitrates the cause.

    This is the embodiment used in the reported results: the physical hypothesis
    test is authoritative for process-side excursions (where being wrong is a
    safety issue), and the learned model resolves drift-versus-poisoning.
    """

    def __init__(self, ml: MLRootCauseClassifier,
                 rules: Optional[RuleRootCauseClassifier] = None,
                 process_override: float = 0.75,
                 instrument_override_margin: float = 0.15):
        self.ml = ml
        self.rules = rules or RuleRootCauseClassifier()
        self.process_override = process_override
        self.instrument_override_margin = instrument_override_margin

    def classify(self, channel: str, feats: Dict[str, float],
                 loc: Localisation) -> RootCauseResult:
        # (a) the physical hypothesis test is authoritative for process-side
        if (loc.hypothesis == PROCESS_HYPOTHESIS
                and loc.score >= self.process_override and loc.margin > 0.10):
            conf = float(np.clip(0.60 + 0.7 * loc.margin, 0.5, 0.97))
            return RootCauseResult(RootCause.PROCESS_EVENT, conf,
                                   {RootCause.PROCESS_EVENT: conf},
                                   "physical hypothesis test: process-side")

        r = self.ml.classify(channel, feats, loc)

        # (b) and its converse.  A genuine process excursion leaves every
        # instrument mutually consistent; it cannot be explained by a bias on a
        # single channel.  So when the constraint violations localise cleanly to
        # *this* instrument, a process verdict is physically incoherent and the
        # best instrument-side cause is taken instead.
        if (r.cause is RootCause.PROCESS_EVENT
                and loc.hypothesis == channel
                and loc.margin > self.instrument_override_margin):
            alt = {c: p for c, p in r.scores.items()
                   if c is not RootCause.PROCESS_EVENT}
            if alt:
                cause = max(alt, key=alt.get)
                total = sum(alt.values()) or 1.0
                return RootCauseResult(
                    cause, float(alt[cause] / total), r.scores,
                    "process verdict overruled: violations localise to this "
                    "instrument")
        return r
