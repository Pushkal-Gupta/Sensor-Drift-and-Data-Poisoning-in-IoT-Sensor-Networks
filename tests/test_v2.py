"""
Tests for the journal-evaluation additions (scenarios, fast inference, the
real-data embodiment, and the self-calibrating baselines).

Runs under pytest, or standalone:  python tests/test_v2.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pircad.scenarios import Attacker, Shift, make_episode_v2
from pircad.simulator import make_episode
from pircad.types import RootCause


def test_scenario_factory_reproduces_the_original_bank():
    """With default knobs, every scenario is sample-for-sample identical."""
    for sc in ("clean", "drift", "stealth_poison", "spike_poison",
               "replay_poison", "process_event"):
        a = make_episode_v2(sc, n=1500, seed=11, onset=600, rate=1.2e-3)
        b = make_episode(sc, n=1500, seed=11, onset=600, rate=1.2e-3)
        for c in a.obs:
            assert np.array_equal(a.obs[c], b.obs[c]), (sc, c)
        assert np.array_equal(a.labels, b.labels), sc


def test_mitm_attack_is_drift_plus_the_genuine_signal():
    """An on-path ramp equals benign drift on the attacked channel, exactly.

    This is the observational-equivalence statement the paper relies on: no
    detector can separate the two from that channel's data alone.
    """
    kw = dict(n=1500, seed=3, onset=600, rate=1.5e-3)
    d = make_episode_v2("drift", **kw)
    m = make_episode_v2("stealth_poison", attacker=Attacker(1.0, "matched", "mitm"), **kw)
    assert np.allclose(d.obs["T_in"], m.obs["T_in"])
    assert (m.labels[600:] == int(RootCause.DATA_POISONING)).all()


def test_coordinated_attack_keeps_instrument_constraints_satisfied():
    from pircad.config import PlantParams
    from pircad.simulator import pressure_from_T
    p = PlantParams()
    ep = make_episode_v2("stealth_poison", n=1500, seed=4, onset=600, rate=2e-3,
                         attacker=Attacker(0.02, "white", "coordinated"))
    gas = ep.obs["P"][600:] - pressure_from_T(ep.obs["T_in"][600:], p)
    # gas-law violation stays at the noise level although T_in is forged
    assert abs(float(np.mean(gas))) < 10.0


def test_shift_changes_the_plant_but_not_the_labels():
    a = make_episode_v2("drift", n=1500, seed=5, onset=600, rate=1e-3)
    b = make_episode_v2("drift", n=1500, seed=5, onset=600, rate=1e-3,
                        shift=Shift(noise_scale=3.0))
    assert np.array_equal(a.labels, b.labels)
    assert np.std(np.diff(b.obs["P"][:500])) > 2.0 * np.std(np.diff(a.obs["P"][:500]))


def test_fast_forest_is_numerically_identical_to_sklearn():
    from sklearn.ensemble import RandomForestClassifier
    from pircad.fastforest import FastForest
    rng = np.random.default_rng(0)
    X = rng.normal(size=(2000, 12)); y = (X[:, 0] + X[:, 3] ** 2 > 1).astype(int) + (X[:, 5] > 1)
    rf = RandomForestClassifier(n_estimators=40, max_depth=9, random_state=0).fit(X, y)
    ff = FastForest(rf)
    Xt = rng.normal(size=(300, 12))
    got = np.vstack([ff.predict_proba(x[None]) for x in Xt])
    assert np.allclose(got, rf.predict_proba(Xt), atol=1e-12)


def test_indoor_signatures_separate_instrument_from_local_heating():
    """The psychrometric signatures must be distinct directions.

    A local heat source (T up, RH down at constant e) leaves the Magnus
    constraint intact; a temperature-transducer bias does not.
    """
    from pircad.indoor import IndoorConsistencyMonitor, IndoorTwin
    from pircad.types import Frame
    tw = IndoorTwin(warmup=200)
    mon = IndoorConsistencyMonitor(tw)
    rng = np.random.default_rng(1)
    e = 0.40 * 610.94 * np.exp(17.625 * 22 / (22 + 243.04))
    for k in range(200):
        T = 22 + 0.05 * rng.normal(); RH = 100 * e / (610.94 * np.exp(17.625 * T / (T + 243.04)))
        fr = Frame(k, k, {"T": T, "RH": RH + 0.2 * rng.normal()},
                   {"T_nb": 22 + 0.05 * rng.normal(), "e_nb": e, "RH_nb": 40.0})
        mon.push(mon.evaluate(fr.values, tw.predict(fr), fr.actuation))
        tw.update(fr, {"T": 1.0, "RH": 1.0})
    mon.finalise_baseline(fr.values)
    S = mon.signatures
    assert abs(float(np.dot(S["T"], S["PROCESS"]))) < 0.9
    assert S["PROCESS"][0] == 0.0          # local heating does not move g_mag
    assert S["T"][0] > 0.0                 # a T bias does


def test_page_hinkley_recalibrator_absorbs_a_step():
    from pircad.baselines_adaptive import PageHinkleyRecalibrator
    y = np.r_[np.zeros(800), np.full(800, 2.0)] + 0.05 * np.random.default_rng(0).normal(size=1600)
    tr = PageHinkleyRecalibrator(delta=0.01, lam=5.0, warmup=400).run(y)
    assert abs(tr.offset[-1] - 2.0) < 0.2


def test_reattribution_policy_switch():
    """Sustained drift evidence releases a quarantined channel only if allowed."""
    from pircad.governor import AdaptationGovernor, ChannelState
    for allow, expect in ((True, ChannelState.DRIFTING), (False, ChannelState.QUARANTINED)):
        g = AdaptationGovernor(["T"], freeze_hold=50, clean_to_resume=20,
                               allow_reattribution=allow)
        g.note_decision(0, "T", RootCause.DATA_POISONING, 0.9, triggered=True)
        for k in range(1, 200):
            g.note_decision(k, "T", RootCause.SENSOR_DRIFT, 0.9, triggered=True)
        assert g.state("T") is expect, (allow, g.state("T"))


def test_single_drift_verdict_never_releases_quarantine():
    from pircad.governor import AdaptationGovernor, ChannelState
    g = AdaptationGovernor(["T"], freeze_hold=50, clean_to_resume=20,
                           allow_reattribution=True)
    g.note_decision(0, "T", RootCause.DATA_POISONING, 0.9, triggered=True)
    g.note_decision(500, "T", RootCause.SENSOR_DRIFT, 0.9, triggered=True)
    assert g.state("T") is ChannelState.QUARANTINED


def _main() -> int:
    fns = [(k, v) for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    failed = 0
    for name, fn in fns:
        try:
            fn()
            print(f"  PASS  {name}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL  {name}: {e}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"  ERROR {name}: {type(e).__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_main())
