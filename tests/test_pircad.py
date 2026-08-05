"""
Unit tests for the PIRCAD elements.

Runs under pytest, or standalone:  python tests/test_pircad.py
Each test checks a property the claims actually rely on, not just that the code
executes.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pircad.actions import CalibrationManager
from pircad.config import CHANNELS, FogConfig, PlantParams
from pircad.consistency import PROCESS_HYPOTHESIS, VesselConsistencyMonitor
from pircad.escalation import EscalationService
from pircad.fognode import PIRCADFogNode
from pircad.governor import AdaptationGovernor, ChannelState
from pircad.quantum import fidelity_kernel
from pircad.simulator import frames, make_episode
from pircad.triggers import ChannelTrigger, TwoSidedCUSUM
from pircad.twin import PhysicsVesselTwin
from pircad.types import Frame, RootCause


# ---------------------------------------------------------------- E1: twin
def test_leave_one_out_residual_sees_its_own_channel_bias():
    """A bias on a channel must appear in that channel's residual at full size.

    This is the property a single closed-loop estimator does not have, and the
    reason the twin is structured as a dedicated-observer scheme.
    """
    ep = make_episode("clean", n=2500, seed=5)
    bias = 1.0
    for biased in (False, True):
        tw = PhysicsVesselTwin()
        res = []
        for f in frames(ep):
            if biased and f.k >= 1200:
                f.values["T_in"] += bias
            p = tw.predict(f)
            res.append(f.values["T_in"] - p["T_in"])
            tw.update(f, {c: 1.0 for c in CHANNELS})
        tail = float(np.mean(res[-400:]))
        if not biased:
            base = tail
        else:
            recovered = tail - base
    assert 0.7 * bias < recovered < 1.3 * bias, recovered


def test_twin_does_not_diverge_when_every_channel_is_distrusted():
    ep = make_episode("clean", n=1500, seed=6)
    tw = PhysicsVesselTwin()
    for f in frames(ep):
        tw.predict(f)
        tw.update(f, {c: 0.0 for c in CHANNELS})
    err = abs(tw.T_est - ep.truth["T_in"][-1])
    assert err < 2.0, f"unanchored estimator drifted by {err:.2f} K"


# ------------------------------------------------------------ E2: residual
def test_residual_is_unit_normalised_on_clean_data():
    ep = make_episode("clean", n=6000, seed=7)
    node = PIRCADFogNode(classifier=None)
    res = node.run(frames(ep))
    for ch in CHANNELS:
        z = np.array([d.residual_norm for d in res.per_channel[ch]])[1200:]
        assert abs(z.mean()) < 0.6, f"{ch} residual mean {z.mean():.2f}"
        assert 0.6 < z.std() < 1.7, f"{ch} residual std {z.std():.2f}"


def test_residual_scale_is_frozen_after_warmup():
    """An adaptive scale is the handle a slow injection uses to hide."""
    ep = make_episode("stealth_poison", n=4000, seed=8)
    node = PIRCADFogNode(classifier=None)
    node.run(frames(ep))
    eng = node.res["T_in"]
    before = eng.base.sigma
    for _ in range(2000):
        eng.refresh_baseline(50.0, allowed=True)
    assert eng.base.sigma == before


# ------------------------------------------------------------- E3: trigger
def test_cusum_signals_on_a_shift_and_restarts():
    c = TwoSidedCUSUM(k=0.6, h=12.0)
    for _ in range(200):
        assert c.update(0.0) < 12.0
    fired = None
    for i in range(200):
        if c.update(2.0) >= 12.0:
            fired = i
            break
    assert fired is not None and fired < 40
    c.reset()
    assert c.update(0.0) == 0.0


def test_trigger_rate_is_low_on_clean_data():
    ep = make_episode("clean", n=6000, seed=9)
    node = PIRCADFogNode(classifier=None)
    res = node.run(frames(ep))
    rate = np.mean([d.triggered for d in res.per_channel["T_in"]][1200:])
    assert rate < 0.10, f"nominal trigger rate {rate:.3f} is too high"


# --------------------------------------------------------- E4: consistency
def test_signatures_are_unit_vectors_and_process_is_instrument_free():
    ep = make_episode("clean", n=1500, seed=10)
    node = PIRCADFogNode(classifier=None)
    node.run(frames(ep))
    sig = node.consistency.signatures
    assert set(sig) == {"T_in", "T_amb", "P", "RH", PROCESS_HYPOTHESIS}
    for name, v in sig.items():
        assert abs(np.linalg.norm(v) - 1.0) < 1e-9, name
    # a process perturbation must leave the two instrument constraints untouched
    proc = sig[PROCESS_HYPOTHESIS]
    names = list(VesselConsistencyMonitor.CONSTRAINTS)
    assert proc[names.index("g_gas")] == 0.0
    assert proc[names.index("g_humidity")] == 0.0


def test_localisation_recovers_the_injected_hypothesis():
    ep = make_episode("clean", n=1500, seed=11)
    node = PIRCADFogNode(classifier=None)
    node.run(frames(ep))
    mon = node.consistency
    for hyp, col in mon.signatures.items():
        for c, name in zip(col, mon.CONSTRAINTS):
            mon.buf[name].clear()
            for _ in range(mon.window):
                mon.buf[name].append(mon.mean[name] + 40.0 * c * mon.sigma[name])
        loc = mon.localise()
        assert loc.hypothesis == hyp, f"{hyp} localised as {loc.hypothesis}"
        assert loc.score > 0.95


# ----------------------------------------------------------- E7: governor
def test_quarantine_freezes_adaptation_and_releases_only_after_a_clean_run():
    g = AdaptationGovernor(["a"], freeze_hold=50, clean_to_resume=20)
    assert g.adaptation_allowed("a")
    g.note_trigger("a", True)
    assert not g.adaptation_allowed("a")           # de-weighted before any verdict
    g.note_decision(0, "a", RootCause.DATA_POISONING, 0.9, triggered=True)
    assert g.state("a") is ChannelState.QUARANTINED
    assert g.trust_weight("a") == 0.0
    assert g.training_frozen()
    assert not g.admit_for_training("a")

    for k in range(1, 49):                          # hold not yet served
        g.note_decision(k, "a", RootCause.NORMAL, 1.0)
    assert g.state("a") is ChannelState.QUARANTINED
    g.note_decision(60, "a", RootCause.NORMAL, 1.0)
    assert g.state("a") is ChannelState.RECOVERING
    for k in range(61, 61 + 19):
        g.note_decision(k, "a", RootCause.NORMAL, 1.0)
    assert g.state("a") is ChannelState.RECOVERING
    g.note_decision(200, "a", RootCause.NORMAL, 1.0)
    assert g.state("a") is ChannelState.TRUSTED
    assert g.adaptation_allowed("a")


def test_a_trigger_during_recovery_restarts_the_clean_run():
    g = AdaptationGovernor(["a"], freeze_hold=5, clean_to_resume=10)
    g.note_decision(0, "a", RootCause.DATA_POISONING, 0.9, triggered=True)
    g.note_decision(10, "a", RootCause.NORMAL, 1.0)
    assert g.state("a") is ChannelState.RECOVERING
    for k in range(11, 19):
        g.note_decision(k, "a", RootCause.NORMAL, 1.0)
    g.note_decision(19, "a", RootCause.NORMAL, 1.0, triggered=True)
    for k in range(20, 28):
        g.note_decision(k, "a", RootCause.NORMAL, 1.0)
    assert g.state("a") is ChannelState.RECOVERING


def test_operator_acknowledgement_is_required_when_configured():
    g = AdaptationGovernor(["a"], freeze_hold=2, clean_to_resume=3,
                           require_operator_ack=True)
    g.note_decision(0, "a", RootCause.DATA_POISONING, 0.9, triggered=True)
    for k in range(3, 40):
        g.note_decision(k, "a", RootCause.NORMAL, 1.0)
    assert g.state("a") is ChannelState.RECOVERING
    g.operator_ack("a")
    for k in range(40, 50):
        g.note_decision(k, "a", RootCause.NORMAL, 1.0)
    assert g.state("a") is ChannelState.TRUSTED


# ------------------------------------------------------------- E8: actions
def test_recalibration_respects_the_rate_limit_and_the_cap():
    cfg = FogConfig()
    cal = CalibrationManager(["T_in"], cfg)
    for k in range(5000):
        d = cal.correct(k, "T_in", 100.0, 0.9)
        assert abs(d) <= cfg.recal_rate_limit + 1e-12
    assert abs(cal.total("T_in")) <= cfg.recal_total_cap["T_in"] + 1e-9
    assert cal.saturated("T_in")
    assert len(cal.audit) > 0


# ---------------------------------------------------------- E6: escalation
def test_escalation_is_asynchronous_and_settles_at_the_due_sample():
    class Stub:
        name = "stub"

        def classify(self, X):
            n = len(X)
            return (np.full(n, int(RootCause.DATA_POISONING)), np.full(n, 0.9))

    esc = EscalationService(backend=Stub(), latency_samples=25)
    t = esc.submit(10, "T_in", np.zeros(22), RootCause.SENSOR_DRIFT, 0.4)
    assert t is not None
    assert esc.poll(20) == []
    v = esc.poll(35)
    assert len(v) == 1 and v[0].cause is RootCause.DATA_POISONING
    assert esc.overridden == 1


def test_escalation_overflow_is_counted_not_silently_dropped():
    class Stub:
        name = "stub"

        def classify(self, X):
            return (np.zeros(len(X), dtype=int), np.ones(len(X)))

    esc = EscalationService(backend=Stub(), latency_samples=10_000, max_in_flight=3)
    for k in range(6):
        esc.submit(k, "T_in", np.zeros(22), RootCause.NORMAL, 0.1)
    assert esc.submitted == 3 and esc.dropped_overflow == 3


# ------------------------------------------------------------ quantum stage
def test_quantum_kernel_is_a_valid_gram_matrix():
    rng = np.random.default_rng(0)
    X = rng.random((12, 4)) * np.pi
    K = fidelity_kernel(X, None, n_qubits=4, reps=2)
    assert np.allclose(np.diag(K), 1.0)
    assert np.allclose(K, K.T)
    assert K.min() >= -1e-12 and K.max() <= 1.0 + 1e-12
    assert np.linalg.eigvalsh(K).min() > -1e-8      # positive semi-definite


# -------------------------------------------------------------- end to end
def test_clean_episode_raises_no_security_alert_and_keeps_forwarding():
    ep = make_episode("clean", n=4000, seed=12)
    node = PIRCADFogNode(classifier=None)
    res = node.run(frames(ep))
    assert not [a for a in res.alerts if a.kind == "security"]
    assert np.mean(res.forwarded) > 0.99
    assert all(v == 0.0 for v in res.calibration_final.values())


def test_warmup_forwards_everything_and_learns_a_baseline():
    ep = make_episode("clean", n=1200, seed=13)
    node = PIRCADFogNode(classifier=None)
    res = node.run(frames(ep))
    w = node.cfg.residual.baseline_window
    assert all(res.forwarded[:w])
    assert node.res.baseline_ready
    assert node.consistency.ready


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
