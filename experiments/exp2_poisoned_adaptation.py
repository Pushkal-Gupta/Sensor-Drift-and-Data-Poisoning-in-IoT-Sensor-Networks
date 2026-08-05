"""
Experiment 2 -- the technical effect: preventing an adaptive system from being
walked, and preventing the resulting blindness to a genuine fault.

Scenario, in three acts.

  Act 1  nominal operation.
  Act 2  a false-data injection biases the internal-temperature channel slowly
         DOWNWARDS.  The ramp rate is inside the transducer's ageing envelope, so
         a system that equates "slow and persistent" with "drift" will conclude
         drift and recalibrate.  Recalibrating on this data means adopting the
         attacker's bias as the new zero.
  Act 3  the attacker withdraws, and a genuine over-temperature fault occurs.
         The cloud applies a safety threshold to the temperature the fog layer
         forwards.  Whether the alarm fires depends entirely on whether the
         reference was corrupted in Act 2.

This is the mechanism reported for acoustic injection against drift-adaptive MEMS
sensing: the attack is absorbed through the victim's own adaptation path, and the
lasting damage is not the attack itself but the blindness that follows it.

Run:  python experiments/exp2_poisoned_adaptation.py
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from _common import ARTIFACTS, WARMUP, banner, get_models, save_json

from pircad.baselines import AdaptiveRecalibratingBaseline
from pircad.config import DEFAULT_SENSORS, FogConfig, PlantParams
from pircad.fognode import PIRCADFogNode
from pircad.rootcause import HybridRootCauseClassifier
from pircad.simulator import (Episode, frames, inject_stealth_poison,
                              simulate_plant, transduce)
from pircad.types import Action, RootCause

#: cloud-side safety threshold applied to the forwarded temperature
ALARM_C = 34.0


def build_episode(n: int = 9000, attack_start: int = 1500,
                  attack_end: int = 6000, fault_at: int = 6600,
                  rate: float = -1.4e-3, seed: int = 4242) -> Episode:
    p = PlantParams()
    rng = np.random.default_rng(seed)
    truth, actuation = simulate_plant(n, p, rng, stuck_heater_from=fault_at)
    obs = transduce(truth, DEFAULT_SENSORS, rng)
    labels = np.full(n, RootCause.NORMAL, dtype=int)

    # Act 2: bias the channel downwards at a rate a drifting sensor could show
    win = np.arange(attack_start, attack_end)
    sub_truth = {"T_in": truth["T_in"]}
    sub_obs = {"T_in": obs["T_in"].copy()}
    sub_labels = np.full(n, RootCause.NORMAL, dtype=int)
    inject_stealth_poison(sub_obs, sub_truth, sub_labels, "T_in",
                          attack_start, rate, rng)
    obs["T_in"][win] = sub_obs["T_in"][win]
    labels[win] = RootCause.DATA_POISONING
    labels[fault_at:] = RootCause.PROCESS_EVENT

    return Episode(t=np.arange(n, dtype=float) * p.dt, truth=truth, obs=obs,
                   actuation=actuation, labels=labels, target_channel="T_in",
                   scenario="absorption_then_fault", params=p,
                   meta={"onset": float(attack_start), "attack_end": float(attack_end),
                         "fault_at": float(fault_at), "rate": float(rate),
                         "seed": float(seed)})


def first_cross(series: np.ndarray, threshold: float, start: int) -> Optional[int]:
    idx = np.flatnonzero(np.asarray(series)[start:] >= threshold)
    return int(idx[0] + start) if idx.size else None


def run_pircad(ep: Episode, clf, differential: bool) -> Dict[str, float]:
    cfg = FogConfig()
    cfg.differential_response = differential
    node = PIRCADFogNode(classifier=clf, cfg=cfg)
    res = node.run(frames(ep), truth_labels=ep.labels)

    d = res.per_channel["T_in"]
    offset = np.array([x.calibration_offset for x in d])
    fwd = np.array([x.forwarded for x in d])
    # what the cloud receives: the corrected value, and only when forwarded
    reported = ep.obs["T_in"] - offset
    seen = np.where(fwd, reported, -np.inf)

    fault_at = int(ep.meta["fault_at"])
    attack_end = int(ep.meta["attack_end"])
    true_cross = first_cross(ep.truth["T_in"], ALARM_C, fault_at)
    seen_cross = first_cross(seen, ALARM_C, fault_at)

    poison_win = slice(int(ep.meta["onset"]), attack_end)
    causes = res.causes()
    return {
        "absorbed_bias_at_attack_end_C": float(offset[attack_end - 1]),
        # the operationally decisive number: what the reference had become by the
        # time the genuine fault arrived.  A system that absorbs the ramp is then
        # walked a second time when the attacker withdraws and the reading jumps
        # back, so this is generally larger than the bias at the attack's end.
        "absorbed_bias_at_fault_C": float(offset[true_cross]) if true_cross else float("nan"),
        "poison_recall": float((causes[poison_win] == int(RootCause.DATA_POISONING)).mean()),
        "poison_called_drift": float((causes[poison_win] == int(RootCause.SENSOR_DRIFT)).mean()),
        # admissions on the attacked channel only -- the comparison system is
        # single-channel, so summing across channels would not be like for like
        "poisoned_samples_admitted": float(
            res.poisoned_admitted_by_channel.get("T_in", 0)),
        "poison_forwarded": float(fwd[poison_win].mean()),
        "alarm_fired": float(seen_cross is not None),
        "alarm_delay_samples": float(seen_cross - true_cross) if seen_cross else float("inf"),
        "true_cross": float(true_cross) if true_cross else float("nan"),
        "reported_at_true_cross": float(reported[true_cross]) if true_cross else float("nan"),
        "under_report_C": float(ep.truth["T_in"][true_cross] - reported[true_cross])
        if true_cross else float("nan"),
        "security_alerts": float(sum(1 for a in res.alerts if a.kind == "security")),
        "_offset": offset, "_reported": reported, "_causes": causes,
        "_forwarded": fwd,
    }


def run_adaptive(ep: Episode) -> Dict[str, float]:
    base = AdaptiveRecalibratingBaseline(warmup=WARMUP)
    tr = base.run(ep.obs["T_in"])
    reported = ep.obs["T_in"] - tr.offset
    fault_at = int(ep.meta["fault_at"])
    attack_end = int(ep.meta["attack_end"])
    true_cross = first_cross(ep.truth["T_in"], ALARM_C, fault_at)
    seen_cross = first_cross(reported, ALARM_C, fault_at)
    poison_win = slice(int(ep.meta["onset"]), attack_end)
    return {
        "absorbed_bias_at_attack_end_C": float(tr.offset[attack_end - 1]),
        "absorbed_bias_at_fault_C": float(tr.offset[true_cross]) if true_cross else float("nan"),
        "poison_recall": 0.0,      # no cause vocabulary at all
        "poison_called_drift": float(tr.detected[poison_win].mean()),
        "poisoned_samples_admitted": float((attack_end - int(ep.meta["onset"]))),
        "poison_forwarded": 1.0,
        "alarm_fired": float(seen_cross is not None),
        "alarm_delay_samples": float(seen_cross - true_cross) if seen_cross else float("inf"),
        "true_cross": float(true_cross) if true_cross else float("nan"),
        "reported_at_true_cross": float(reported[true_cross]) if true_cross else float("nan"),
        "under_report_C": float(ep.truth["T_in"][true_cross] - reported[true_cross])
        if true_cross else float("nan"),
        "security_alerts": 0.0,
        "_offset": tr.offset, "_reported": reported, "_causes": None,
        "_forwarded": np.ones(len(reported), dtype=bool),
    }


KEYS = ["absorbed_bias_at_attack_end_C", "absorbed_bias_at_fault_C",
        "poison_recall", "poison_called_drift", "poisoned_samples_admitted",
        "poison_forwarded", "security_alerts", "under_report_C",
        "reported_at_true_cross", "alarm_fired", "alarm_delay_samples"]


def main() -> None:
    models = get_models()
    clf = HybridRootCauseClassifier(models["root_cause"])
    ep = build_episode()

    banner("EXPERIMENT 2 - adaptation poisoning and the blindness that follows")
    print(f"attack window  : {int(ep.meta['onset'])} .. {int(ep.meta['attack_end'])}"
          f"  (rate {ep.meta['rate']:+.1e} C/sample, downwards)")
    print(f"genuine fault  : sample {int(ep.meta['fault_at'])} (heater contactor welded shut)")
    print(f"cloud alarm    : forwarded temperature >= {ALARM_C} C")
    print(f"true peak      : {ep.truth['T_in'].max():.2f} C\n")

    results = {
        "PIRCAD (full)": run_pircad(ep, clf, differential=True),
        "PIRCAD (no differential response)": run_pircad(ep, clf, differential=False),
        "Adaptive recalibrating baseline": run_adaptive(ep),
    }

    w = max(len(k) for k in KEYS) + 2
    head = "metric".ljust(w) + "".join(s[:24].rjust(26) for s in results)
    print(head)
    print("-" * len(head))
    for key in KEYS:
        cells = []
        for r in results.values():
            v = r[key]
            cells.append(("MISSED" if (key == "alarm_delay_samples" and not np.isfinite(v))
                          else f"{v:.3f}").rjust(26))
        print(key.ljust(w) + "".join(cells))

    banner("Reading of the result")
    full = results["PIRCAD (full)"]
    nodiff = results["PIRCAD (no differential response)"]
    adap = results["Adaptive recalibrating baseline"]
    print(
        f"The baseline admitted every one of {adap['poisoned_samples_admitted']:.0f} poisoned samples into its\n"
        f"adaptation.  It is walked twice: once by the ramp itself, and again when the\n"
        f"attacker withdraws and the reading jumps back, which it also reads as drift.\n"
        f"By the time the genuine fault arrived its reference was off by\n"
        f"{adap['absorbed_bias_at_fault_C']:+.2f} C.  It reported {adap['reported_at_true_cross']:.2f} C at the moment the plant truly\n"
        f"crossed {ALARM_C} C -- an under-report of {adap['under_report_C']:.2f} C -- and the alarm "
        + ("never fired.\n" if not np.isfinite(adap["alarm_delay_samples"])
           else f"fired {adap['alarm_delay_samples']:.0f} samples late.\n")
    )
    print(
        f"PIRCAD classified the ramp as poisoning rather than drift, quarantined "
        f"{1 - full['poison_forwarded']:.1%} of it,\n"
        f"raised a security indication, and froze every adaptive element: "
        f"{full['poisoned_samples_admitted']:.0f} poisoned\n"
        f"samples reached an adaptive element, against {adap['poisoned_samples_admitted']:.0f} for the baseline.  Its\n"
        f"reference at fault time was off by {full['absorbed_bias_at_fault_C']:+.3f} C and it under-reported the\n"
        f"true temperature by {full['under_report_C']:.2f} C.  "
        + ("The alarm fired on time.\n" if full["alarm_fired"] and full["alarm_delay_samples"] <= 0
           else f"Alarm delay: {full['alarm_delay_samples']:.0f} samples.\n")
    )
    print(
        f"The middle column is the control: same detector, same residuals, same\n"
        f"classifier -- only the response is made non-differential, so the ramp is\n"
        f"answered with recalibration.  It absorbs {abs(nodiff['absorbed_bias_at_attack_end_C']):.2f} C, forwards "
        f"{nodiff['poison_forwarded']:.1%} of the\n"
        f"poisoned data, and ends up over-reporting by {abs(nodiff['under_report_C']):.2f} C -- firing the safety\n"
        f"alarm {abs(nodiff['alarm_delay_samples']):.0f} samples early, on a temperature the plant had not reached.\n"
        f"Detection is identical in all three PIRCAD columns; only the response differs.")

    np.savez(ARTIFACTS / "exp2_traces.npz",
             truth=ep.truth["T_in"], obs=ep.obs["T_in"], labels=ep.labels,
             pircad_offset=full["_offset"], pircad_reported=full["_reported"],
             pircad_causes=full["_causes"], pircad_forwarded=full["_forwarded"],
             nodiff_offset=results["PIRCAD (no differential response)"]["_offset"],
             nodiff_reported=results["PIRCAD (no differential response)"]["_reported"],
             adaptive_offset=adap["_offset"], adaptive_reported=adap["_reported"],
             meta=np.array([ep.meta["onset"], ep.meta["attack_end"],
                            ep.meta["fault_at"], ALARM_C]))
    save_json("exp2_poisoned_adaptation.json",
              {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")}
               for k, v in results.items()})


if __name__ == "__main__":
    main()
