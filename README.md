# Sensor Drift and Data Poisoning in IoT Sensor Networks

Two generations of the same line of work.

**Generation 1 — FHAD** (`fog_node.py`, `hybrid_detector.py`, `evaluate_*.py`,
`random_forest_model.py`, `anomaly_injector.py`, `dataset_loader.py`): the
fog-based hybrid anomaly detector described in the original IDF — rolling-mean
detrending, CUSUM, and a windowed classifier activated only on flagged windows.
Left untouched.

**Generation 2 — PIRCAD** (`pircad/`, `experiments/`, `tests/`, `docs/`):
physics-informed fog-layer **root-cause disambiguation** of sensor drift and data
poisoning, with **differential response** and **adaptation inhibition**.

Start with `docs/PATENT_PACKAGE.md` for the filing material,
`docs/ARCHITECTURE.md` for the engineering description, and
`docs/IDF_B_GEN2.md` for a disclosure form in the same structure as the original.

---

## Why there is a Generation 2

The patentability search report (docket KM27250) found all five extracted
features of the Gen-1 disclosure — fog anomaly-detection architecture,
multi-stage hybrid framework, conditional ML activation, window preprocessing,
selective forwarding — disclosed across prior art D1–D4, and concluded the
invention "represents an obvious aggregation of known elements".

The report is right about the claimed feature set. The deeper problem is that the
Gen-1 title promises detection of *sensor drift **and** data poisoning*, but
nothing in the Gen-1 pipeline distinguishes them, and the system does the same
thing — discard — either way. Generation 2 makes the distinction the invention:

* it compares measurements against what the **physical process** says they should
  be, rather than against their own recent history;
* it decides **why** a deviation occurred — benign transducer drift, adversarial
  injection, or a genuine process excursion;
* it acts **differently** on each, and in particular it keeps poisoned data out of
  every adaptive element in the system.

## Quick start

```bash
python tests/test_pircad.py                       # 17 unit tests, ~2 min
python experiments/exp1_root_cause_benchmark.py   # the headline comparison
python experiments/run_all.py                     # full evidence suite
python experiments/run_all.py --quantum           # plus the quantum-kernel backend
```

Requires numpy, pandas, scikit-learn, matplotlib, joblib. Models are cached in
`artifacts/models.joblib` after the first run (fitting takes several minutes);
delete it to refit. The full suite takes roughly an hour on a laptop — most of it
is `exp3_ablation.py`, which refits and re-evaluates six configurations.

Results land in `artifacts/*.json`, figures in `figures/*.png`.

## Layout

```
pircad/
  simulator.py    experimental plant, transducers, attack models (evidence only)
  twin.py         E1  expected-response generation (dedicated-observer scheme)
  residual.py     E2  residual generation + character features
  triggers.py     E3  lightweight statistical trigger on the residual
  consistency.py  E4  cross-sensor physical-consistency localisation
  rootcause.py    E5  root-cause disambiguation + confidence
  escalation.py   E6  confidence-gated escalation
  quantum.py          optional gate-model quantum-kernel backend
  governor.py     E7  adaptation / retraining inhibition
  actions.py      E8  differential response + bounded recalibration
  fognode.py          the assembled system
  baselines.py        Gen-1 FHAD, and an adaptive-recalibrating comparison system
  metrics.py          evaluation suite
  training.py         corpus collection and model fitting

experiments/
  exp1_root_cause_benchmark.py   Gen-1 vs Gen-2 across six scenarios
  exp2_poisoned_adaptation.py    the technical effect: absorption and blindness
  exp3_ablation.py               what each claimed element is worth
  exp4_escalation.py             confidence gate + heavy/quantum backends
  exp5_legacy_dataset.py         the original single-channel record
  make_figures.py                figures for the disclosure

tests/
  test_pircad.py                 one test per claimed element

docs/
  ARCHITECTURE.md      engineering description
  PATENT_PACKAGE.md    title, abstract, claims, claim→code→evidence mapping
  IDF_B_GEN2.md        disclosure form, same structure as the Gen-1 IDF
```

## Headline result

Six scenarios, four held-out seeds, training and evaluation seeds disjoint.
Generation 1 is re-implemented faithfully and given the same training episodes and
the same four labels, so the gap is architectural.

| | PIRCAD (Gen 2) | FHAD (Gen 1) |
|---|---|---|
| root-cause accuracy | **0.963** | 0.540 |
| rate-matched injection vs benign drift | **0.992** | 0.298 |
| genuine process event correctly identified | 0.982 | 0.000 |
| poisoning contained (not forwarded) | 0.959 | 0.245 |
| injection misclassified as drift | **0.000** | 0.000 |

## The one experiment to look at

`exp2_poisoned_adaptation.py`. A false-data injection biases a temperature
channel slowly downwards at a rate inside the transducer's ageing envelope. A
system that equates "slow and persistent" with "drift" recalibrates — adopting the
attacker's bias as its new zero. The attacker then withdraws and a genuine
over-temperature fault occurs, against a cloud-side safety threshold applied to
whatever the fog layer forwards.

The comparison system absorbs the bias and under-reports the real fault. PIRCAD
classifies the ramp as poisoning rather than drift, quarantines it, and freezes
every adaptive element, so its reference is intact when the real fault arrives.

The middle column of that experiment is the control: same detector, same
residuals, same classifier, only the response made non-differential. It isolates
the contribution of acting on the cause from the contribution of detecting better.

## Scope, stated plainly

The strong separation of rate-matched drift from rate-matched injection needs
**physical redundancy** — several instruments whose readings are constrained by
known physics. `exp5_legacy_dataset.py` runs the same architecture on the original
single-channel temperature record, where that redundancy is absent; it works, but
less well, and instrument-versus-process separation is unavailable entirely. The
modelling assumptions behind the headline numbers are listed in
`docs/ARCHITECTURE.md` §4.
