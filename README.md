# Sensor Drift and Data Poisoning in IoT Sensor Networks

Telling **benign sensor drift** apart from **adversarial data poisoning** at the
fog layer, and acting differently on each.

Three tiers, one architecture:

| Tier | Where | What it is |
|---|---|---|
| **Generation 1 — FHAD** | repository root | the original fog-based hybrid detector: rolling-mean detrending, CUSUM, and a windowed classifier activated only on flagged windows. Left untouched. |
| **Generation 2 — PIRCAD** | `pircad/`, `experiments/` | the multi-sensor fog pipeline: physics-informed residuals, cross-sensor consistency, root-cause disambiguation, differential response and adaptation inhibition |
| **Hardware prototype** | `hardware/` | ESP32 + DHT22 edge node running the same architecture on two coupled channels, in Wokwi |

## Start here

| Document | For |
|---|---|
| `docs/ARCHITECTURE.md` | how the fog pipeline works and why it is shaped that way |
| `docs/PROTOTYPE.md` | circuit, dataset, mathematical model, algorithms, results |
| `docs/PATENT_PACKAGE.md` | title, abstract, claims, prior-art delta, filing caveats |
| `docs/IDF_B_GEN2.md` | disclosure form in the same structure as the Generation-1 IDF |
| `hardware/README.md` | running the node in Wokwi |

## Why there is a Generation 2

The patentability search report (docket KM27250) found all five extracted
features of the Generation-1 disclosure — fog anomaly-detection architecture,
multi-stage hybrid framework, conditional ML activation, window preprocessing,
selective forwarding — disclosed across prior art D1–D4, and concluded the
invention was "an obvious aggregation of known elements".

The report is right about the claimed feature set. The deeper problem is that
the Generation-1 title promises detection of *sensor drift **and** data
poisoning*, but nothing in the pipeline distinguishes them, and the system does
the same thing — discard — either way. Discarding is wrong for both: a drifting
sensor is still a working sensor, and poisoned data must be kept out of every
adaptive element or the attack is absorbed and the detector goes blind to
genuine faults.

Generation 2 makes the distinction the invention:

* it compares measurements against what the **physical process** says they
  should be, rather than against their own recent history;
* it decides **why** a deviation occurred — benign transducer drift, adversarial
  injection, or a genuine process excursion;
* it acts **differently** on each, and keeps poisoned data out of every adaptive
  element in the system.

## Running it

```bash
# fog pipeline
python tests/test_pircad.py                       # 17 unit tests
python experiments/exp1_root_cause_benchmark.py   # headline comparison
python experiments/run_all.py                     # full evidence suite

# hardware prototype — no board required
cd hardware && ./run_checks.sh
```

Fog side needs numpy, pandas, scikit-learn, matplotlib, joblib. Hardware side
needs g++ and python3; `fog_bridge.py --host` additionally needs paho-mqtt.

## Headline results

**Fog pipeline** — six scenarios, four held-out seeds, training and evaluation
seeds disjoint. Generation 1 is re-implemented faithfully and given the same
training episodes and the same four labels, so the gap is architectural.

| | PIRCAD (Gen 2) | FHAD (Gen 1) |
|---|---|---|
| root-cause accuracy | **0.963** | 0.540 |
| rate-matched injection vs benign drift | **0.992** | 0.298 |
| genuine process event correctly identified | 0.982 | 0.000 |
| poisoning contained (not forwarded) | 0.959 | 0.245 |
| injection misclassified as drift | **0.000** | 0.000 |

**Edge node** — per-mode recall, identical input to all three configurations:

| Injection | Baseline | Draft | **Proposed** |
|---|---|---|---|
| held +6 °C step | 0.0 % | 0.6 % | **100.0 %** |
| stealth ramp (+0.045 °C/sample) | 0.0 % | 0.0 % | **93.1 %** |
| benign linear ageing | 0.0 % | 0.0 % | **88.1 %** |

The baseline absorbs 5.8 °C of the step and 6.8 °C of the stealth ramp into its
own reference. The proposed configuration admits under 1 °C of either.

## The one experiment to look at

`experiments/exp2_poisoned_adaptation.py`. A false-data injection biases a
temperature channel slowly downwards at a rate inside the transducer's ageing
envelope. A system that equates "slow and persistent" with "drift" recalibrates
— adopting the attacker's bias as its new zero. The attacker then withdraws and
a genuine over-temperature fault occurs, against a cloud-side safety threshold
applied to whatever the fog layer forwards.

The comparison system absorbs the bias and under-reports the real fault by
5.99 °C; its alarm never fires. PIRCAD classifies the ramp as poisoning,
quarantines it, and freezes every adaptive element, so its reference is intact
when the real fault arrives.

The middle column of that experiment is the control: same detector, same
residuals, same classifier, only the response made non-differential. It isolates
the contribution of acting on the cause from the contribution of detecting
better.

## Scope, stated plainly

The strong separation of rate-matched drift from rate-matched injection needs
**physical redundancy** — several instruments whose readings are constrained by
known physics. `experiments/exp5_legacy_dataset.py` runs the same architecture
on the original single-channel temperature record, where that redundancy is
absent; it works, but less well. The edge node recovers part of it by using the
DHT22's two coupled channels, but a *rate-matched* injection remains
inseparable there, and the test suite asserts that it is — see
`docs/PROTOTYPE.md` §6.

## Layout

```
fog_node.py, hybrid_detector.py, evaluate_*.py   Generation 1, untouched
pircad/          Generation 2 fog pipeline
experiments/     evidence suite for the fog pipeline
tests/           unit tests, one per claimed element
hardware/        ESP32 + DHT22 prototype, Wokwi
docs/            architecture, prototype, patent package, disclosure form
```
