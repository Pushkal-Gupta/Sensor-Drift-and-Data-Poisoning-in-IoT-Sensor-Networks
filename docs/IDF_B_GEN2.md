# Invention Disclosure Format (IDF)-B — Generation 2

*Drafted to the same section structure as the Generation-1 IDF (Document No.
02-IPR-R003). Figures referenced here are produced by
`python experiments/make_figures.py` and land in `figures/`.*

> **Filing note.** This is a disclosure for a **new** invention, not an amendment
> to KM27250. See `docs/PATENT_PACKAGE.md` §0 on Section 59.

---

## 1. Title of the invention

**Physics-Informed Fog-Assisted System and Method for Root-Cause Disambiguation
of Sensor Drift and Data Poisoning in IoT Sensor Networks with Differential
Response Control**

## 2. Field / Area of invention

Internet of Things (IoT); fog and edge computing; sensor data reliability and
integrity; model-based fault detection and isolation; industrial process
monitoring and control. Specifically, the invention concerns determining *why* an
IoT sensor stream has deviated — benign transducer degradation, adversarial data
injection, or a genuine excursion of the monitored physical process — and
applying a different control action to each at the fog layer.

## 3. Prior patents and publications

| # | Reference | Year | Area | Key points | Gap versus the present invention |
|---|---|---|---|---|---|
| D1 | Systems, methods and computer program products for anomaly detection (Yokogawa Electric / Yantai Univ., US) | 2019 | Fog anomaly detection | Multi-stage anomaly detection with conditional activation and selective forwarding | Detects *that* an anomaly occurred. No process model, no cause determination, one response for all anomalies |
| D2 | Anomaly detection method based on edge computing gateway (Hebei Zhuozhi) | 2021 | Edge anomaly detection | Gateway-side hybrid statistical/ML detection | As D1. No cross-sensor physical constraints, no adaptation gating |
| D3 | AI-based predictive maintenance for distributed energy storage (Bitower Shanghai) | 2025 | Predictive maintenance | Learned baselines, conditional ML | Learned baseline from the signal's own history, not from process physics; no drift/attack distinction |
| D4 | Using an adaptive threshold for anomaly detection (Datashapes Inc.) | 2024 | Adaptive thresholding | Threshold adapts to the observed data | The adaptive threshold is precisely the mechanism a slow injection exploits; no adaptation inhibition |
| P1 | Acoustic injection attacks on drift-adaptive MEMS accelerometers, *Sensors* (MDPI) | 2024 | Attack literature | An attack absorbed through the victim's own drift adaptation; the system then reads abnormal states as normal | Establishes the problem. Provides no detection or mitigation architecture |
| P2 | Korycki & Krawczyk, *Machine Learning* | 2022 | Robust drift detection | Drift detection under poisoning | Robustness of the detector; does not classify the cause or act differently on it |
| P3 | De Silva, Kalman-residual sensor fault detection | 2020 | Model-based FDI | Residual-based fault detection | Residual detection only; no adversarial model, no differential response |
| P4 | Sayghe, digital-twin-residual intrusion detection | 2025 | Digital twin + IDS | Twin residuals with drift-gated recalibration | Closest single reference. Recalibrates on drift-gated triggers; does not classify drift versus attack, and does not inhibit adaptation |
| P5 | Alotibi & Tipper, physics-informed detection of false data injection (wind farm data) | — | Physics-informed detection | Physics residuals detect FDI (73–97 %) | Detection only, single application, no cause vocabulary and no response control |
| P6 | Elsedimy, *Cluster Computing*; Hdaib et al. | 2024 | Quantum ML for IoT IDS | QSVM / quantum autoencoder intrusion detection | Network-level intrusion detection; not sensor-stream cause disambiguation. Cited to bound the optional quantum claim |
| P7 | US20230094389A1 / US12566987B2 (Cornell) | 2020 prio. | Quantum-assisted fault detection | Neural fault detection **trained** using quantum sampling | The optional quantum claim here is inference-side and gate-model, deliberately clear of this |

## 4. Summary and background — the gap addressed

The Generation-1 disclosure (KM27250, "Fog-Assisted Early Detection of Sensor
Drift and Data Poisoning") was found by patentability search to be fully
anticipated: fog-based anomaly detection, multi-stage hybrid detection,
conditional ML activation, window preprocessing and selective forwarding are each
disclosed in D1–D4.

The deeper deficiency is that the Gen-1 architecture cannot do what its title
promises. CUSUM flags a deviation; Isolation Forest labels a window abnormal;
neither says *why*, and the system discards the data in either case. Discarding is
the wrong action for both causes:

* a drifting sensor is still a working sensor — the correct response is a
  calibration correction, not deletion;
* poisoned data must be kept out of every adaptive element, or the attack is
  absorbed and the system is afterwards blind to genuine faults (P1).

There is a third case neither generation of the art recognises: the deviation may
be **real**. The instruments may be healthy and the plant may have moved. A system
that answers every persistent deviation with recalibration will erase a genuine
fault.

The present invention makes the distinction the invention. Measurements are
compared against what a model of the **monitored physical process** says they
should have been; the character of the resulting residual and the pattern of
violations of physical constraints among the sensors determine the *cause*; and
the fog node then applies a **different control action** to each cause, with
adaptation of every model in the system gated on the outcome.

## 5. Objectives

1. To determine the root cause of a deviation in an IoT sensor stream —
   benign sensor drift, adversarial data poisoning, or a genuine process
   excursion — at the fog layer, rather than merely detecting that a deviation
   occurred.
2. To ground that determination in the physics of the monitored process, by
   generating each channel's expected response from a model anchored on the other
   channels and on the process dynamics.
3. To prevent an adaptive monitoring system from being walked by a slow data
   injection, by inhibiting every adaptive element on suspect data.
4. To keep a degraded but usable sensor in service through bounded, rate-limited,
   audited recalibration instead of discarding its data.
5. To ensure a genuine process excursion is never masked by recalibration.
6. To keep the cost at the fog node bounded, by activating the expensive stage
   only on triggered channels and escalating only unresolved ones.

## 6. Working principle (brief)

At the fog node, for every synchronous multi-channel sample:

1. a standing calibration correction is applied;
2. a process model produces an expected response for each channel, from a state
   anchored on the *other* channels and on the modelled dynamics;
3. the residual (measured − expected) is formed and normalised against statistics
   learned during commissioning and thereafter frozen;
4. a lightweight statistical chart on the residual decides whether the expensive
   stage runs at all;
5. violations of stored physical constraints among the channels are attributed to
   a hypothesis — a bias on one channel, or a perturbation of the process itself;
6. the residual's temporal character (growth rate against the sensor's ageing
   envelope, onset discontinuity, preservation of the channel's noise signature,
   coherence with the model-predicted dynamics) plus that attribution give the
   root cause and a confidence;
7. the fog node acts on the cause: recalibrate and keep forwarding for drift;
   quarantine, freeze adaptation and alert for poisoning; forward, alarm and
   forbid calibration for a process excursion; hold pending escalation if unsure.

## 7. Detailed description

### 7.1 Expected-response generation (Figure 1, upper panels)

The expectation comes from the process, not from the signal's own history. In the
evaluation testbed the process is a heated sealed vessel instrumented with four
sensors — internal temperature, ambient temperature, pressure and relative
humidity — related by

```
C dT/dt = Q·u(t) − k (T − T_amb)          Newton heating/cooling
P        = P_ref (T + 273.15)/(T_ref + 273.15)     isochoric ideal gas
RH       = 100 e_vap / e_s(T)             conserved absolute humidity (Magnus)
```

Critically, each channel's expectation is generated by a state estimate that
**excludes that channel** (a dedicated-observer structure):

```
expected(T_in)  ← P, RH, thermal dynamics
expected(P)     ← T_in, RH
expected(RH)    ← T_in, P
expected(T_amb) ← energy balance: T_in − Q·duty/k
```

This is not a stylistic choice. A single closed-loop estimator tracks its own
measurements, so the innovation of a slowly biased channel remains at the noise
floor and the bias never appears — exactly the blind spot a slow injection
exploits. It was observed directly during development: with a single estimator the
residual under a 4 °C drift stayed at ≈1σ; with the dedicated-observer structure
it reached ≈44σ.

### 7.2 Residual character features (Figure 2)

Figure 2 shows the decisive comparison: benign drift and a false-data injection
of **identical amplitude and identical growth rate**. Their residuals are
indistinguishable in magnitude. They differ in character:

| feature | physical statement |
|---|---|
| `slope_ratio` | growth rate against the transducer's stored ageing envelope |
| `monotone_frac` | ageing bias is smooth and one-directional |
| `corner`, `jump_max` | an injection has an onset — a discontinuity ageing cannot produce |
| `acf_shift`, `var_ratio`, `flatness`, `hf_ratio`, `kurt_incr` | a drifting transducer still observes the real process, so its innovation statistics are preserved; a substituted stream carries the attacker's noise instead |
| `dyn_coherence` | whether the reported stream still follows the modelled micro-dynamics of the plant |

`dyn_coherence` is the most legible of these. It is the correlation between the
residual's increments and the model-predicted increments, negated. A
healthy-but-drifting sensor still tracks the plant, so its residual increments are
uncorrelated with the predicted ones (measured −0.036). A substituted stream does
not follow the real micro-dynamics, so its residual increments become the negative
of the predicted increments (measured +0.430). Forging this requires a real-time,
exact model of the plant *and* of the controller's switching phase.

These features are, however, **individually redundant and collectively
necessary**, and §8.4 reports the ablation that shows it: removing
`dyn_coherence` alone costs 0.001 of accuracy, because the noise-signature and
onset features reach the same conclusion by another route; removing the group
costs 0.086 and raises the rate at which injections are misread as drift from
0.000 to 0.093. The protection to seek is over the set, not over any one feature —
which is also what makes the scheme robust to an attacker who defeats one of them.

### 7.3 Statistical trigger

CUSUM, EWMA or Page-Hinkley on the residual, plus an instantaneous spike test.
This is the only Gen-1 stage retained structurally, and it is deliberately
demoted: it is no longer the detector, only the cheap gate deciding when the
expensive stage may run. Because it now operates on a residual whose nominal
distribution is stationary by construction, rolling-mean detrending and seasonal
handling are unnecessary. Measured nominal firing rate: 0.6–5.4 % per channel.

### 7.4 Cross-sensor physical-consistency localisation (Figure 3)

Five constraints are evaluated on the reported measurements — gas law, humidity,
an open-loop increment check, the quasi-steady energy balance, and a control
constraint against the setpoint. Each candidate cause has a **signature**: the
direction in which it pushes the violation vector. Signatures are obtained by
perturbing the *model*, so no labelled fault data is required. The hypothesis set
includes an explicit **process-perturbation** hypothesis alongside the per-channel
bias hypotheses.

The asymmetry that makes this work: an instrument bias violates the gas-law and
humidity constraints — no process fault can do that, because pressure and humidity
really did follow the real temperature — while a process fault leaves both
untouched and breaks the balance and control constraints instead.

### 7.5 Adaptation governor

A per-channel gate in front of every adaptive element: the process model's state
re-anchoring, the residual normalisation statistics, the calibration offsets, and
any retraining of the classification stage.

```
TRUSTED ─trigger─► WATCH ─drift verdict─► DRIFTING (held while compensating)
                     └─poison verdict─► QUARANTINED ─hold─► RECOVERING ─clean run─► TRUSTED
```

A channel that is merely *triggered* already stops feeding the model — before any
cause has been decided. Trust weights are slewed rather than switched, because a
step change in one anchor's weight re-partitions the estimator and appears as a
jump in every other channel's residual.

### 7.6 Differential response (Figure 4)

| cause | action |
|---|---|
| benign sensor drift | bounded, rate-limited, audited calibration correction; **data keeps flowing** |
| suspected data poisoning | quarantine; adaptation frozen; security indication; nothing forwarded |
| genuine process excursion | forward **and** alarm; calibration explicitly inhibited |
| unresolved | held pending asynchronous escalation; no irreversible action |

## 8. Experimental validation results

Six scenarios × four held-out seeds (training and evaluation seeds disjoint),
4 000 samples per episode, four channels. Generation-1 FHAD is re-implemented
faithfully and given the same training episodes and the same four labels, so any
gap is attributable to architecture rather than to label vocabulary.

### 8.1 Root-cause accuracy by scenario (Figure 5)

| scenario | PIRCAD (Gen 2) | FHAD (Gen 1) | Δ |
|---|---|---|---|
| clean | 1.000 | 1.000 | +0.000 |
| benign drift | 0.846 | 0.314 | +0.531 |
| **rate-matched injection** | **0.992** | **0.298** | **+0.694** |
| spike poisoning | 0.996 | 0.999 | −0.003 |
| replay poisoning | 0.953 | 0.299 | +0.653 |
| genuine process event | 0.989 | 0.331 | +0.658 |

### 8.2 Overall

| metric | PIRCAD (Gen 2) | FHAD (Gen 1) |
|---|---|---|
| root-cause accuracy | **0.963** | 0.540 |
| detection accuracy | 0.983 | 0.540 |
| drift recall | 0.775 | 0.000 |
| poisoning recall | 0.959 | 0.245 |
| process-event recall | 0.982 | 0.000 |
| false-drift rate (injection called drift) | **0.000** | 0.000 |
| process-masking rate | 0.015 | 0.000 |
| false-alarm rate | 0.003 | 0.000 |
| poison containment | 0.959 | 0.245 |
| drift kept flowing | 0.998 | 1.000 |
| retraining protection | 0.986 | not applicable |
| bandwidth saved | 0.228 | ≈0.000 |
| conditional-activation rate | 0.296 | — |
| average latency | 3 × 10⁻³ s/sample | 3 × 10⁻⁶ s/sample |

Generation 1 collapses to "everything is normal" on three of the six scenarios;
its apparent 0.540 accuracy is almost entirely the normal class.

### 8.3 The technical effect (Figure 4)

An injection biases the temperature channel downwards for 4 500 samples at a rate
inside the ageing envelope; the attacker then withdraws and a genuine
over-temperature fault occurs, against a cloud-side alarm at 34 °C.

| | PIRCAD (full) | response made non-differential | adaptive recalibrating baseline |
|---|---|---|---|
| poisoning recall | 0.991 | 0.980 | — |
| poisoned samples admitted to adaptation (attacked channel) | **0 of 4 500** | 0 of 4 500 | **4 500 of 4 500** |
| poisoned samples forwarded | **0.9 %** | 99.8 % | 100 % |
| bias absorbed by the end of the attack | **0.000 °C** | −6.09 °C | −1.46 °C |
| reference error when the real fault arrived | **0.000 °C** | −2.80 °C | **+5.95 °C** |
| security indications | 4 627 | 78 | **0** |
| error at the true 34 °C crossing | **+0.04 °C** | −2.76 °C | **+5.99 °C** |
| safety alarm | fired (2 samples late) | fired 99 samples **early**, on a temperature never reached | **never fired** |

The baseline is walked twice: once by the ramp, and again when the attacker
withdraws and the reading jumps back — which it also reads as drift, driving its
reference to +5.95 °C in the wrong direction. That second effect is why the
under-report at the crossing (5.99 °C) exceeds the bias absorbed during the
attack itself.

The middle column is the control: identical detector, residuals and classifier,
with only the response made non-differential. It isolates the contribution of
acting on the cause.

### 8.4 Ablation — what each element is worth

24 held-out episodes, one element removed per row, everything else unchanged.

| configuration | root-cause accuracy | injection read as drift | poisoning contained | process fault masked |
|---|---|---|---|---|
| **full** | **0.963** | **0.000** | **0.959** | 0.015 |
| − physics residual (Gen-1 rolling mean) | 0.701 | 0.041 | 0.506 | 0.137 |
| − all residual-character features | 0.877 | 0.093 | 0.690 | 0.067 |
| − differential response | 0.890 | 0.014 | **0.118** | **0.311** |
| − cross-sensor consistency | 0.950 | 0.000 | 0.970 | 0.000 |
| − excitation coherence alone | 0.962 | 0.000 | 0.949 | 0.015 |
| − noise-signature features alone | 0.954 | 0.000 | 0.857 | 0.014 |

The physics residual is the single largest contributor to accuracy. The
differential response contributes little to accuracy and nearly everything to
effect: without it, containment of poisoned data collapses from 0.959 to 0.118 and
the rate at which a genuine process fault is masked by recalibration rises from
0.015 to 0.311.

### 8.5 Honest negative results

* **Escalation to a heavier classifier does not pay for itself.** The confidence
  gate does isolate the hard cases (local accuracy 0.978 when confident, 0.501 on
  the 6.9 % it withholds), but a heavier model over the same features recovers only
  +0.043 on those samples, and end-to-end accuracy is unchanged within noise
  (0.956 → 0.952). The value of escalation here is bounded local cost and no
  irreversible action on unresolved samples, not accuracy.
* **The quantum-kernel backend performed worse than the local verdict**: 0.303
  against 0.501 on the escalated samples (−0.198). The near-term kernel requires
  projecting 22 residual features onto 4 qubits, and the discriminating structure
  does not survive that compression. This is measured on a working ZZ-feature-map
  fidelity kernel. The conclusion for the filing is that quantum should be a
  narrow optional dependent feature only — it must not be presented as improving
  detection, because on this evidence it does not.
* **The single-channel case is much weaker.** On the original Daily Minimum
  Temperature record with no physical redundancy (§8.6), the architecture still
  separates a substituted stream from drift better than Gen-1 (poison F1 0.687 vs
  0.253), but overall accuracy is close (0.660 vs 0.642) and the false-alarm rate
  is high (0.23). The strong result requires redundancy.
* **Drift is detected with a delay of ≈76 samples.** This is physics-limited — the
  bias must exceed the noise floor — not a defect, but it is real.

### 8.6 Continuity with the Generation-1 dataset

| metric | PIRCAD (single channel) | FHAD (Gen 1) |
|---|---|---|
| root-cause accuracy | 0.660 | 0.642 |
| drift recall | 0.631 | 0.629 |
| poisoning recall | **0.524** | 0.147 |
| poisoning F1 | **0.687** | 0.253 |
| drift F1 | 0.591 | 0.760 |
| false-alarm rate | 0.230 | 0.024 |

## 9. What aspects need protection

The combination, at a fog node interposed between physical sensors and a
centralised system, of:

1. generating each sensor channel's **expected response** from a model of the
   monitored physical process whose state is anchored on the *other* channels and
   on the process dynamics, to the exclusion of that channel;
2. determining the **root cause** of a residual deviation as benign sensor drift,
   suspected data poisoning, or a genuine process excursion, from (a) temporal
   characteristics of the residual including its growth rate against a stored
   ageing envelope and its coherence with the model-predicted response, and
   (b) attribution of physical-constraint violations to a hypothesis set that
   includes both per-channel biases and a process perturbation;
3. applying a **different control action** per cause — bounded, rate-limited,
   audited recalibration with continued forwarding for drift; quarantine with
   inhibition of every adaptive element and a security indication for poisoning;
   forwarding with an alarm and explicit inhibition of calibration for a process
   excursion;
4. **inhibiting adaptation** of the process model, the residual normalisation
   statistics, the calibration offsets and any retraining of the classification
   stage on data classified as suspect, with de-weighting beginning at trigger
   time before a cause has been determined;
5. maintaining a drift classification for a channel while a calibration correction
   remains in force, notwithstanding the subsequent absence of residual deviation.

Subsidiary: confidence-gated **asynchronous** escalation with the sample withheld
and the local verdict standing if no response returns; and, as a narrow dependent
feature, a gate-model quantum-kernel classifier as that remote service (drafted
inference-side to remain clear of P7).

## 10. Technology readiness level

**TRL 4** — component and system validated in a laboratory environment. The full
pipeline runs end to end on a simulated physical plant with characterised
transducers and four distinct attack models, with training and evaluation seeds
disjoint, a unit-test suite covering each claimed element, and an ablation
isolating each element's contribution. Generation 1 was assessed at TRL 3.

Advancing to TRL 5 requires deployment on instrumented hardware — the vessel
testbed is directly realisable with a heated enclosure, a thermistor, a MEMS
pressure sensor and a capacitive humidity sensor on a fog gateway.
