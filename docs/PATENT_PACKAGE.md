# Generation-2 disclosure package

Prepared against docket **KM27250** / reference **IPR0004174P**, whose
patentability search report (issued 29/04/2026) concluded that the Generation-1
claimed feature set F1–F5 was fully disclosed by D1, D2 and D4 and substantially
by D3, and that the invention was "an obvious aggregation of known elements".

> **Legal note, read this first.** This document specifies what a *new* filing
> should claim. It is **not** a set of amendments to KM27250. The original
> specification describes rolling-mean preprocessing, CUSUM, Isolation Forest,
> conditional activation and filtering. Physics/process-model residuals,
> cross-sensor consistency localisation, root-cause disambiguation, adaptation
> inhibition and confidence-gated escalation are **not** disclosed there, and
> inserting them by amendment would almost certainly be new matter under
> **Section 59 of the Patents Act, 1970**. The realistic routes are a fresh
> application (optionally with KM27250 as a related/parent filing) or a patent of
> addition. Your IP cell or external counsel must make that call — this document
> assumes a fresh filing.
>
> A second timing point: several relevant publications post-date a 2026 original
> filing but pre-date a filing made now. For a **new** application they are all
> citable prior art.

---

## 1. Title

**Physics-Informed Fog-Assisted System and Method for Root-Cause Disambiguation
of Sensor Drift and Data Poisoning in IoT Sensor Networks with Differential
Response Control**

Shorter alternative if the office prefers brevity:

**Fog-Layer Root-Cause Disambiguation and Adaptive-Response Control for IoT
Sensor Networks**

Deliberately absent from the title: CUSUM, Isolation Forest, digital twin,
quantum. Those are embodiments, not the invention.

## 2. Abstract

> A fog-assisted system and method for distinguishing benign sensor drift from
> adversarial data poisoning in an Internet-of-Things sensor network, and for
> responding differently to each, is disclosed. Measurements from a plurality of
> physical sensors monitoring a physical process are received at an intermediate
> fog node. The fog node determines, for each sensor channel, an expected sensor
> response from a model of the monitored physical process — a physics-based
> process model, a digital representation, a state estimator, or a learned
> physics-constrained relation — wherein the expected response of a given channel
> is generated from a model state anchored on the remaining channels and on the
> process dynamics to the exclusion of that given channel, and generates a
> residual signal representing the deviation between the measured and the
> expected response. A lightweight statistical monitoring stage evaluates the
> residual and conditionally activates a root-cause classification stage. The
> root-cause classification stage evaluates temporal characteristics of the
> residual, including its rate of growth relative to a stored ageing envelope of
> the sensor, the preservation of the sensor's innovation statistics, and the
> degree to which variation of the residual remains coherent with variation of
> the model-predicted response, together with an attribution of violations of
> stored physical-consistency constraints among the sensor channels to a
> hypothesised cause selected from a set comprising a bias on each channel and a
> perturbation of the monitored process, and thereby classifies the deviation as
> benign sensor drift, suspected data poisoning, or a genuine process excursion,
> with an associated confidence. Based on the determined root cause, the fog node
> applies differential control: for benign drift, a bounded, rate-limited and
> audited calibration correction is applied while data continues to be forwarded;
> for suspected data poisoning, the affected data is quarantined, incorporation of
> the affected data into any adaptive element of the system — including the
> process model state, the residual normalisation statistics, the calibration
> offsets and any retraining of the classification stage — is inhibited, and a
> security indication is generated; for a genuine process excursion, the data is
> forwarded together with an alarm and calibration is inhibited. Where the
> confidence is below a threshold the sample is withheld from forwarding and from
> adaptation pending an asynchronously obtained classification from a remote
> classifier, the locally determined classification being applied if no remote
> classification is returned.

## 3. The technical problem

An IoT sensor stream can deviate from nominal for physically distinct reasons
that demand opposite responses:

| cause | correct response | wrong response and its consequence |
|---|---|---|
| transducer ageing (drift) | correct and keep using the sensor | discarding it loses coverage |
| adversarial injection | quarantine **and freeze adaptation** | recalibrating adopts the attacker's bias as the new zero |
| genuine process excursion | forward and alarm | recalibrating erases a real fault |

Generation 1, and D1–D4, treat all three identically: detect, then filter. Worse,
an adaptive detector that keeps learning from whatever it accepts can be *walked*:
an injection that grows more slowly than the system's own adaptation becomes the
new normal, after which the system is blind to genuine faults because the absorbed
offset cancels them. This is the mechanism documented for acoustic injection
against drift-adaptive MEMS sensing.

## 4. Claimed elements, and why each is not in D1–D4

| # | element | D1–D4 position |
|---|---|---|
| **E1** | expected response from a model of the **monitored physical process**, with each channel's expectation generated from a state anchored on the *other* channels (dedicated-observer structure) | D1–D4 derive the reference from the signal's own history / a learned baseline. No physical process model, no leave-one-out structure. |
| **E2** | residual **character** features that separate causes rather than magnitudes: growth rate vs a stored ageing envelope, onset discontinuity, preservation of the channel's innovation statistics, and coherence between residual variation and model-predicted variation | Not present. D1–D4 threshold the magnitude or feed raw windows to a classifier. |
| **E3** | lightweight statistical trigger **on the residual**, gating the expensive stage | This alone is F2/F3 and *is* disclosed. Claimed only in combination. |
| **E4** | attribution of physical-consistency-constraint violations to a hypothesis set containing per-channel bias **and a process-perturbation hypothesis**, with signatures obtained by perturbing the model rather than from labelled faults | Not present. D1–D4 have no notion of instrument-side vs process-side. |
| **E5** | classification of the **root cause** into {drift, poisoning, process excursion} with confidence | Not present. D1–D4 output an anomaly label. |
| **E6** | confidence-gated, **asynchronous** escalation, with the sample withheld meanwhile and the local verdict standing if no answer returns | Not present. |
| **E7** | inhibition of **every** adaptive element on suspect data, with de-weighting beginning at trigger time before any cause is decided, a minimum freeze hold, and a clean-run/operator-acknowledgement release | Not present, and it is the element with the clearest technical effect. |
| **E8** | **differential** control action per cause, including bounded rate-limited audited recalibration for drift and explicit inhibition of calibration on a process excursion | Not present. D1–D4 filter or alarm. |

**The inventive concept is the combination E1+E2+E4+E5+E7+E8** — expectation from
physics, causes separated by residual character and constraint attribution, and
different action taken on each, with adaptation gated on the outcome. E3 is the
known part and is claimed only as a cooperating element. E6 is optional.

## 5. Claims

### Independent claim 1 (system)

> A system for managing sensor data reliability in an Internet-of-Things sensor
> network, comprising:
>
> a plurality of physical sensors arranged to measure a monitored physical
> process; and
>
> a fog node communicatively coupled between the plurality of physical sensors
> and a centralised computing system, the fog node comprising a processor and a
> memory storing instructions that, when executed, cause the fog node to:
>
> (a) receive measurements from the plurality of physical sensors;
>
> (b) determine, for each of a plurality of sensor channels, an expected sensor
> response using a stored model representing the monitored physical process,
> wherein the expected sensor response for a given sensor channel is determined
> from a state of the stored model that is anchored on measurements of sensor
> channels other than the given sensor channel and on modelled dynamics of the
> monitored physical process;
>
> (c) generate a residual signal for each sensor channel from a difference
> between the measurement and the expected sensor response of that sensor
> channel;
>
> (d) evaluate the residual signal with a statistical monitoring stage and, on the
> statistical monitoring stage satisfying a triggering criterion for a sensor
> channel, conditionally activate a root-cause classification stage for that
> sensor channel;
>
> (e) evaluate, at the root-cause classification stage, (i) one or more temporal
> characteristics of the residual signal, comprising at least a rate of change of
> the residual signal compared with a stored envelope of physically plausible
> degradation of the sensor, and a measure of coherence between variation of the
> residual signal and variation of the expected sensor response, and (ii) an
> attribution of violations of a plurality of stored physical-consistency
> constraints relating measurements of two or more of the sensor channels, said
> attribution being to a hypothesis selected from a hypothesis set comprising a
> bias on each of the sensor channels and a perturbation of the monitored
> physical process;
>
> (f) determine therefrom a root cause of the deviation, the root cause being one
> of a benign sensor drift, a suspected data poisoning, and a process excursion of
> the monitored physical process, and a confidence associated therewith; and
>
> (g) apply a control action selected in dependence on the determined root cause,
> wherein:
>
>   (i) responsive to a determination of benign sensor drift, the fog node applies
>   a calibration correction to subsequent measurements of the sensor channel,
>   said calibration correction being limited to a maximum change per sample and
>   to a maximum cumulative magnitude, and continues to forward measurements of
>   that sensor channel to the centralised computing system;
>
>   (ii) responsive to a determination of suspected data poisoning, the fog node
>   withholds measurements of the sensor channel from the centralised computing
>   system, inhibits incorporation of measurements of that sensor channel into
>   each adaptive element of the system, said adaptive elements comprising at
>   least the state of the stored model, statistics used to normalise the residual
>   signal, and the calibration correction, and generates a security indication;
>   and
>
>   (iii) responsive to a determination of a process excursion, the fog node
>   forwards measurements of the sensor channel together with an alarm indication
>   and inhibits application of a calibration correction to that sensor channel.

### Independent claim 2 (method)

The corresponding method claim, same steps (a)–(g).

### Independent claim 3 (computer-readable medium)

A non-transitory computer-readable medium storing instructions which, when
executed by a processor of a fog node, cause it to perform the method of claim 2.

### Dependent claims

| # | subject | code |
|---|---|---|
| 4 | the stored model comprises a first-principles model of the monitored physical process expressed as a differential relation driven by an actuation command of the process | `twin.PhysicsVesselTwin` |
| 5 | the stored model comprises a digital twin or digital representation of the monitored physical process | `twin.ProcessModel` |
| 6 | the stored model comprises a state estimator, and a separate estimator instance is maintained for each sensor channel, each instance excluding its own channel from its state update | `twin.PhysicsVesselTwin.OBSERVERS` |
| 7 | the stored model comprises a periodic-component model of an environmental variable fitted over a commissioning window | `twin.SeasonalHarmonicTwin` |
| 8 | the statistical monitoring stage comprises a cumulative-sum chart applied to the residual signal and reset upon signalling | `triggers.TwoSidedCUSUM` |
| 9 | ... an exponentially-weighted moving-average chart or a Page-Hinkley test | `triggers.EWMAChart`, `PageHinkley` |
| 10 | the temporal characteristics further comprise a measure of discontinuity at the onset of the deviation | `residual` feature `corner` |
| 11 | the temporal characteristics further comprise a comparison of autocorrelation and variance of increments of the residual signal against values learned during a commissioning window and thereafter held fixed | `residual` features `acf_shift`, `var_ratio`; `BaselineStats.frozen` |
| 12 | the coherence measure is a correlation between increments of the residual signal and increments of the expected sensor response | `residual` feature `dyn_coherence` |
| 13 | signatures of each hypothesis in the hypothesis set are obtained by perturbing the stored model, without labelled fault data | `consistency._build_signatures` |
| 14 | each physical-consistency constraint is normalised by the greater of a statistic learned over a commissioning window and a stored physical-resolution floor for that constraint | `consistency.sigma_floors` |
| 15 | a physical-consistency constraint compares a measured change of a state variable over a horizon against a change integrated open-loop from the measurements themselves | `consistency` constraint `g_dyn` |
| 16 | a physical-consistency constraint compares a smoothed state variable against a control setpoint of the monitored process | `consistency` constraint `g_control` |
| 17 | the root-cause classification stage comprises a supervised classifier over a feature vector comprising said temporal characteristics and said attribution | `rootcause.MLRootCauseClassifier` |
| 18 | the root-cause classification stage comprises a rule set in which each term is a physical statement, requiring no training data | `rootcause.RuleRootCauseClassifier` |
| 19 | a determination of a process excursion by the attribution overrides a contrary output of the classifier, and a classifier output of process excursion is overridden where the attribution localises the violations to the sensor channel under evaluation | `rootcause.HybridRootCauseClassifier` |
| 20 | the fog node reduces a weight with which a sensor channel contributes to the state of the stored model upon the triggering criterion being satisfied and before a root cause has been determined | `governor.TRUST_WEIGHT` (`WATCH`) |
| 21 | said weight is transitioned progressively rather than stepwise | `twin.weight_slew` |
| 22 | a non-zero floor weight is retained so that the model state remains anchored when every channel is suspect | `twin.floor_weight` |
| 23 | inhibition persists for a minimum hold interval and is released only after a run of samples not satisfying the triggering criterion | `governor.freeze_hold`, `clean_to_resume` |
| 24 | release of inhibition further requires an operator acknowledgement received out of band | `governor.require_operator_ack`, `operator_ack()` |
| 25 | a determination of benign sensor drift is maintained for the sensor channel while a calibration correction exceeding a threshold remains in force, notwithstanding a subsequent absence of residual deviation | `fognode` standing-classification rule; `governor` `compensating` |
| 26 | each calibration correction is recorded in an audit record comprising sample index, magnitude and confidence | `actions.CalibrationRecord` |
| 27 | responsive to the confidence being below a threshold, the fog node transmits the feature vector to a remote classification service, withholds the measurement from forwarding and from adaptation, and applies the locally determined root cause if no response is received | `escalation.EscalationService` |
| 28 | the remote classification service comprises a gate-model quantum processor evaluating a quantum kernel between feature vectors | `escalation.QuantumEscalationBackend`, `quantum.py` |
| 29 | the quantum kernel is a fidelity kernel of states prepared by a parameterised entangling feature map | `quantum.fidelity_kernel` |
| 30 | the root-cause classification stage is re-evaluated at an interval, and is re-evaluated unconditionally whenever the triggering criterion is newly satisfied | `fognode.reclassify_every`, `due = st.fired or ...` |

## 6. Claim → code → evidence

| element | module | experiment | measured |
|---|---|---|---|
| E1 physics expectation | `twin.py` | exp3 row `-physics residual` | accuracy 0.963 → **0.701**; poisoning recall 0.959 → 0.506 |
| E2 residual character | `residual.py` | exp3 row `-all character feats` | accuracy 0.963 → **0.877**; injections misread as drift 0.000 → **0.093** |
| E3 trigger | `triggers.py` | exp1 `conditional_activation_rate` | 0.296 of channel-samples |
| E4 consistency attribution | `consistency.py` | exp3 row `-cross-sensor consistency` | accuracy 0.963 → 0.950 (small on accuracy; it is what makes the process/instrument distinction possible at all) |
| E5 root cause | `rootcause.py` | exp1 | root-cause accuracy **0.963** vs **0.540** |
| E6 escalation | `escalation.py`, `quantum.py` | exp4 | gate isolates the hard cases (local accuracy 0.978 when confident vs 0.501 when unsure, 6.9 % escalated); gradient-boosting backend +0.043 on those samples, **quantum-kernel backend −0.198** — see the caution below |
| E7 adaptation inhibition | `governor.py` | exp2 | **0** of 4 500 poisoned samples reached an adaptive element on the attacked channel, vs **4 500** for the comparison system; exp1 `retraining_protection` 0.998 |
| E8 differential response | `actions.py` | exp2 middle column, exp3 row `-differential response` | see below |

### Experiment 2 in one table

A rate-matched injection biases the temperature channel downwards for 4 500
samples; the attacker then withdraws and a genuine over-temperature fault occurs,
against a cloud-side alarm at 34 °C.

| | PIRCAD (full) | PIRCAD, response made non-differential | adaptive recalibrating baseline |
|---|---|---|---|
| poisoning recall | 0.991 | 0.980 | — (no cause vocabulary) |
| poisoned samples admitted to adaptation *on the attacked channel* | **0 of 4 500** | 0 of 4 500 † | **4 500 of 4 500** |
| poisoned samples forwarded | **0.9 %** | 99.8 % | 100 % |
| bias absorbed by the end of the attack | **0.000 °C** | −6.09 °C | −1.46 °C |
| reference error when the real fault arrived | **0.000 °C** | −2.80 °C | **+5.95 °C** |
| security indications raised | 4 627 | 78 | **0** |
| error at the true 34 °C crossing | **+0.04 °C** | −2.76 °C (over-reads) | **+5.99 °C (under-reads)** |
| safety alarm | fired, 2 samples late | fired **99 samples early**, on a temperature the plant never reached | **never fired** |

† The middle column also admits nothing, but for a different reason: treating the
ramp as drift puts the channel in the drifting state, which is itself not
adaptation-eligible. Its failure is visible in the rows below — it *recalibrates*
on attacker data and forwards it. Read those rows, not this one, for that column.

The middle column is the control: identical detector, identical residuals,
identical classifier — only the *response* is made non-differential. Detection is
the same in all three PIRCAD columns. That isolates the contribution of acting on
the cause from the contribution of detecting better, and is the direct evidence
for element E8.

## 7. Technical effects (Section 3(k) / CRI positioning)

Indian practice on computer-related inventions turns on a technical
contribution/effect rather than on an algorithm as such. The effects below are
physical, measured, and tied to hardware and to physical measurements:

1. **Prevention of adaptive-model corruption by injected sensor data.** 0 of
   4 500 injected samples reached any adaptive element on the attacked channel,
   against 4 500 of 4 500 for the comparison system; reference error when the real
   fault arrived 0.000 °C against +5.95 °C.
2. **Preservation of the ability to detect a genuine physical fault after an
   attack.** The cloud-side over-temperature alarm fires (2 samples late, error
   +0.04 °C at the crossing) where the comparison system misses it entirely,
   under-reporting the true temperature by 5.99 °C.
3. **Continued availability of a degraded sensor.** A drifting sensor is corrected
   and kept in service instead of discarded (`drift_kept_flowing` 0.998).
4. **Prevention of a real process fault being erased by recalibration.**
   `process_masking_rate` 0.015.
5. **Reduction of transmitted data** — `bandwidth_saved` 0.228 versus effectively
   zero for the comparison system.
6. **Bounded computational cost at the fog node** — the expensive stage runs on
   0.296 of channel-samples; the nominal path is one model evaluation, one
   subtraction and one chart update per channel.

The claim is hardware-bound (physical sensors, an intermediate fog node, a
centralised system), driven by physical measurements of a physical process, and
its output is a physical control action (calibration of a transducer, suppression
of transmission, generation of an alarm) — not the presentation of a result.

## 8. Non-obviousness argument

Against the search report's "predictable integration of known techniques":

1. **The combination solves a problem D1–D4 do not pose.** None of D1–D4 attempt
   to determine *why* a deviation occurred; all four stop at *that* it occurred.
   A skilled person combining them obtains a better anomaly detector, never a
   differential responder.
2. **A key element runs contrary to conventional design.** Conventional adaptive
   detectors maximise adaptation. E7 deliberately *suspends* adaptation, and
   suspends it at trigger time — before the cause is known — accepting a
   temporarily degraded estimate in exchange for integrity. The teaching in the
   art points the other way.
3. **The decisive discriminator is not a tuning choice but a physical asymmetry.**
   A drifting transducer still observes the real process; a substituted stream
   does not. Turning that into a computable quantity (coherence between residual
   increments and model-predicted increments) is not an obvious step from any of
   D1–D4, none of which has a model-predicted response at all.
4. **Measured effect in the regime where magnitude reasoning is provably useless.**
   In the rate-matched comparison, where drift and injection have identical
   amplitude and identical growth rate, root-cause accuracy is 0.874 versus 0.306,
   and the injection containment rate 0.988 versus 0.000.
5. **The ablation shows no single element carries the result** (exp3, 24 held-out
   episodes, one element removed per row):

   | configuration | root-cause accuracy | injection read as drift | poisoning contained | process fault masked |
   |---|---|---|---|---|
   | **full** | **0.963** | **0.000** | **0.959** | 0.015 |
   | − physics residual (Gen-1 rolling mean instead) | 0.701 | 0.041 | 0.506 | 0.137 |
   | − all residual-character features | 0.877 | 0.093 | 0.690 | 0.067 |
   | − differential response | 0.890 | 0.014 | **0.118** | **0.311** |
   | − cross-sensor consistency | 0.950 | 0.000 | 0.970 | 0.000 |
   | − excitation coherence alone | 0.962 | 0.000 | 0.949 | 0.015 |
   | − noise-signature features alone | 0.954 | 0.000 | 0.857 | 0.014 |

   Two readings matter, and one of them is a correction to how this work was
   initially framed:

   * **The character features are individually redundant and collectively
     necessary.** Removing the excitation-coherence feature alone costs almost
     nothing (0.963 → 0.962) because the noise-signature and onset features carry
     the same information; removing the group costs a great deal (→ 0.877, with
     injections misread as drift rising from 0.000 to 0.093). The coherence
     feature is highly *diagnostic* — it reads −0.036 on drift and +0.430 on a
     rate-matched injection — but it should be claimed as one member of a
     mutually-reinforcing set, not as the load-bearing element.
   * **The differential response contributes almost nothing to accuracy and
     nearly everything to effect.** Removing it costs 0.073 of accuracy, but
     poisoning containment collapses from 0.959 to 0.118 and the rate at which a
     genuine process fault is masked by recalibration rises from 0.015 to 0.311.
     An examiner assessing technical effect should be pointed at those two
     columns, not at the accuracy column.

## 9. Honest limitations, to be disclosed rather than discovered

* The strong separation requires **physical redundancy** — several instruments
  constrained by known physics. exp5 runs the same architecture on the original
  single-channel record: it works, but instrument-vs-process separation is
  unavailable and the margins narrow. Claim 1 should therefore be supported by a
  dependent claim covering single-channel operation without E4.
* An attacker who reproduces the plant's micro-dynamics **and** the controller's
  switching phase in real time defeats the coherence feature. The contribution is
  a large increase in attacker cost, not impossibility. Do not claim otherwise.
* Two hypotheses in the attribution set — a bias on the ambient sensor and an
  unmodelled heat input — are close to collinear given this instrument set. The
  design handles it by escalating rather than guessing; a specification should say
  so.
* Drift is detected with a delay of ~76 samples in the testbed. That delay is
  physics-limited (the bias must exceed the noise floor), not a defect, but it
  should be stated.
* **Escalation does not pay for itself here, and the claims should reflect that.**
  The confidence gate works — it isolates exactly the hard cases (local accuracy
  0.978 on confident samples against 0.501 on the 6.9 % it holds back). But a
  heavier classifier over the *same* feature vector recovers only +0.043 on those
  samples, and end-to-end accuracy is unchanged within noise (0.956 without
  escalation, 0.952 with). The reason is straightforward: the gate is unsure
  because the features are genuinely ambiguous there, and a bigger model over
  ambiguous features stays ambiguous. Escalation is therefore claimed as a
  dependent feature (claims 27–29), and its value should be presented as *bounded
  local cost and no irreversible action on unresolved samples*, not as an accuracy
  gain.
* **The quantum-kernel backend measured *worse* than the local verdict**, not
  better: 0.303 against 0.501 on the escalated samples, a change of −0.198. The
  cause is the compression the near-term kernel requires — 22 residual features
  are projected onto 4 qubits, and the discriminating structure does not survive
  the projection. This is a real result on a working ZZ-feature-map fidelity
  kernel, not a shortcoming of the simulation, and it is consistent with the
  modest accuracies published for NISQ-scale kernel pipelines.

  The practical conclusion is the one to act on: **do not build the filing around
  quantum.** Claims 28–29 remain worth having as narrow dependent claims — they
  cost nothing, they are drafted clear of the Cornell training-side art, and they
  preserve optionality if hardware improves — but the specification must not
  suggest a quantum stage improves detection, because on this evidence it does
  not. The stage is optional, off the real-time path, and the system's degradation
  path when it is absent or slow is the *designed* behaviour, not a fallback.
* The **single-channel** configuration (exp5) is not tuned to the same standard as
  the multi-channel one; its false-alarm rate is high (0.23). Treat exp5 as
  evidence that the architecture degrades gracefully, not as a performance claim.

## 10. Prior art to place before the examiner

Beyond D1–D4, the following were identified as relevant during preparation and
should be cited rather than left to be found:

* Acoustic injection against drift-adaptive MEMS accelerometers (MDPI *Sensors*,
  2024) — establishes the attack this invention defends against.
* Robust drift detection under poisoning (Korycki & Krawczyk, *Machine Learning*,
  2022).
* Kalman-residual sensor fault detection (De Silva, 2020).
* Digital-twin-residual intrusion detection (Sayghe, 2025) — the closest single
  reference; distinguish on **root-cause classification with differential action**,
  not on drift-gated recalibration.
* Physics-informed detection of false-data injection on wind-farm data
  (Alotibi & Tipper).
* Blockchain/federated attestation at the edge (EdgeGuard-IoT, 2025) — do **not**
  rest any claim on attestation; that space is occupied.
* Quantum ML for IoT intrusion detection: QSVM+IGWO (Elsedimy, *Cluster
  Computing*, 2024); quantum autoencoder + quantum kNN (Hdaib, 2024); and
  US20230094389A1 / US12566987B2 (Cornell), which claims neural-network fault
  detection trained using **quantum sampling** — claim 28 is drafted
  inference-side and gate-model to avoid it.

## 11. Reproducing every number in this document

```bash
python experiments/run_all.py --quantum
```

Numbers land in `artifacts/*.json`; the exact configuration that produced them is
in `pircad/config.py`. Training seeds and evaluation seeds are disjoint and are
declared in `experiments/_common.py`.
