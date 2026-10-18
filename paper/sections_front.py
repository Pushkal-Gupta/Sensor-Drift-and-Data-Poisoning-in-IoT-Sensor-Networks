"""Front matter and Sections 1-5 (no result-dependent numbers live here)."""

from __future__ import annotations

from pathlib import Path

HERE = Path(__file__).resolve().parent

TITLE = ("Ageing or Attack? Cause-Aware Learning for Safe Self‑Calibration Under "
         "Disguised Sensor Poisoning")
# Author block laid out as in the authors' original draft (Paper.docx).
AUTHORS = ("Pushkal Gupta^{1}, Shridhar M^{2}, Vanka Kaushik^{3}, "
           "Syamasudha Veeragandham^{4}")
AFFIL = ("^{1,2,3,4}School of Computer Science and Engineering, Vellore Institute of "
         "Technology, Vellore, Tamil Nadu, India – 632014")
EMAILS = ("Email: ^{1}pushkal.gupta2023@vitstudent.ac.in, "
          "^{2}shridhar.m2023@vitstudent.ac.in, ^{3}vanka.kaushik2023@vitstudent.ac.in, "
          "^{4}syamasudha.v@vit.ac.in")
CORR = "^{4}Corresponding author"

KEYWORDS = ("Sensor drift; Data poisoning; Self-calibration; Root-cause "
            "classification; Physics-informed machine learning; Adversarial robustness; "
            "Internet of Things; Fog computing")


def methods(m, R):
    """Sections 1-5.  `R` carries the few headline numbers the introduction quotes."""
    # ================================================================ 1
    m.h1("1. Introduction")
    m.p("Networked sensing underpins industrial monitoring, building automation and "
        "environmental observation, and every analytic, control decision and safety "
        "interlock computed downstream inherits the trustworthiness of the stream it is "
        "given {cite:mishra2024,xing2024}. In practice these streams come from many "
        "low-cost transducers left to run unattended, and their readings can become "
        "wrong for two quite different reasons. The first is ageing. Over time a "
        "transducer picks up a slow, smooth offset, yet it keeps measuring the process, "
        "so the right action is to recalibrate it, not to throw it away "
        "{cite:ahmad2024,schaller2025}. The second is attack. False-data injection "
        "produces values that each look reasonable on their own "
        "{cite:xing2024,shen2024}, and acoustic or thermal tampering at the sensor "
        "itself happens before any encryption on the link can help "
        "{cite:sahu2025,ino2024,morand2024}.")
    m.p("Ageing and attack need opposite responses, but an attacker can make one look "
        "like the other. An offset that rises by a few hundredths of a degree each "
        "minute is what an old thermistor does, and it is also what a careful attacker "
        "would inject. The problem is sharpened by the fact that IoT pipelines now "
        "*adapt*. They recalibrate on the data they have accepted {cite:ahmad2024}, "
        "update their baselines and thresholds {cite:lee2025}, and absorb concept drift "
        "so that it stops raising alarms {cite:xu2024,gama2014}. An adaptive element that treats every "
        "slow, persistent deviation as benign drift will absorb an injection shaped like "
        "drift into its own reference, and the damage outlasts the attack: once the "
        "reference has moved, genuine faults are under-reported. This has been shown "
        "against drift-adaptive detectors in industrial control "
        "{cite:kravchik2021,kravchik2022} and, physically, by acoustically poisoning the training data of an on-device "
        "anomaly detector until the machine fault it existed to detect became "
        "invisible {cite:ino2024}.")
    m.p("The dominant machine-learning formulation does not pose the question that "
        "would prevent this. Time-series anomaly detectors emit a score or a flag "
        "{cite:darban2025,yap2024,reis2025}; drift detectors emit a change point "
        "{cite:gama2014}. Neither carries a vocabulary of *causes*, so a system built on "
        "them has nothing on which to condition its response, and it does the same "
        "thing whatever the reason for the deviation. Recent work has begun to separate "
        "faults from attacks {cite:liu2025,hu2024,shen2024,shukla2023} and adversarial "
        "from real concept drift {cite:korycki2023}, but only between hypotheses that differ "
        "in their signature at the plant, not against an injection deliberately matched "
        "to benign ageing; and none of it closes the loop from the verdict to *which "
        "adaptive elements may learn from the data*.")
    m.p("This paper treats the problem as one of **cause-aware learning under "
        "adversarial distribution shift**. We ask a fog-layer learner to decide, for "
        "every deviation, whether it is benign transducer drift, data poisoning, or a "
        "genuine excursion of the monitored process; and to let that verdict gate every "
        "element that adapts. The approach, PIRCAD (physics-anchored root-cause "
        "attribution with adaptation governance), rests on one structural idea: each "
        "channel is judged against an expectation that it cannot influence. For each "
        "channel we build that expectation from the other instruments and the process "
        "physics only (a leave-one-channel-out, or dedicated-observer, estimator). A "
        "slow bias on the channel then shows up in the residual at its full size. Size "
        "and rate cannot tell ageing from attack, but the *character* of the residual "
        "can: a drifting transducer still sees the real process, while a substituted "
        "stream does not.")
    m.p("The contributions are:")
    m.bullets([
        "**A formulation and threat model** for drift-versus-poisoning attribution in "
        "which the adversary matches benign ageing in magnitude and rate, organised as "
        "a hierarchy of attacker knowledge. We state the limit of the problem "
        "explicitly: an on-path attacker that adds a ramp to the genuine signal is "
        "observationally equivalent to drift on the attacked channel (Proposition 1), "
        "and we show that a physics-anchored, bounded correction nevertheless bounds "
        "the damage such an attacker can do (Proposition 2).",
        "**A physics-anchored root-cause learner**: leave-one-out residuals, a compact "
        "set of residual-character features, and a localisation of physical-constraint "
        "violations whose hypothesis signatures come from perturbing the model rather "
        "than from labelled faults. The learner is back-end agnostic; we report five.",
        "**Adaptation governance**: a per-channel state machine that withholds suspect "
        "data from every adaptive element, with a cause-dependent response — bounded "
        "recalibration for drift, quarantine for poisoning, forward-and-alarm for "
        "process events — and a rule that no irreversible action (quarantine) is "
        "taken on an uncertain verdict, so that one misclassification cannot lock a "
        "serviceable sensor out of correction.",
        "**A statistically powered evaluation** against supervised deep and "
        "feature-based learners, unsupervised detectors and tuned self-calibrating "
        "estimators: 30 held-out seeds with paired tests, an attacker-capability sweep, "
        "eleven distribution-shift conditions without retraining, an end-to-end "
        "safety experiment, and a real-data study on the Intel Berkeley Research Lab "
        "deployment with sensor-disjoint train and test splits {cite:intel2004}.",
        "**Reproducibility**: all code, the attack generators and the scripts that "
        "produce every number and figure are released; Section 7.2 also documents three "
        "defects that the stronger evaluation exposed in our own earlier pipeline, and "
        "their fixes.",
    ])
    m.p("Section 2 reviews related work. Section 3 formalises the problem and the "
        "threat model. The method is given in Section 4 and the experimental design in "
        "Section 5, and Section 6 presents the results. Section 7 discusses limitations and threats "
        "to validity, and Section 8 concludes.")

    # ================================================================ 2
    m.h1("2. Related work")
    m.p("Fog-, edge- and on-device anomaly detection place lightweight learners next "
        "to the sensors {cite:reis2025,yap2024}, and deep time-series anomaly "
        "detection has been surveyed comprehensively {cite:darban2025}; work on the "
        "decision stage makes thresholds adaptive and uncertainty-aware {cite:lee2025}. "
        "Isolation-based scoring {cite:liu2008iforest} and sequential change detection "
        "{cite:page1954} remain the low-cost workhorses. These methods improve the "
        "score, not its semantics: the output is a single anomaly label, so there is "
        "nothing on which to condition a cause-specific response.",
        lead="Anomaly and change detection.")
    m.p("Field recalibration of low-cost sensor networks {cite:ahmad2024} and "
        "learned drift compensation {cite:schaller2025} treat slow, persistent, "
        "one-directional bias as something to absorb — correctly, when the cause "
        "really is ageing — and drift-aware IoT detectors adapt to configuration "
        "changes to suppress false positives {cite:xu2024}; concept-drift adaptation "
        "more broadly is surveyed in {cite:gama2014}. None of this literature carries "
        "an adversarial hypothesis, so absorbing an injection shaped like ageing lies "
        "outside its threat model.",
        lead="Drift compensation and adaptation.")
    m.p("Stealthy false-data injection against state estimation {cite:liu2009fdi} "
        "and replay attacks {cite:mo2009} define the classical adversary; surveys "
        "review detectors {cite:xing2024}, and random-matrix methods separate injection "
        "from replay {cite:shen2024}. Learned detectors are themselves attackable: "
        "detectors that predict each sensor from the others can be evaded "
        "{cite:ghafouri2018}, and gradient-crafted injections are misclassified as "
        "legitimate faults {cite:saber2025}. Most directly relevant to this paper, "
        "detectors retrained online to follow drift can be poisoned by fake sensor "
        "signals so that later attacks go undetected {cite:kravchik2021,kravchik2022}; "
        "drift detectors can be evaded by adversarially constructed drift "
        "{cite:hinder2025}; on-device learners have been poisoned physically "
        "{cite:ino2024}; and thermal attacks have exploited the temperature-dependent "
        "internal calibration of low-cost pollution sensors {cite:morand2024}. "
        "Outside IoT, a physical attack on autonomous vehicles first triggers "
        "camera–LiDAR online recalibration and then steers the estimate "
        "{cite:liu2026calib} — the same exploit of a self-calibrating element. "
        "Poisoning defences for federated learners {cite:yang2024} protect a training "
        "corpus, not the calibration offsets and normalisation statistics of a fog "
        "pipeline.",
        lead="Attacks on sensing and on adaptive learners.")
    m.p("Model-based diagnosis generates residuals against a process model and "
        "exploits analytical redundancy {cite:isermann2005}; observer-based residuals "
        "remain effective for data-efficient diagnosis {cite:ullah2025}. "
        "Physics-informed learning brings known laws into anomaly detection "
        "{cite:wu2024,zideh2024}, invariant rules among process variables detect "
        "attacks in industrial control {cite:zhu2025}, and digital twins act as "
        "observers {cite:sayghe2025,kallesoe2024}. The nearest edge system, PhySec-Edge, "
        "combines process-model residuals, Kalman estimation and cross-sensor checks "
        "with learned detectors {cite:radovanovic2026}; it detects attack classes, but "
        "benign ageing is not among its hypotheses and the residual is not used to "
        "select a response.",
        lead="Physics-informed and model-based detection.")
    m.p("The closest work in objective separates which *kind* of anomaly occurred: "
        "attacks from faults in interconnected cyber-physical systems {cite:liu2025}, "
        "replay attacks from sensor faults {cite:hu2024}, malicious anomalies from "
        "sensor faults and genuine events with an ensemble pipeline {cite:shukla2023}, "
        "and adversarial from real concept drift in generic data streams "
        "{cite:korycki2023}. Table 1 positions these against PIRCAD. Three properties "
        "are, to our knowledge, not combined elsewhere: an injection matched to benign "
        "ageing in *both* magnitude and rate as the decisive hypothesis; attribution "
        "grounded in physical constraints whose signatures need no labelled faults; and "
        "the verdict gating *every* adaptive element, evaluated by its end effect on a "
        "downstream safety function rather than by label accuracy alone.",
        lead="Distinguishing faults, attacks and drift.")
    m.table(
        "Positioning against the closest prior work. ✓ = addressed; ~ = partially; – = not addressed.",
        ["Work", "Ageing-matched attack as hypothesis", "Drift / attack / process event",
         "Physics-grounded attribution", "Cause-dependent response", "Gates adaptation",
         "Real data"],
        [
            ["Drift compensation {cite:ahmad2024,schaller2025}", "–", "drift only", "–", "recalibrates", "–", "✓"],
            ["Drift-aware IoT AD {cite:xu2024}", "–", "–", "–", "adapts", "–", "✓"],
            ["Poisoning of online ICS detectors {cite:kravchik2021,kravchik2022}", "~ (attack study)", "–", "–", "–", "–", "✓"],
            ["Adversarial vs real drift {cite:korycki2023}", "~", "drift vs adversarial", "–", "–", "~", "✓"],
            ["Fault vs attack {cite:liu2025,hu2024}", "–", "fault vs attack", "✓ (observers)", "–", "–", "–"],
            ["Malicious vs fault vs event {cite:shukla2023}", "–", "✓", "–", "–", "–", "✓"],
            ["PhySec-Edge {cite:radovanovic2026}", "–", "attack classes", "✓", "–", "–", "~"],
            ["**PIRCAD (this work)**", "**✓ (hierarchy A0–A5)**", "**✓**", "**✓**", "**✓**", "**✓ (all elements)**", "**✓ (semi-synthetic)**"],
        ], widths=[3.6, 2.2, 2.2, 2.0, 2.0, 1.8, 1.6], font=8)

    # ================================================================ 3
    m.h1("3. Problem formulation and threat model")
    m.h2("3.1 Setting")
    m.p("A fog node receives synchronous samples y(k) = [y_{1}(k), …, y_{C}(k)] from C "
        "transducers observing a common physical process with state x(k) and known "
        "inputs u(k) (actuator commands, or exogenous measurements such as neighbouring "
        "sensors). In nominal operation y_{i}(k) = h_{i}(x(k)) + e_{i}(k), with e_{i} "
        "the transducer noise. The node forwards samples to a cloud tier and maintains "
        "a set 𝒜 of *adaptive elements*: calibration offsets c_{i}, residual "
        "normalisation statistics, the state of its process model, and any learned "
        "model that is retrained on accepted data. Each sample of each channel carries "
        "a latent cause z_{i}(k) ∈ {NORMAL, DRIFT, POISON, PROCESS}:")
    m.bullets([
        "**DRIFT** — the transducer degrades: y_{i}(k) = h_{i}(x(k)) + e_{i}(k) + "
        "b_{i}(k), with b_{i} smooth and |ḃ_{i}| ≤ ρ_{i}, the manufacturer's ageing "
        "envelope. The sensor still observes the real process.",
        "**POISON** — the reported value is replaced or modified by an adversary "
        "(Section 3.2).",
        "**PROCESS** — the plant itself departs from nominal (e.g. an actuator fails); "
        "the instruments remain healthy and mutually consistent.",
    ])
    m.p("The learning task is to infer ẑ_{i}(k) causally and to choose an action "
        "a_{i}(k) = π(ẑ_{i}(k)) and an adaptation mask g_{i}(k) ∈ {0, 1} that decides "
        "whether sample k of channel i may update any element of 𝒜. The objective is "
        "not label accuracy alone but its consequences: drifting sensors should stay "
        "in service with their bias removed, poisoned samples should neither be "
        "forwarded nor reach 𝒜, and process events should be forwarded and alarmed "
        "without ever being calibrated away.")
    m.h2("3.2 Threat model: an attacker-capability hierarchy")
    m.p("The adversary controls the reported stream of one channel (or, at level A5, "
        "several) and wants its injection absorbed as drift. Throughout, the injected "
        "deviation is **rate-matched** to benign drift — the same ramp rate, within "
        "the ageing envelope — so any detector reasoning about magnitude, persistence "
        "or growth rate is at chance by construction. The levels differ in what the "
        "attacker knows about the live process:")
    m.bullets([
        "**A0 (stale surrogate).** The attacker replaces the reading with a delayed "
        "copy of the process (an exponential moving average, factor 0.02), adds the ramp, "
        "and adds white noise with the same variance as the transducer.",
        "**A1 (noise-matched).** As A0, but the noise reproduces the transducer's "
        "coloured (AR(1)) spectrum.",
        "**A2 (higher fidelity).** The surrogate tracks the live process with smoothing "
        "factor 0.05–0.30.",
        "**A3 (exact state).** The attacker has the true process state in real time "
        "(factor 1.0); only the transducer's own noise is unknown to it.",
        "**A4 (on-path).** Sitting on the link, the attacker leaves the *genuine* "
        "reading in place and simply adds the ramp to it.",
        "**A5 (coordinated).** The attacker controls T, P and RH and derives all three "
        "from a single forged temperature, so the physical relations between the "
        "instruments still hold.",
    ])
    m.p("We also test two attack families that are not rate-matched: isolated spike "
        "substitutions, and replay of an earlier genuine window. On the defender's side "
        "we assume a clean commissioning window and a reasonably accurate process "
        "model; at levels A0–A4 we assume the attacker cannot touch the inputs u(k).")
    m.h2("3.3 Limits of separability")
    m.p("**Proposition 1 (observational equivalence).** Let an A4 attacker add the "
        "ramp b(k) = ρ(k − k_{0}) with |ρ| inside the ageing envelope. The sequence the "
        "attacked channel reports then has exactly the same distribution as that channel "
        "would have under benign drift with b_{i} = b. *Proof.* Both equal "
        "h_{i}(x(k)) + e_{i}(k) + ρ(k − k_{0}) with the same noise realisation; no "
        "function of {y_{i}(k)} can separate them, and the other channels are "
        "untouched in both cases. ∎")
    m.p("Proposition 1 fixes the frontier of the problem: no detector — physics-based "
        "or learned — can label A4 correctly from the data alone. What a system *can* "
        "control is the consequence of mislabelling it.")
    m.p("**Proposition 2 (bounded harm of anchored correction).** Let the "
        "calibration offset c_{i} be driven only by residuals against an expectation "
        "ŷ_{i}(k | y_{−i}(k), u(k)) that excludes channel i, with per-sample rate limit "
        "δ and total cap c_{max}. Suppose an A4 ramp b(k) is labelled as drift. Then "
        "the corrected value that is forwarded obeys |y_{i} − c_{i} − h_{i}(x)| ≤ "
        "|ε_{ref}| + |e_{i}| + max(0, |b| − c_{max}) + L, with ε_{ref} the error of the "
        "reference and L ≤ max(0, |ḃ| − δ)·t the lag of the tracker. *Proof.* The "
        "expectation does not use channel i, so the residual is an estimate of "
        "b + ε_{ref}, and the corrector subtracts it within its rate and cap limits. ∎")
    m.p("This is why the response matters as much as the classifier. An attack that "
        "can only be labelled drift is still corrected against the physics, instead of "
        "being folded into a baseline built from the channel itself. A corrector that "
        "follows the channel's own history (a rolling mean, an EWMA reference, or "
        "recalibration triggered by Page-Hinkley) does the opposite: it follows b, and "
        "the whole bias reaches the forwarded value. Both claims are tested in "
        "Sections 6.2 and 6.4.")

    # ================================================================ 4
    m.h1("4. Method")
    m.figure(HERE / "figures" / "architecture.png",
             "Structure of PIRCAD on the fog node. Stages E4, E5 and E8 are costly and "
             "run only after the lightweight trigger E3 fires. The adaptation governor "
             "E7 decides what each learning element may use: the process-model state, "
             "the residual statistics and the calibration offsets.", 15.5)
    m.h2("4.1 Leave-one-out physics residuals (E1–E2)")
    m.p("For each channel i the expected value is produced by a dedicated observer "
        "whose state excludes that channel:")
    m.eq("r_{i}(k) = y_{i}(k) − c_{i}(k) − ŷ_{i}(k | y_{−i}(k), u(k))")
    m.p("A single closed-loop estimator that also uses channel i tracks a slow bias "
        "on that channel, so its innovation stays at the noise floor; excluding the "
        "channel makes the bias appear in r_{i} at full amplitude (with a retrained "
        "classifier the practical effect is small; Section 6.6). Each observer is "
        "re-anchored only on channels the governor currently trusts, with weights that "
        "are slewed rather than switched (a step change in an anchor's weight would "
        "itself appear as a jump in other channels' residuals). Residual location and "
        "scale are learned on a clean commissioning window and then frozen: an adaptive "
        "scale is precisely the handle a slow injection would use to normalise itself "
        "away.")
    m.p("In the simulated testbed (Section 5.1) the process is a heated, sealed vessel "
        "with a bang-bang controller, and the observers combine Newtonian heating, the "
        "isochoric ideal-gas law P ∝ T and conservation of absolute humidity through "
        "the Magnus relation {cite:alduchov1996}:")
    m.eq("e_{s}(T) = 610.94 · exp(17.625 T / (T + 243.04)),   RH = 100 e / e_{s}(T)")
    m.p("In the real-data embodiment (Section 5.2) there is no actuator; the expectation "
        "of a mote's temperature fuses the Magnus inversion of its own humidity at the "
        "neighbours' vapour pressure with the neighbours' mean temperature "
        "(inverse-variance weights fixed during commissioning), and the expectation of "
        "its humidity follows from its temperature and the neighbours' vapour pressure.")
    m.h2("4.2 Residual-character features (E2)")
    m.p("A rate-matched injection and benign drift have, by construction, the same "
        "residual magnitude and slope. The features therefore describe *how* the "
        "residual evolves (Table 2). Their physical rationale is that a drifting "
        "transducer still observes the real process: its innovation statistics and its "
        "response to the modelled micro-dynamics are preserved, whereas a substituted "
        "stream carries the attacker's noise and does not follow the plant. The most "
        "legible example is excitation coherence,")
    m.eq("κ_{i}(k) = − corr_{W}( Δr_{i}, Δŷ_{i} )")
    m.p("the negated correlation, over a window W, between residual increments and "
        "model-predicted increments: near zero for a sensor that tracks the real "
        "dynamics, positive for a stream whose increments do not. Forging it requires "
        "the attacker to reproduce the plant's micro-dynamics in real time — which is "
        "exactly the knowledge axis of the hierarchy in Section 3.2.")
    m.table(
        "Feature vector of the root-cause learner (22 features).",
        ["Group", "Features", "What separates the causes"],
        [
            ["Size", "z_now, z_abs_max, r_mean_norm, r_std_norm, slope_norm, persistence, exceed_frac",
             "Trigger context only; uninformative between rate-matched causes by design"],
            ["Ageing envelope & onset", "slope_ratio (vs. ρ_{i}), monotone_frac, corner, jump_max",
             "Ageing is smooth, one-directional and inside ρ_{i}; an injection has an onset"],
            ["Noise signature", "acf_shift, var_ratio, flatness, hf_ratio, kurt_incr",
             "Changes in the innovation statistics relative to commissioning"],
            ["Dynamics", "dyn_coherence (κ)", "Does the stream still follow the modelled micro-dynamics?"],
            ["Localisation (E4)", "loc_score, loc_margin, loc_magnitude, loc_is_self, loc_is_process",
             "Which hypothesis best explains the constraint-violation direction"],
        ], widths=[3.0, 6.5, 6.5])
    m.h2("4.3 Physics-signature localisation (E4)")
    m.p("A vector g(k) of physical-constraint violations is evaluated on the reported "
        "measurements and normalised by commissioning statistics (vessel: gas law, "
        "humidity conservation, open-loop dynamics, energy balance, regulation; indoor: "
        "Magnus consistency, spatial temperature, spatial humidity). For each "
        "hypothesis h — a unit bias on each channel, or an unmodelled process input — a "
        "signature s_{h} is obtained by perturbing the *model* at the commissioning "
        "operating point, so no labelled fault data are needed. Localisation reports "
        "the hypothesis that best explains the direction of the windowed mean "
        "violation ḡ, its share of the violation energy and its margin over the "
        "runner-up:")
    m.eq("h* = argmax_{h} ( s_{h}ᵀ ḡ / ‖ḡ‖ )²")
    m.p("Localisation does not decide between drift and poisoning. It asks a "
        "different question: did an instrument go wrong, or did the plant itself "
        "change? Getting this right is what stops a real process event from being "
        "calibrated out. Indoors, for instance, a local heat source shifts temperature "
        "and humidity together while vapour pressure stays fixed, so the Magnus "
        "relation still holds; a biased transducer breaks it.")
    m.h2("4.4 Root-cause learner (E5)")
    m.p("A four-class classifier takes the feature vector as input. To build its "
        "training set we run the full pipeline with the classifier switched off, which "
        "guarantees that the features seen in training match those seen at inference. "
        "Channels other than the target are labelled NORMAL, so the learner learns to "
        "ignore triggers caused by a neighbour's fault. The hybrid version adds a small "
        "physical override. When localisation points confidently at the process, the "
        "verdict is PROCESS, since a mistake there affects safety; when the violations "
        "point clearly at the channel itself, a PROCESS verdict is rejected. Any "
        "classifier can be used. We test a random forest {cite:breiman2001} (the "
        "default), gradient boosting, logistic regression, a multilayer perceptron and "
        "a rule set that needs no training. A fog node classifies samples one at a "
        "time, so we store the forest as flat arrays and walk all trees in one "
        "vectorised pass. The output matches the library implementation exactly (the "
        "test suite checks this) and is about 200× faster per sample.")
    m.h2("4.5 Adaptation governance and differential response (E7–E8)")
    m.p("Every channel is in one of five states: TRUSTED, WATCH, DRIFTING, "
        "QUARANTINED or RECOVERING. As soon as its trigger fires, a channel drops out of "
        "TRUSTED, even before a cause has been assigned. From that point it carries less "
        "weight in the process model and no longer updates the residual statistics. "
        "What happens next depends on the cause:")
    m.bullets([
        "**DRIFT** → bounded, rate-limited, audited recalibration "
        "c_{i} ← clip(c_{i} + clip(γ b̂_{i}, ±δ), ±c_{max}), with b̂_{i} the robust "
        "residual bias; data keep flowing. The channel stays in the drift state while "
        "the correction is active: a residual kept at zero by the correction shows that "
        "the correction is working, not that the sensor has healed.",
        "**POISON** → quarantine. The node forwards nothing, freezes all adaptive "
        "elements and raises a security alert, and the channel must wait out a minimum "
        "hold before it can recover.",
        "**PROCESS** → the data are forwarded with an alarm, and no calibration is "
        "allowed.",
    ])
    m.p("**Quarantine policy.** A drift verdict cannot release a quarantined channel; "
        "if it could, an attacker would only need to push one verdict towards drift to "
        "get recalibrated. The node also follows the governor's current state, not the "
        "most recent verdict, so a quarantined channel forwards nothing even when it is "
        "labelled drift. Quarantine is *latched* by default, meaning the channel comes "
        "back only after the freeze hold has passed and a clean run has been seen. We also evaluate an "
        "availability-oriented alternative, *re-attribution*, in which a quarantined "
        "channel moves to DRIFTING after the hold plus a sustained run of 200 "
        "consecutive drift verdicts; Section 6.4 shows why latching is the safer "
        "default.")
    m.p("**Uncertain verdicts.** A verdict below the confidence threshold (0.62) is "
        "escalated to a heavier back-end when one is configured. Without one, an "
        "uncertain POISON verdict holds that sample — it is neither forwarded nor "
        "admitted to any adaptive element — but it is recorded by the governor as "
        "UNKNOWN and cannot latch a quarantine; only a confident POISON verdict does. "
        "Uncertain DRIFT and PROCESS verdicts act locally, since their actions are "
        "bounded and reversible. This realises the rule that no irreversible action is "
        "taken on an ambiguous sample.")
    m.h2("4.6 Cost")
    m.p("Per sample and channel, the nominal path is one model evaluation, a "
        "subtraction and a CUSUM update; features, localisation and classification run "
        "only on engaged channels, at a configurable re-evaluation interval (a fresh "
        "trigger always forces re-evaluation). Section 6.7 reports measured latency.")

    # ================================================================ 5
    m.h1("5. Experimental design")
    m.h2("5.1 Simulated benchmark")
    m.p("The testbed is a heated vessel (thermal capacitance 5 kJ/K, loss 20 W/K, "
        "400 W heater, bang-bang control at 30 °C with 0.3 °C hysteresis, sinusoidal "
        "ambient with a random-walk component) instrumented with internal temperature "
        "T_{in}, pressure P, humidity RH and ambient temperature T_{amb}, each with "
        "coloured AR(1) noise and quantisation. The fog node's model is deliberately "
        "mismatched (thermal capacitance +3 %, loss −2.5 %, heater +1 %, reference "
        "pressure +0.08 %, vapour pressure +1.5 %). Six scenarios are generated: clean, "
        "benign drift, rate-matched poisoning (A0), spike poisoning, replay poisoning "
        "and a process event (heater contactor welded on or failed open). Onset and rate "
        "are drawn per episode (onset in the second quarter of the episode; rate "
        "0.8–2.5 × 10^{−3} units per sample), and the event polarity — drift or ramp "
        "direction, heater on or off — is balanced. Episodes are 4000 samples, of which "
        "the first 600 are a scored-out warm-up.")
    m.p("**Splits.** Models are trained on 6 seeds × 6 scenarios × 2 polarities "
        "(72 episodes, {n_corpus} labelled channel-samples) and evaluated on 30 "
        "disjoint seeds × 6 scenarios (180 episodes). Every reported number comes from "
        "the held-out bank.".replace("{n_corpus}", R.get("n_corpus", "≈510 000")))
    m.h2("5.2 Real data: Intel Berkeley Research Lab")
    m.p("The Intel Berkeley Research Lab deployment {cite:intel2004} recorded "
        "temperature, humidity, light and battery voltage from 54 Mica2Dot motes in an "
        "office between February and April 2004. We bin the ≈31 s epochs into "
        "2-minute samples, interpolate gaps of up to 30 minutes, and use days 0–10, "
        "before battery failures make readings meaningless; 51 motes have ≤2 % missing "
        "data after interpolation. The clean signal — with its real micro-climate, "
        "occupancy and packet loss — is therefore real. Ground truth for drift and "
        "poisoning can only exist if events are injected, so they are injected into the "
        "real temperature streams (semi-synthetic evaluation): drift and A0 poisoning "
        "at 0.6–2.0 × 10^{−3} °C per sample, spikes, a replay of the previous day, and a "
        "genuine-process surrogate — a local heat source or draught that changes "
        "temperature by 1.5–4 °C and humidity exactly as psychrometrics requires at "
        "constant vapour pressure. Each mote uses its four nearest neighbours as "
        "physical references. **Motes are split at random into disjoint train and "
        "test halves**; every model is fitted on train motes only. Clean test episodes "
        "measure false alarms on unmodified real data.")
    m.h2("5.3 Comparison systems")
    m.p("All learned baselines are trained on the same episodes with the same four "
        "labels and see at least the information PIRCAD sees — every raw channel plus "
        "the actuator command or neighbour aggregates — but no physics (Table 3). "
        "Self-calibrating baselines were tuned to minimise their own correction error "
        "on benign drift (training seeds only), so that none can appear safe merely by "
        "not adapting.")
    m.table(
        "Comparison systems.",
        ["System", "Type", "Input", "Output"],
        [
            ["Multi-scale 1D-CNN", "Supervised deep learner; dilated convolutions over the last 128 samples and an 8× pooled 1024-sample context",
             "All raw channels + command, standardised on commissioning", "4 causes"],
            ["Generic-feature RF", "Random forest over multi-scale window statistics incl. rolling cross-channel correlations",
             "As above", "4 causes"],
            ["Gen-1 FHAD", "Rolling-mean detrending + CUSUM + windowed classifier (authors' earlier pipeline)",
             "Target channel", "4 causes"],
            ["Isolation Forest {cite:liu2008iforest}, PCA-SPE, autoencoder", "Unsupervised detectors on generic features / raw windows, 1 % clean FPR",
             "All channels", "anomaly flag"],
            ["Kalman χ² {cite:kalman1960}", "Single closed-loop observer over all channels, windowed NIS test",
             "All channels + command", "anomaly flag"],
            ["EWMA self-recalibration", "EWMA reference; z-test-triggered absorption", "Target channel", "corrected value"],
            ["Kalman bias estimator", "Augmented state [T, b], same process model as PIRCAD", "Target + command + ambient", "corrected value"],
            ["Page-Hinkley recalibration {cite:page1954}", "Detect persistent shift → recalibrate → restart", "Target channel", "corrected value"],
        ], widths=[3.4, 6.0, 4.0, 2.4], font=8)
    m.h2("5.4 Metrics and statistics")
    m.p("Root-cause accuracy and **macro-F1** over the classes present in an episode "
        "are computed per sample after warm-up. Two directional errors are reported "
        "separately because they have opposite consequences: the **false-drift rate** "
        "(poisoned samples labelled drift — the attack is recalibrated into the "
        "reference) and the **false-poison rate** (drifting samples labelled poisoned — "
        "a serviceable sensor is discarded); the **process-masking rate** is the "
        "fraction of genuine process-event samples labelled as an instrument fault. "
        "Effect-level metrics measure what the system *does*: **poisoning containment** "
        "(poisoned samples not forwarded), **retraining protection** (1 − poisoned "
        "samples admitted to any adaptive element / poisoned samples) and **forwarded "
        "error** (mean absolute error between the forwarded, corrected value of the "
        "target channel and its true physical value over the event window, with "
        "withheld samples contributing zero). All metrics are computed per episode; we "
        "report means with 95 % bootstrap confidence intervals (4000 resamples) and "
        "compare systems with paired two-sided Wilcoxon signed-rank tests "
        "{cite:wilcoxon1945} matched by episode, with Holm correction {cite:holm1979} "
        "and rank-biserial effect sizes.")
