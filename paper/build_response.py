"""Point-by-point response to the first-round reviews (Array, ARRAY-D-26-07637).

    python paper/build_response.py

Every number is read from the same artifacts as the manuscript.
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from docbuilder import Manuscript  # noqa: E402
from references import REFS  # noqa: E402
import sections_results as res  # noqa: E402
from build_manuscript import verdicts  # noqa: E402


def _dry_run(R):
    import sections_front as front
    d = Manuscript(REFS)
    front.methods(d, R)
    res.results_section(d, R)


def comment(m, tag, text):
    p = m.p(text, lead=tag)
    for r in p.runs[1:]:
        r.italic = True


def main():
    R = res.collect()
    R["variant_verdict"], R["variant_summary"], R["aug_verdict"] = verdicts(R)
    _dry_run(R)            # fills in the table numbers the letter cites
    S6, S9, S10 = R["e6"]["summary"], R["e9"]["summary"], R["e10"]["summary"]
    D9 = R["e9"]["drift_summary"]
    th = R["e14thr"]
    ref = R["e14ref"]
    pol = R["e15"]["polarity"]
    DT = R["e15"]["intel_shift"]["drop_tests"]
    safe = R["e14safe"]["by_attacker"]
    kb = R["best_firing"]["Kalman bias estimator"]
    c16 = R["e16"]
    e17 = R["e17"]["results"]

    m = Manuscript(REFS)
    m.title("Response to the reviewers")
    m.centered("Manuscript ARRAY-D-26-07637 — “Ageing or Attack: Cause-Aware Learning for "
               "Safe Self-Calibration Under Disguised Sensor Poisoning”", 11, after=10)
    m.p("We thank both reviewers for their careful reading. Reviewer 1's comments in "
        "particular identified places where our framing claimed more than our evidence "
        "supports, and several analyses that were missing. We have revised the paper in "
        "two directions: we narrowed the claims to what the evidence shows, and we ran "
        "the additional experiments needed to test the claims we kept. Every new number "
        "is produced by released scripts (experiments/exp14–exp17) from released "
        "per-episode artifacts. In summary:")
    m.bullets([
        "**Framing.** The abstract, highlights, contributions, discussion and conclusion "
        "now lead with the governance result and state plainly that PIRCAD is not the "
        "most accurate classifier: the generic-feature forest beats it in distribution, "
        "and both supervised baselines beat it on real data. The “bounded worst case "
        "under shift” is no longer presented as a property of the method; it is "
        "reported as a simulation finding that did not reproduce on real data. A new "
        "“Scope of the claims” paragraph (Section 1) and a “What is not defended” "
        "paragraph (Section 3.2) make the conditions explicit.",
        "**Theory.** Proposition 1 is now Observation 1, with an explanation of why it "
        "is stated (it makes attribution ill-posed for A4 by design, so only the "
        "consequence can be evaluated). Proposition 2 is restated as a bound *relative "
        "to the reference error*, and that reference error is now measured in "
        "simulation and on the Intel motes.",
        "**New experiments.** (i) The safety experiment repeated for every ablated "
        "variant, for the gradient-boosting back-end, for two other confidence "
        "thresholds, and for the A4 and A5 attackers; (ii) the self-calibrating "
        "baselines swept over their adaptation parameter; (iii) dose-response over "
        "attack rate, attack duration and alarm threshold; (iv) sensitivity to the "
        "confidence threshold on simulated and real data; (v) a per-polarity breakdown "
        "of the held-out bank; (vi) the leave-one-out reference error on clean "
        "simulated and real data; (vii) noise-augmented training as a mitigation of "
        "the noise-shift weakness; (viii) memory and throughput figures.",
        "**Statistics.** Confidence intervals and Holm-corrected paired tests for the "
        "attacker sweep, the distribution-shift table and the real-data noise shift; "
        "rank-biserial effect sizes in every comparison table.",
        "**Negative results we found in revision and now report.** The on-path (A4) "
        "attacker, corrected as drift while it lasts, leaves an over-correction when it "
        "withdraws (Section 6.4); PIRCAD's false-poison rate on real data is "
        f"{S10['PIRCAD-rf']['false_poison_rate'][0]:.2f}, which the original Table 9 did "
        "not show.",
        "**Editorial.** Fig. 1 redrawn with the stages in order (E6 was missing); "
        "metric terminology aligned between highlights and text; the reference list "
        "repaired so that every in-text number points to the work it names.",
    ])

    # =================================================================== R1
    m.h1("Reviewer 1")
    comment(m, "R1.1 (central claim; real-data result).",
            "“The paper's central claim … is not supported by its own strongest evidence. "
            "On the real Intel Berkeley Lab data … PIRCAD reaches macro-F1 0.712 against "
            "0.854 … and 0.845 … the robustness advantage does not transfer to the one "
            "real deployment tested … the real-data result shows the method "
            "underperforming a well-trained supervised baseline on the very attribution "
            "task it defines.”")
    m.p("We agree, and we have re-framed the paper around the claim the evidence does "
        "support. The paper's contribution is now stated as adaptation governance — "
        "the use of a cause verdict to decide what each adaptive element may learn — "
        "evaluated by its effect on a downstream safety function; the root-cause "
        "learner is described as the component that supplies the verdict, and we state "
        "in the abstract, contributions and conclusion that it is not the most accurate "
        "classifier, in simulation or on real data. The “bounded worst case under "
        "shift” has been removed from the list of the method's advantages (Section "
        "7.1) and is reported as a simulation result that did not reproduce on real "
        "data; a new experiment for R2.1 further shows that, even in simulation, its "
        "noise-shift part disappears when the baselines are trained at several noise "
        "levels. We also "
        "now test the real-data noise shift statistically (new Table "
        f"{R.get('t_intel_shift', '')}): PIRCAD's drop of "
        f"{DT['PIRCAD-rf']['drop'][0]:.3f} is significantly larger than the CNN's "
        f"{DT['Multi-scale 1D-CNN']['drop'][0]:.3f}, which is statistically detectable but "
        "negligible. Regarding the title: “safe self-calibration” refers to the "
        "governance result (Section 6.4), not to attribution accuracy; the new "
        "“Scope of the claims” paragraph states that this result is a simulation result "
        "conditional on an accurate reference. [[AUTHORS: if you prefer, the title can "
        "be narrowed further, e.g. “Ageing or Attack: Cause-Gated Self-Calibration Under "
        "Disguised Sensor Poisoning”.]]")
    m.p("**Changes:** Abstract; Highlights; Section 1 (contributions, new “Scope of "
        "the claims”); Section 6.6 (new Table with CIs and paired tests); Section 7.1; "
        "Section 8.", size=10)

    comment(m, "R1.2 (simulation, semi-synthetic data, bounded threat model, A5).",
            "“The evaluation is dominated by a simulated vessel … the real-data study is "
            "semi-synthetic … the A5 coordinated forgery is explicitly acknowledged to "
            "defeat the constraint attribution entirely … a competent attacker who can "
            "forge coupled instruments consistently is not defended against at all.”")
    m.p("We agree with each point and have made them explicit rather than leaving them "
        "to the limitations list. Section 1 now carries a “Scope of the claims” "
        "paragraph stating which results are simulated, which are semi-synthetic, and "
        "that behaviour under real unlabelled ageing is untested. Section 3.2 has a new "
        "paragraph, “What is not defended”, listing A5, actuator control and "
        "classifier-adaptive attackers. On A5 we ran the safety experiment with a "
        "coordinated attacker (new Table "
        f"{R.get('t_safety_variants', '')}), which sharpens the statement: A5 is "
        "detected and its forged values are forwarded while the attack lasts — the "
        "method does not protect the data — but because a process verdict forbids "
        "recalibration, A5 leaves no trace in the calibration, and the alarm fired in "
        f"{safe['PIRCAD']['A5 coordinated']['alarm_34'][0]*100:.0f} % of episodes afterwards "
        f"(crossing error {safe['PIRCAD']['A5 coordinated']['err_34'][0]:+.2f} °C). We "
        "describe A5 as “detected but not stopped”, and we do not claim the remedy (an "
        "independent or attested channel) as part of the system.")
    m.p("**Changes:** Section 1 (Scope of the claims); Section 3.2 (What is not "
        "defended); Section 6.4 (A4/A5 safety arms); Section 7.4.", size=10)

    comment(m, "R1.3 (propositions).",
            "“Proposition 1 … is correct but trivial … it means the core classification "
            "task is ill-posed for the A4 attacker … Proposition 2 … establishes safety "
            "relative to a reference whose error is unquantified outside simulation.”")
    m.p("We accept both points. Proposition 1 is now Observation 1, described as "
        "elementary. We explain why it is stated: it is precisely *because* the task is "
        "ill-posed for A4 that the evaluation must measure the consequence of the "
        "inevitable misattribution rather than attribution accuracy, and an evaluation "
        "reporting high accuracy against A4 would indicate a leak. The hierarchy is "
        "built around this boundary (A0–A3 separable to a degree we measure; A4 not). "
        "Proposition 2 is restated as a bound relative to the reference error, followed "
        "by a paragraph on what it does and does not establish: it does not establish "
        "safety where the reference error is unknown. We now quantify that error. On "
        "clean simulated data the leave-one-out reference error is "
        f"{ref['sim_abs_eref']['median']:.2f} °C (median; 95th percentile "
        f"{ref['sim_abs_eref']['p95']:.2f} °C); on the Intel motes the clean residual is "
        f"{ref['intel_abs_residual']['median']:.2f} °C (95th percentile "
        f"{ref['intel_abs_residual']['p95']:.2f} °C), with per-mote medians from "
        f"{ref['intel_per_mote_median']['min']:.2f} to {ref['intel_per_mote_median']['max']:.2f} °C "
        "(Sections 3.3 and 6.6). We also added Remark 1, on why a reference that uses "
        "the channel exposes less of the bias. We state it qualitatively and test it "
        "empirically rather than claim a bound, since its size depends on the "
        "estimator's gains.")
    m.p("**Changes:** Section 3.3 (Observation 1, Proposition 2, Remark 1, new "
        "paragraph on scope); Section 6.6 (reference error).", size=10)

    comment(m, "R1.4 (ablation undercuts the design rationale).",
            "“Replacing the dedicated leave-one-out observers … does not significantly "
            "change macro-F1 … dropping cross-sensor localisation leaves macro-F1 at "
            "0.937 … Removing the noise-signature feature group actually raises macro-F1 "
            "… the paper … does not resolve which configuration should be deployed.”")
    m.p("This was the most useful comment for us, because it pointed at an untested "
        "claim: the ablation measured labels, while the design rationale is about "
        "effect. We therefore ran the attack-then-fault safety experiment for each of "
        "these variants, using their retrained classifiers, under A0, A2, A4 and A5 "
        f"(new Table {R.get('t_safety_variants', '')}). {R['variant_verdict']} The "
        "ablation discussion (Section 6.7) now refers to these results instead of "
        f"asserting the rationale: {R['variant_summary']}")
    m.p("**Changes:** Section 6.4 (new variant table and discussion); Section 6.7.",
        size=10)

    comment(m, "R1.5 (governance results rest on one experiment).",
            "“… the paper does not report whether the baselines were tuned or configured "
            "for this scenario … the safety-alarm result is a single binary outcome "
            "(100 % vs 0 %) with no dose-response or sensitivity analysis.”")
    m.p("Section 5.3 now states exactly how the baselines were tuned (on training seeds "
        "disjoint from evaluation, to minimise their own drift-correction error, never "
        "shown the attack-then-fault scenario) and gives the selected values. Because "
        "that choice could decide the outcome, we swept each baseline's adaptation "
        "parameter over one to two orders of magnitude (Appendix Table A.1 and a new "
        "figure). EWMA and Page-Hinkley recalibration never fire the alarm at any "
        "setting. The Kalman bias estimator does at slow settings, but "
        + (f"only by leaving {kb['drift_fwd_error'][0]:.2f} °C of benign drift uncorrected "
           f"(PIRCAD {D9['PIRCAD']['drift_fwd_error'][0]:.2f} °C) and with spurious trips in "
           f"{kb['spurious_34'][0]*100:.0f} % of episodes. " if kb else ". ")
        + "So the outcome is not an artefact of tuning: a self-calibrating estimator "
        "without a cause vocabulary keeps the alarm only by giving up its job. We also "
        "added dose-response analyses (Appendix Table A.2): three attack rates, two "
        "attackers, three attack durations (1500, 3000, 4500 samples) and five alarm "
        "thresholds (32–36 °C), and two further attackers (A4, A5). PIRCAD fires the "
        "alarm in every A0/A2 condition; the per-condition results and the A4 "
        "over-correction are reported in Section 6.4.")
    m.p("**Changes:** Section 5.3; Section 6.4 (tuning sweep, dose-response, attackers); "
        "Appendix A.", size=10)

    comment(m, "R1.6 (the three defects; polarity; threshold 0.62).",
            "“… the paper does not report whether the current classifier still exhibits "
            "any sign-dependent behaviour … the confidence threshold 0.62 appears "
            "without justification … the evaluation's ability to detect analogous "
            "defects in the current system is not established.”")
    m.p("We added a per-polarity breakdown of the held-out bank (new Table "
        f"{R.get('t_pol', '')}). No system's macro-F1 differs between rising and falling "
        f"events (Mann–Whitney p ≥ {min(pol[s]['macro_f1']['p'] for s in pol):.2f}); for "
        f"PIRCAD it is {pol['PIRCAD-rf']['macro_f1']['rising'][0]:.3f} vs. "
        f"{pol['PIRCAD-rf']['macro_f1']['falling'][0]:.3f}. On the threshold: the "
        "repository history shows 0.62 was fixed in the original prototype before any "
        "held-out evaluation existed; the paper now says so and adds a sensitivity "
        f"analysis (new Section 6.5) from 0.40 to 0.90. In simulation 0.62 sits at the "
        "edge of a plateau (identical error rates from 0.62 to 0.90; below it the "
        "false-poison rate rises to "
        f"{th['sim']['0.40']['summary']['false_poison_rate'][0]:.3f}). On real data there "
        "is a monotone trade-off: raising the threshold to 0.90 lowers the false-poison "
        f"rate from {th['intel']['0.62']['summary']['false_poison_rate'][0]:.3f} to "
        f"{th['intel']['0.90']['summary']['false_poison_rate'][0]:.3f} but raises the "
        f"false-drift rate from {th['intel']['0.62']['summary']['false_drift_rate'][0]:.3f} "
        f"to {th['intel']['0.90']['summary']['false_drift_rate'][0]:.3f}; we describe the "
        "threshold as a deployment decision and explain its interaction with escalation "
        "and latched quarantine. On detecting analogous defects, we agree that a "
        "guarantee is impossible, and we say so (Section 7.3). We added a regression "
        "test for each defect that fails on the defective behaviour — we confirmed "
        "this by reintroducing the third defect, which the test catches — and the "
        "evaluation now reports effect-level metrics, the measure that exposed the "
        "second and third defects.")
    m.p("**Changes:** Sections 4.5, 6.1 (polarity table), 6.5 (new), 7.2, 7.3; "
        "tests/test_v2.py.", size=10)

    comment(m, "R1.7 (abstract and highlights).",
            "“The abstract … selectively emphasises parity with the CNN while omitting "
            "that the forest beats PIRCAD … the highlight ‘PIRCAD achieves 96.2 % "
            "root-cause accuracy’ reports the gradient-boosting macro-F1 … The paper does "
            "not explain why the random forest is the default.”")
    m.p("Both are corrected. The abstract now reports only the default configuration "
        f"(macro-F1 {R['mf1']['PIRCAD-rf']:.3f}) and states that the generic-feature "
        f"forest ({R['mf1']['Generic-feature RF']:.3f}) is better in distribution and that "
        "both supervised baselines are better on real data. The 96.2 % highlight has "
        "been removed; it mislabelled a macro-F1 as accuracy and reported a non-default "
        "back-end. The new highlights contain no accuracy figure. A new paragraph "
        "(Section 4.4, “Choice of default back-end”) explains the default: back-ends are "
        "chosen by the dangerous errors, and gradient boosting's false-poison rate "
        f"({R['fp_hgb']:.3f} against {R['fp_rf']:.3f}) means removing serviceable sensors "
        "under latched quarantine; it is also worse on real data "
        f"({S10['PIRCAD-hgb']['macro_f1'][0]:.3f} vs. {S10['PIRCAD-rf']['macro_f1'][0]:.3f}). "
        "We state that this choice is made on the error criterion, and gradient boosting "
        "remains reported alongside.")
    m.p("**Changes:** Abstract; Highlights; Sections 4.4 and 6.1.", size=10)

    p = c16["pircad"]
    comment(m, "R1.8 (computational and deployment claims).",
            "“Latency is reported … on an Apple-silicon laptop, which is not a fog-node "
            "or edge measurement; the ESP32 result is explicitly simulation-only … the "
            "edge results are not positioned against alternatives … no absolute "
            "throughput or memory figures for the fog tier.”")
    m.p("We agree and have rewritten Section 6.8. It now opens by stating that all "
        "timings are laptop measurements, to be read as relative costs and an upper "
        "bound on throughput. It reports absolute figures — median "
        f"{p['ms_median']:.2f} ms, 95th percentile {p['ms_p95']:.2f} ms and 99th "
        f"{p['ms_p99']:.2f} ms per frame, {p['frames_per_s']:.0f} frames/s sustained, a "
        f"{c16['forest']['bytes'] / 2**20:.1f} MB forest ({c16['forest']['trees']} trees, "
        f"{c16['forest']['nodes']:,} nodes) — and the per-sample cost of the "
        "self-calibrating baselines on the same machine, against which PIRCAD's median "
        "frame is roughly "
        f"{p['ms_median'] * 1e3 / c16['baselines_single_channel']['Kalman bias estimator']['us_per_sample']:.0f} to "
        f"{p['ms_median'] * 1e3 / c16['baselines_single_channel']['EWMA recalibration']['us_per_sample']:.0f} times more expensive. The ESP32 kernel is now described as "
        "a feasibility demonstration in simulation, compared with one baseline only, "
        "and explicitly not counted among the contributions. We do not have fog or "
        "edge hardware results and say so.")
    m.p("**Changes:** Section 6.8 (rewritten); Section 7.4.", size=10)

    comment(m, "R1.9 (related-work positioning).",
            "“The table marks PIRCAD as addressing … but … without a citation-level "
            "argument … the paper does not discuss work on physics-based fault diagnosis "
            "with cause-specific responses … drift-aware IoT detectors [10] … the "
            "distinction … is asserted rather than argued against specific mechanisms.”")
    m.p("We have rewritten the relevant paragraphs of Section 2 to argue against "
        "mechanisms. For the drift-aware detector of Xu et al., we describe its "
        "mechanism (detect drift, interpret which features moved, update the model) and "
        "explain why it conditions adaptation on *whether* the distribution changed, not "
        "why, so a gradual injection supplies exactly the evidence that triggers the "
        "update. For observer-based diagnosis we now acknowledge that it *does* pair "
        "diagnosis with a cause-specific response (fault accommodation), and state the "
        "two differences precisely: its hypotheses are components, so ageing and an "
        "ageing-shaped attack on the same sensor are one hypothesis, and accommodation "
        "acts on the control law rather than gating the learning elements of a data "
        "pipeline. We added this literature as a row in Table 1, revised the PIRCAD "
        "row's real-data entry to “~ (semi-synthetic)”, and state that the table "
        "reflects our reading of each paper and that the “not combined” claim is limited "
        "to the works examined, not the result of a systematic review.")
    m.p("**Changes:** Section 2; Table 1; Section 7.4.", size=10)

    comment(m, "R1.10 (statistical reporting).",
            "“… the attacker-capability sweep (Table 6) reports … without confidence "
            "intervals or significance tests … Table 7 … no CIs and no tests, and the "
            "process-model-mismatch rows … is not a comparison at all … Table 10 reports "
            "only point estimates … rank-biserial values are not reported in any table.”")
    m.p("All addressed. The attacker sweep is now two tables with 95 % CIs for every "
        "system and Holm-corrected paired Wilcoxon tests against PIRCAD; this showed "
        "that in the A0–A2 regime most differences are not significant, which we now "
        "state. The distribution-shift table has CIs, paired tests against both "
        "baselines and effect sizes; the process-model-mismatch rows have moved to "
        "their own table, labelled as a sensitivity analysis of PIRCAD and explicitly "
        "not a comparison. The Intel noise-shift table has CIs, and the drops are "
        "tested: PIRCAD's drop is significantly larger than the CNN's "
        f"(p = {DT['PIRCAD-rf vs Multi-scale 1D-CNN']['p_holm']:.1e}, r = "
        f"{DT['PIRCAD-rf vs Multi-scale 1D-CNN']['r_rb']:+.2f}); the CNN's own drop is "
        f"significant but negligible ({DT['Multi-scale 1D-CNN']['drop'][0]:.4f}). "
        "Rank-biserial effect sizes are now reported in every comparison table (the "
        "in-distribution benchmark, distribution shift, Intel Lab and ablation tables), "
        "and Section 5.4 states the sign convention.")
    m.p("**Changes:** Section 5.4; Section 6 tables throughout; experiments/exp15_posthoc.py.",
        size=10)

    comment(m, "R1.11 (handling of negative results).",
            "“… the conclusion (Section 8) still proposes ‘combining the governance … "
            "with learned expectation models’ … without addressing that the one "
            "combined-feature experiment reported degraded robustness … the abstract's "
            "framing … still leads with PIRCAD's macro-F1 figures.”")
    m.p("We agree that this was inconsistent. The conclusion now distinguishes the "
        "combination we tested — pooling physics and generic features in one learner, "
        "which inherits the fragility of the generic features — from the one we "
        "propose, which is to learn better *references* that remain independent of the "
        "channel being judged; it states the negative result explicitly. The abstract "
        "now leads with the governance result and reports the classifier comparison, "
        "including the real-data deficit, as such.")
    m.p("**Changes:** Abstract; Sections 6.7, 7.1, 7.2 and 8.", size=10)

    comment(m, "R1.12 (reproducibility, AI use and authorship).",
            "“The division of labour between the human author and the AI tool in "
            "experiment execution and analysis is not specified … the absence of any "
            "independent replication or external audit … the single-author dominance in "
            "a four-author paper … is not explained.”")
    m.p("We have expanded the generative-AI declaration to state which parts of the "
        "work the tool was used for and what the authors decided and checked. "
        "[[AUTHORS: confirm the declaration text in the manuscript, and replace this "
        "sentence with a short factual account of each co-author's role, e.g. the edge "
        "node, reviewing and re-running experiments, checking statistics, "
        "supervision.]] On replication, we cannot claim an external audit. We have "
        "taken the steps available to us: every number is generated by released "
        "scripts from released per-episode artifacts; for this revision the stored "
        "results were reproduced exactly in a fresh software environment (Python 3.13, "
        "scikit-learn 1.9: all 900 values of the 180-episode benchmark matched, "
        "experiments/repro_check.py); and each defect of Section 7.3 now has a regression test. "
        "The lack of independent replication is listed as a limitation (Section 7.4).")
    m.p("**Changes:** Declarations; Section 7.4; README of the repository.", size=10)

    # =================================================================== R2
    m.h1("Reviewer 2")
    m.p("We thank the reviewer for the positive assessment and address the three "
        "points in turn.")
    x3 = e17["noise x3"]
    comment(m, "R2.1 (mitigating sensitivity to noise shift and model error).",
            "“… the discussion should be expanded regarding how to mitigate the "
            "framework's sensitivity to sensor-noise shifts and process-model "
            "inaccuracies …”")
    m.p("We added Section 7.2, “Mitigating sensitivity to sensor noise and model "
        "error”, and tested one mitigation. Training the root-cause learner on the same "
        "episodes at noise ×1, ×1.5 and ×2 changes macro-F1 under a ×3 noise shift, a "
        f"level outside the augmented range, from {x3['PIRCAD']['macro_f1'][0]:.3f} to "
        f"{x3['PIRCAD noise-augmented']['macro_f1'][0]:.3f}, and the false-drift rate from "
        f"{x3['PIRCAD']['false_drift_rate'][0]:.3f} to {x3['PIRCAD noise-augmented']['false_drift_rate'][0]:.3f}. "
        "For fairness we applied the same augmentation to the generic-feature forest, "
        f"which then reaches {x3['Generic-feature RF noise-augmented']['macro_f1'][0]:.3f} at ×3, "
        "better than PIRCAD; we report this in Sections 6.3, 7.1 and 7.2, since it "
        "removes the noise-shift part of the robustness advantage we had claimed. For "
        "process-model error, we propose — and identify as untested — measuring the "
        "reference error at commissioning to cap or disable correction, governed "
        "online re-estimation of model parameters (learning only from trusted "
        "channels), and learned expectation models that remain leave-one-out; and we "
        "describe the confidence threshold as a site-specific deployment decision, "
        "supported by the new sensitivity analysis (Section 6.5).")
    m.p("**Changes:** Section 7.2 (new); Section 6.5 (new); experiments/exp17.", size=10)

    comment(m, "R2.2 (necessity of components that do not change accuracy).",
            "“… clarifying their practical necessity in bounding downstream error and "
            "protecting safety alarms would better justify the overall system "
            "complexity.”")
    m.p("This is the same question as R1.4, and we answered it with the same new "
        f"experiment: the safety test run for each ablated variant. {R['variant_verdict']} "
        "For the two governance elements, the existing ablation already showed "
        "that removing them leaves macro-F1 unchanged but drops retraining protection "
        "to zero or containment to zero, and the non-differential control in Section "
        "6.4 shows the consequence for the safety alarm.")
    m.p("**Changes:** Section 6.4; Section 6.7.", size=10)

    comment(m, "R2.3 (editorial).",
            "“… consistent stage numbering in the system architecture diagram, align the "
            "performance metric terminology between the highlights and the main text, "
            "and verify that all foundational datasets and statistical methods in the "
            "reference list accurately match their in-text citations.”")
    m.p("All three are corrected. (i) Fig. 1 omitted the confidence gate E6, so the "
        "numbering jumped from E5 to E8; it has been redrawn with all eight stages in "
        "processing order, and the caption and Section 4.5 now name E6. (ii) The "
        "highlight that reported a macro-F1 as “root-cause accuracy” has been removed; "
        "the highlights now contain no figure that the text does not use under the same "
        "name. (iii) We are grateful for this check: in the submitted list, seventeen "
        "entries had been updated to more recent works while the in-text numbers were "
        "left unchanged, so that, for example, the Intel Berkeley Lab "
        "dataset, the Wilcoxon and Holm procedures, the random forest, the Kalman "
        "filter and the Magnus formula were cited as unrelated papers. The list is now "
        "generated from the citations, in order of first appearance: each foundational "
        "source (the dataset, Wilcoxon, Holm, Breiman, Kalman, Alduchov–Eskridge, Liu "
        "et al., Page, and the other original references) is restored where the text "
        "relies on it, and the recent works are kept and cited where their content "
        "supports the text.")
    m.p("**Changes:** Fig. 1; Section 4.5; Highlights; References.", size=10)

    out = HERE / "Response_to_Reviewers_R1.docx"
    m.save(out)
    print("wrote", out)


if __name__ == "__main__":
    main()
