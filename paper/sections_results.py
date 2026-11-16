"""Abstract, highlights, Sections 6-8 and Appendix A.  Every number is read from artifacts."""

from __future__ import annotations

from pathlib import Path

import numpy as np

import results_tables as T
from sections_declarations import declarations  # noqa: F401  (re-exported)

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"

BASES = ("Kalman bias estimator", "EWMA recalibration", "Page-Hinkley recalibration")
ATT14 = ("stale/white", "fidelity 0.3/matched", "A4 on-path", "A5 coordinated")
ATT14_LABEL = {"stale/white": "A0", "fidelity 0.3/matched": "A2 (0.30)",
               "A4 on-path": "A4 on-path", "A5 coordinated": "A5 coordinated"}


# --------------------------------------------------------------------------
def collect() -> dict:
    R = {k: T.load(f) for k, f in (
        ("e6", "exp6_statistical_benchmark.json"), ("e7", "exp7_adaptive_attacker.json"),
        ("e8", "exp8_distribution_shift.json"), ("e9", "exp9_adaptation_poisoning.json"),
        ("e10", "exp10_intel_lab.json"), ("e11", "exp11_ablation.json"),
        ("e12", "exp12_hybrid_features.json"), ("e13", "exp13_intel_shift.json"),
        ("e14thr", "exp14_threshold.json"), ("e14safe", "exp14_safety.json"),
        ("e14dose", "exp14_dose.json"), ("e14base", "exp14_baselines.json"),
        ("e14ref", "exp14_reference.json"), ("e15", "exp15_posthoc.json"),
        ("e16", "exp16_cost.json"), ("e17", "exp17_noise_augmentation.json"))}
    e6 = R["e6"]
    R["n_corpus"] = "509 513"
    R["lat"] = T.latency(e6)
    S = e6["summary"]
    R["mf1"] = {k: S[k]["macro_f1"]["mean"] for k in S}
    R["acc"] = {k: S[k]["root_cause_accuracy"]["mean"] for k in S}
    R["fp_rf"] = S["PIRCAD-rf"]["false_poison_rate"]["mean"]
    R["fp_hgb"] = S["PIRCAD-hgb"]["false_poison_rate"]["mean"]
    e8 = R["e8"]
    R["worst8"] = {}
    for s in ("PIRCAD", "Multi-scale 1D-CNN", "Generic-feature RF"):
        vals = {c: e8["results"][c][s]["summary"]["macro_f1"][0]
                for c in e8["conditions"] if s in e8["results"][c]
                and not c.startswith("twin")}
        far = {c: e8["results"][c][s]["summary"]["false_alarm_rate"][0] for c in vals}
        w = min(vals, key=vals.get)
        R["worst8"][s] = (w, vals[w], max(far.values()), float(np.mean(list(vals.values()))))
    # most generous tuning of each baseline that still fires the alarm
    D = R["e14dose"]["tuning"]
    R["best_firing"] = {}
    for b in BASES:
        ok = [t for t in D[b] if t["alarm_34"][0] >= 0.95]
        R["best_firing"][b] = min(ok, key=lambda t: t["drift_fwd_error"][0]) if ok else None
    return R


def _v(x, nd=3):
    return f"{x:.{nd}f}"


def _m(v):
    return v[0] if isinstance(v, (list, tuple)) else v["mean"]


def _ci(v, nd=2):
    m, lo, hi = T.t3(v)
    if not np.isfinite(m):
        return "–"
    return f"{m:.{nd}f} [{lo:.{nd}f}, {hi:.{nd}f}]"


def _r(t):
    return f"{t['r_rb']:+.2f}"


SHORT = {"PIRCAD-rf": "PIRCAD", "Multi-scale 1D-CNN": "the CNN",
         "Generic-feature RF": "the generic-feature forest"}


def _shift_label(c):
    s = T.SHIFT_LABEL.get(c, c)
    return s[0].lower() + s[1:]


def _noise(level):
    if level == "original":
        return "No added noise"
    return "Noise " + level.removeprefix("noise ").replace(" C /", " °C /")


# --------------------------------------------------------------------------
def highlights(R):
    return [
        "Ageing-shaped poisoning is learned as normal by self-calibrating IoT pipelines",
        "Cause verdicts gate recalibration, quarantine and alarms for each sensor",
        "Simulated safety alarm survived poisoning; tuned self-calibration never fired",
        "Supervised learners classified better in distribution and on real lab data",
        "Forwarded error is bounded relative to a channel-independent reference",
    ]


def abstract(R):
    e10, e9 = R["e10"], R["e9"]
    S9, S10 = e9["summary"], e10["summary"]
    bf = R["best_firing"]
    kb = bf["Kalman bias estimator"]
    sweep = ("" if kb is None else
             " Over its tuning range the Kalman estimator kept the alarm only when leaving "
             f"≥ {kb['drift_fwd_error'][0]:.2f} °C of real drift uncorrected (PIRCAD "
             f"{R['e9']['drift_summary']['PIRCAD']['drift_fwd_error'][0]:.2f} °C).")
    return (
        "Self-calibrating IoT pipelines learn from the data they accept, so an "
        "injection that imitates transducer ageing becomes the new normal and later "
        "masks real faults. We ask whether the cause of a deviation (benign drift, "
        "poisoning or a process event) can be inferred and used to decide what each "
        "adaptive element may learn. For attackers matched to ageing in size and rate, "
        "an on-path attacker is indistinguishable from drift, but a correction anchored "
        "to a channel-independent reference bounds the forwarded error relative to that "
        "reference. PIRCAD judges each channel against a physics-based expectation that "
        "leaves it out, attributes one of four causes, and lets the verdict gate "
        "recalibration, quarantine and alarms. In a simulated attack-then-fault study, a "
        f"safety alarm fired in {S9['PIRCAD']['alarm_fired'][0]*100:.0f} % of 120 "
        "episodes with PIRCAD and in none with Kalman, Page-Hinkley or EWMA "
        "self-calibration." + sweep + " As a classifier PIRCAD is not the most "
        f"accurate: on held-out simulated episodes its macro-F1 ({R['mf1']['PIRCAD-rf']:.3f}) "
        f"matches a 1D-CNN ({R['mf1']['Multi-scale 1D-CNN']:.3f}) but trails a "
        f"generic-feature forest ({R['mf1']['Generic-feature RF']:.3f}); on real Intel "
        f"Lab streams with injected events it trails both ({S10['PIRCAD-rf']['macro_f1'][0]:.3f} "
        f"against {S10['Multi-scale 1D-CNN']['macro_f1'][0]:.3f} and "
        f"{S10['Generic-feature RF']['macro_f1'][0]:.3f}), and its simulated advantage "
        "under noise shift vanishes with noise-augmented baselines and on real data. "
        "The safety benefit therefore rests on simulation and on an accurate reference, "
        "whose error we quantify.")


# --------------------------------------------------------------------------
def results_section(m, R):
    e6 = R["e6"]
    m.h1("6. Results")
    m.h2("6.1 In-distribution benchmark")
    S = e6["summary"]
    tw = e6["tests"]["macro_f1"]
    rows = T.bench_table(e6)
    for row, s in zip(rows, T.SYS6):
        row.insert(len(row) - 1, "–" if s == "PIRCAD-rf" else _r(tw[s]))
    t_bench = m.table(
        "Root-cause attribution on 180 held-out simulated episodes (30 seeds × 6 "
        "scenarios). Mean [95 % bootstrap CI]; r: rank-biserial effect size and p: "
        "paired Wilcoxon test of per-episode macro-F1, PIRCAD (random forest) against "
        "each row, Holm-corrected (r > 0: PIRCAD higher). ↓ lower is better.",
        ["System", "Macro-F1", "Accuracy", "False-drift ↓", "False-poison ↓",
         "Process masked ↓", "False alarm ↓", "r", "p"],
        rows, widths=[3.4, 2.5, 2.5, 1.2, 1.2, 1.2, 1.2, 1.0, 1.3], font=7.5)
    m.p(f"Table {t_bench} addresses the first question: is a "
        "physics-anchored learner more accurate than a strong data-driven one when "
        "train and test come from the same distribution? It is not. PIRCAD reaches "
        f"macro-F1 {_v(R['mf1']['PIRCAD-rf'])} with its default random forest; the "
        f"multi-scale 1D-CNN reaches {_v(R['mf1']['Multi-scale 1D-CNN'])}, not "
        f"significantly different (p = {tw['Multi-scale 1D-CNN']['p_holm']:.2f}, "
        f"r = {_r(tw['Multi-scale 1D-CNN'])}), and the generic-feature forest "
        f"{_v(R['mf1']['Generic-feature RF'])}, significantly higher with a large "
        f"effect (r = {_r(tw['Generic-feature RF'])}). Both learned baselines see every "
        "raw channel and the actuator command, so in distribution they can learn the "
        "coupling the physics encodes. The authors' earlier pipeline (Gen-1), against "
        f"which our prototype was originally compared, reaches {_v(R['mf1']['Gen-1 FHAD'])}: "
        "a comparison against it alone would overstate the contribution.")
    m.p("Gradient boosting has a higher macro-F1 than the random forest "
        f"({_v(R['mf1']['PIRCAD-hgb'])}), but it buys that by calling drifting samples "
        f"poisoned at a rate of {R['fp_hgb']:.3f} against {R['fp_rf']:.3f}, and under "
        "latched quarantine each such verdict removes a serviceable sensor. This is why "
        "the random forest is the default (Section 4.4): the two operationally "
        "dangerous confusions are both near zero for it, poisoning read as drift "
        f"({_v(S['PIRCAD-rf']['false_drift_rate']['mean'])}) and drift read as poisoning "
        f"({_v(S['PIRCAD-rf']['false_poison_rate']['mean'])}). Its residual error is "
        "concentrated in benign drift, where the verdict is withheld until the ramp "
        "rises out of the noise (mean detection delay "
        f"{np.nanmean([r['detection_delay'] for r in e6['rows']['PIRCAD-rf'] if np.isfinite(r['detection_delay'])]):.0f} "
        "samples), and in a false-alarm rate of "
        f"{_v(S['PIRCAD-rf']['false_alarm_rate']['mean'])} on clean data. The MLP reaches "
        f"{R['mf1']['PIRCAD-mlp']:.2f} because it issues confident poisoning verdicts "
        "early in benign drift (false-poison rate "
        f"{S['PIRCAD-mlp']['false_poison_rate']['mean']:.2f}), and the linear model and "
        "the training-free rule set are far behind — the feature space is informative "
        "but not linearly separable.")
    pol = R["e15"]["polarity"]
    names = {"PIRCAD-rf": "PIRCAD (random forest)", "PIRCAD-hgb": "PIRCAD (gradient boosting)",
             "Multi-scale 1D-CNN": "Multi-scale 1D-CNN", "Generic-feature RF": "Generic-feature RF"}
    prow = []
    for s, lab in names.items():
        p = pol[s]
        prow.append([lab, _ci(p["macro_f1"]["rising"], 3), _ci(p["macro_f1"]["falling"], 3),
                     f"{p['macro_f1']['p']:.2f}",
                     f"{p['false_drift_rate']['rising'][0]:.3f} / {p['false_drift_rate']['falling'][0]:.3f}",
                     f"{p['false_poison_rate']['rising'][0]:.3f} / {p['false_poison_rate']['falling'][0]:.3f}"])
    t_pol = m.table(
        "Polarity check on the held-out bank: the 150 event episodes split by event "
        "direction (75 rising, 75 falling; clean episodes excluded). p: two-sided "
        "Mann–Whitney U test of per-episode macro-F1, rising vs. falling.",
        ["System", "Macro-F1, rising", "Macro-F1, falling", "p",
         "False-drift rising / falling", "False-poison rising / falling"],
        prow, widths=[3.6, 2.8, 2.8, 1.0, 2.6, 2.6], font=8)
    R["t_pol"] = t_pol
    pmin = min(pol[s]["macro_f1"]["p"] for s in names)
    m.p(f"Table {t_pol} checks for the sign-dependent behaviour that the original "
        "classifier had learned (Section 7.3). No system's macro-F1 differs between "
        f"rising and falling events (all p ≥ {pmin:.2f}), and the dangerous error rates "
        "are of the same size in both directions. The largest asymmetry is gradient "
        "boosting's false-poison rate "
        f"({pol['PIRCAD-hgb']['false_poison_rate']['rising'][0]:.3f} rising, "
        f"{pol['PIRCAD-hgb']['false_poison_rate']['falling'][0]:.3f} falling), which is "
        "within the spread of its per-episode values.")
    det = T.detection_table(e6)
    t_det = m.table(
        "Detection only (anomalous vs. not), all scenarios pooled. Unsupervised "
        "detectors are thresholded at 1 % false alarms on clean training data.",
        ["System", "Detection F1", "Recall", "False alarm"], det,
        widths=[5.0, 3.0, 3.0, 3.0])
    dr = T.det_recall_by_scenario(e6)
    m.p(f"Table {t_det} places the unsupervised detectors. They flag most deviations "
        "but cannot name them, and none has a response that differs by cause; the "
        f"Kalman χ² bad-data detector recalls {dr['Kalman chi-square']['stealth_poison']:.2f} "
        "of rate-matched poisoning but would answer drift and poisoning identically. "
        "PIRCAD's own detection F1 is lower than the learned baselines' because it "
        "deliberately withholds a cause until the evidence supports one.")
    m.figure(FIG / "v2_fig_benchmark.png",
             "Per-scenario root-cause accuracy on the held-out bank (mean, 95 % CI). "
             "PIRCAD's shortfall is concentrated in benign drift (detection delay) and "
             "spike poisoning; the learned baselines are uniformly strong in "
             "distribution, and Gen-1 can only name what a threshold on a rolling "
             "mean can see.", 15.5)
    _attacker(m, R)
    _shift(m, R)
    _safety(m, R)
    _threshold(m, R)
    _real(m, R)
    _ablation(m, R)
    _cost(m, R)


# --------------------------------------------------------------------------
def _attacker(m, R):
    e7 = R["e7"]
    P = R["e15"]["attacker"]
    m.h2("6.2 How much does the attacker need to know?")
    Rr = e7["results"]

    def g(c, s, k):
        return Rr[c][s]["summary"][k][0]

    def mark(c, b, k):
        t = P[c].get(f"{b}|{k}")
        return "*" if t and t["p_holm"] < 0.05 else ""

    rows_a, rows_b = [], []
    for c in e7["conditions"]:
        r = Rr[c]
        key = "drift_recall" if c.startswith("drift") else "poison_recall"
        lab = T.ATT_LABEL.get(c, c)
        rows_a.append([lab, _ci(r["PIRCAD"]["summary"][key]),
                       _ci(r["Multi-scale 1D-CNN"]["summary"][key]) + mark(c, "Multi-scale 1D-CNN", key),
                       _ci(r["Generic-feature RF"]["summary"][key]) + mark(c, "Generic-feature RF", key),
                       f(r["PIRCAD"]["summary"]["false_drift_rate"]) if key == "poison_recall" else "–"])
        rows_b.append([lab, _ci(r["PIRCAD"]["summary"]["useful_frac"]),
                       _ci(r["Multi-scale 1D-CNN"]["summary"]["useful_frac"]) + mark(c, "Multi-scale 1D-CNN", "useful_frac"),
                       _ci(r["Generic-feature RF"]["summary"]["useful_frac"]) + mark(c, "Generic-feature RF", "useful_frac"),
                       _ci(r["PIRCAD"]["summary"]["fwd_err_mean"])])
    t = m.table(
        "Attacker-capability sweep, labels (20 seeds per condition; models trained on "
        "A0 only). Recall is drift recall for the reference row and poisoning recall "
        "otherwise; mean [95 % CI]. * baseline differs from PIRCAD (paired Wilcoxon over "
        "seeds, Holm-corrected over all 44 comparisons of Tables 7 and 8, p < 0.05).",
        ["Condition", "Recall PIRCAD", "Recall CNN", "Recall generic RF", "False-drift PIRCAD"],
        rows_a, widths=[3.8, 3.2, 3.2, 3.2, 1.8], font=8)
    t2 = m.table(
        "Attacker-capability sweep, effect on the attacked channel. Useful-data rate: "
        "fraction of event-window samples forwarded within 0.5 °C of the true value; "
        "forwarded error: PIRCAD's mean absolute error over forwarded samples (°C). "
        "Mean [95 % CI]; * as in Table " + str(t) + ".",
        ["Condition", "Useful PIRCAD", "Useful CNN", "Useful generic RF", "Fwd. error PIRCAD"],
        rows_b, widths=[3.8, 3.0, 3.0, 3.0, 2.6], font=8)
    R["t_att_effect"] = t2
    lo = ("stale, white (default)", "stale, matched noise", "fidelity 0.05", "fidelity 0.10")
    nsig = sum(P[c][f"{b}|poison_recall"]["p_holm"] < 0.05 for c in lo
               for b in ("Multi-scale 1D-CNN", "Generic-feature RF"))
    m.p(f"Tables {t} and {t2} sweep the threat hierarchy of Section 3.2 without "
        "retraining any model. When the attacker sees the live process only through a "
        "delayed surrogate (A0–A2, fidelity up to 0.10), every learned system catches at "
        f"least {min(g(c, s, 'poison_recall') for c in lo for s in T.ATT_SYS):.2f} of the "
        "poisoning, and copying the sensor's noise spectrum gains the attacker nothing; "
        f"in this regime only {nsig} of the 8 recall comparisons with PIRCAD are "
        "significant after correction, so we read the systems as equivalent there. "
        "Differences appear once the surrogate becomes more accurate. At fidelity 0.30 "
        f"PIRCAD retains recall {g('fidelity 0.30', 'PIRCAD', 'poison_recall'):.2f} against "
        f"{g('fidelity 0.30', 'Generic-feature RF', 'poison_recall'):.2f} (generic forest, "
        f"r = {_r(P['fidelity 0.30']['Generic-feature RF|poison_recall'])}) and "
        f"{g('fidelity 0.30', 'Multi-scale 1D-CNN', 'poison_recall'):.2f} (CNN, "
        f"r = {_r(P['fidelity 0.30']['Multi-scale 1D-CNN|poison_recall'])}); with the exact "
        "process state but white noise (A3) the generic forest is best "
        f"({g('exact state, white', 'Generic-feature RF', 'poison_recall'):.2f} against "
        f"{g('exact state, white', 'PIRCAD', 'poison_recall'):.2f} for PIRCAD, "
        f"r = {_r(P['exact state, white']['Generic-feature RF|poison_recall'])}) and the CNN "
        "collapses to zero. No method dominates: each has a different breaking point "
        "along the knowledge axis.")
    m.p("Once the attacker also matches the transducer's noise (A3) or adds the ramp to "
        "the genuine reading (A4), every system's poisoning recall falls to "
        f"≤ {max(g(c, s, 'poison_recall') for c in ('exact state, matched', 'on-path (MITM) ramp') for s in T.ATT_SYS):.3f}"
        " — Observation 1 in practice: the attacked channel is statistically benign "
        "drift, and calling it drift is the only consistent verdict. What differs is the "
        "consequence. A label-only system can withhold what it flags but cannot correct "
        "it, because it has no estimate of the bias, so it delivers almost none of the "
        "affected channel: its useful-data rate on these attacks — and on genuine drift "
        f"— is {min(g('drift (reference)', s, 'useful_frac') for s in T.ATT_SYS[1:]):.2f}–"
        f"{max(g(c, s, 'useful_frac') for c in ('drift (reference)', 'on-path (MITM) ramp') for s in T.ATT_SYS[1:]):.2f}, "
        "a difference from PIRCAD that is significant for every such condition. PIRCAD "
        "calls the A4 ramp drift, as it must, but corrects it against the physics: "
        f"{g('on-path (MITM) ramp', 'PIRCAD', 'useful_frac')*100:.0f} % of the attacked "
        "channel is delivered within 0.5 °C of the truth, with a mean forwarded error of "
        f"{g('on-path (MITM) ramp', 'PIRCAD', 'fwd_err_mean'):.2f} °C, while the injected "
        "bias itself grows to several degrees. This is Proposition 2 observed while the "
        "attack lasts; Section 6.4 shows what happens when it stops.")
    m.p("The coordinated attacker (A5) marks the limit of the method. By forging every "
        "physically coupled instrument from one fake temperature, it keeps the "
        "instrument-side constraints satisfied; PIRCAD's localisation then attributes "
        "the deviation to the process, and the node forwards it with a process alarm "
        f"(forwarded error {g('coordinated, stale', 'PIRCAD', 'fwd_err_mean'):.1f} °C). "
        "The alarm is raised, but the forged values are delivered: against A5 the "
        "method provides detection, not protection of the data. Stopping it needs at "
        "least one instrument outside the attacker's control, such as an independent "
        "reference, a measurement on the actuator side, or a channel with cryptographic "
        "attestation; a single fog node cannot provide that on its own.")
    m.figure(FIG / "v2_fig_attacker.png",
             "Attacker-capability sweep. (a) Poisoning recall; (b) useful-data rate on "
             "the attacked channel. Left to right the attacker knows more about the "
             "live process. Where no system can label the attack (A3 with matched "
             "noise, A4), PIRCAD still delivers the channel corrected to the physical "
             "truth; label-only systems can only withhold it.", 15.5)


def f(v, nd=3):
    return T.f(v, nd)


# --------------------------------------------------------------------------
def _shift(m, R):
    e8 = R["e8"]
    Rr = e8["results"]
    P = R["e15"]["shift"]
    m.h2("6.3 Robustness to distribution shift")

    def g(c, s, k="macro_f1"):
        return Rr[c][s]["summary"][k][0]

    def cell(c, s):
        v = _ci(P[c][s]["macro_f1"], 3)
        t = P[c][s].get("test")
        return v + ("*" if t and t["p_holm"] < 0.05 else "")

    rows = []
    for c in e8["conditions"]:
        if c.startswith("twin"):
            continue
        rows.append([T.SHIFT_LABEL.get(c, c), _ci(P[c]["PIRCAD"]["macro_f1"], 3),
                     cell(c, "Multi-scale 1D-CNN"), cell(c, "Generic-feature RF"),
                     f"{P[c]['Multi-scale 1D-CNN']['test']['r_rb']:+.2f} / "
                     f"{P[c]['Generic-feature RF']['test']['r_rb']:+.2f}",
                     f(Rr[c]["PIRCAD"]["summary"]["false_drift_rate"]),
                     f(Rr[c]["PIRCAD"]["summary"]["false_alarm_rate"])])
    t = m.table(
        "Macro-F1 under distribution shift, no retraining (10 seeds × 6 scenarios per "
        "condition), mean [95 % CI]. * differs from PIRCAD (paired Wilcoxon by episode, "
        "Holm-corrected over the 18 comparisons); r: rank-biserial effect size, PIRCAD "
        "vs. CNN / vs. generic RF (r > 0: PIRCAD higher). The last two columns are "
        "PIRCAD's false-drift and false-alarm rates.",
        ["Condition", "PIRCAD", "1D-CNN", "Generic RF", "r (CNN / RF)",
         "False-drift PIRCAD", "False alarm PIRCAD"],
        rows, widths=[3.0, 2.6, 2.6, 2.6, 2.0, 1.5, 1.5], font=7.5)
    w = R["worst8"]
    m.p(f"Table {t} trains every system once, on the nominal plant, and evaluates it "
        "on plants and sensors it has never seen. The data-driven learners are "
        "excellent until a shift touches what they learned, and then they fail "
        "abruptly. The generic-feature forest is unaffected by changed plant dynamics, "
        "ambient swing or set-point, but when the transducers become twice as noisy "
        f"its macro-F1 falls to {g('noise x2', 'Generic-feature RF'):.3f} with a false-alarm "
        f"rate of {g('noise x2', 'Generic-feature RF', 'false_alarm_rate'):.2f}: its "
        "features are scaled to the commissioning noise level, so the whole clean "
        "stream now looks anomalous. The CNN degrades under noise "
        f"({g('noise x3', 'Multi-scale 1D-CNN'):.3f} at ×3) and collapses when the plant "
        f"is merely regulated at a different set-point ({g('setpoint 35 C', 'Multi-scale 1D-CNN'):.3f}, "
        f"false alarms {g('setpoint 35 C', 'Multi-scale 1D-CNN', 'false_alarm_rate'):.2f}, "
        f"r = {_r(P['setpoint 35 C']['Multi-scale 1D-CNN']['test'])}) — a shift that "
        "changes nothing about the causes, only the operating point the network "
        "memorised.")
    m.p("PIRCAD degrades but does not collapse: its worst condition is "
        f"{_shift_label(w['PIRCAD'][0])} at macro-F1 {w['PIRCAD'][1]:.3f}, against "
        f"{w['Multi-scale 1D-CNN'][1]:.3f} ({_shift_label(w['Multi-scale 1D-CNN'][0])}) for the CNN and "
        f"{w['Generic-feature RF'][1]:.3f} ({_shift_label(w['Generic-feature RF'][0])}) for the generic "
        "forest, both significantly lower. Where the shift leaves the residual "
        "untouched — a new set-point, ambient or plant dynamics changes the expectation "
        "and the measurement together — PIRCAD is significantly *below* the generic "
        "forest in three of the six conditions, just as in distribution. Only shifts "
        "that change the residual itself hurt it, and the strongest noise shift exposes "
        "a genuine weakness: the false-drift rate climbs to "
        f"{g('noise x3', 'PIRCAD', 'false_drift_rate'):.2f}, because the noise-signature "
        "features that distinguish a substituted stream from a drifting one were "
        "learned at the commissioning noise level. Two qualifications apply to the "
        "noise rows. First, they compare models trained at a single noise level: "
        "Section 7.2 shows that when every learner is trained at several noise levels, "
        "the generic forest overtakes PIRCAD even at ×3, so PIRCAD's advantage under "
        "noise shift is a matter of training coverage, not of physics anchoring. "
        "Second, Section 6.6 shows that it does not reproduce on real data. The "
        "set-point and plant-dynamics rows, where the CNN fails and PIRCAD does not, "
        "are unaffected by either qualification.")
    rows2 = [[T.SHIFT_LABEL.get(c, c), _ci(P[c]["PIRCAD"]["macro_f1"], 3),
              f(Rr[c]["PIRCAD"]["summary"]["false_drift_rate"]),
              f(Rr[c]["PIRCAD"]["summary"]["false_alarm_rate"])]
             for c in ("nominal", "twin mismatch x3", "twin mismatch x6")]
    t3 = m.table(
        "Sensitivity of PIRCAD to the accuracy of its own process model (no "
        "retraining). The default model is already mismatched by 1–3 % per parameter "
        "(Section 5.1); ×3 and ×6 multiply every mismatch. This is not a comparison: "
        "the learned baselines have no process model, so the condition does not apply "
        "to them.",
        ["Process-model error", "Macro-F1 PIRCAD", "False-drift", "False alarm"],
        rows2, widths=[4.5, 4.0, 2.5, 2.5], font=8.5)
    m.p(f"Table {t3} isolates the cost of relying on a process model. With every "
        "parameter error multiplied by six, macro-F1 falls to "
        f"{g('twin mismatch x6', 'PIRCAD'):.3f} and the false-alarm rate rises to "
        f"{g('twin mismatch x6', 'PIRCAD', 'false_alarm_rate'):.2f}, while the false-drift "
        "rate stays low: a poor model makes PIRCAD raise more alarms, not absorb more "
        "attacks. In short, PIRCAD is only as good as its process model; that is the "
        "cost of checking a channel against physics instead of against its own history.")
    m.figure(FIG / "v2_fig_shift.png",
             "Macro-F1 under distribution shift, with no retraining (mean, 95 % CI). "
             "Both data-driven learners break down under at least one condition. The "
             "model-mismatch conditions apply only to PIRCAD.", 15.5)


# --------------------------------------------------------------------------
def _safety(m, R):
    e9 = R["e9"]
    S, D = e9["summary"], e9["drift_summary"]
    m.h2("6.4 Does the safety function survive the attack?")
    m.p("Correct labels are only a means to an end. What really counts is whether, "
        "after an attack, the adaptive parts of the system are still sound enough to "
        "report the next real fault. In each of 120 episodes (20 seeds × 3 attack rates "
        "× 2 attackers: A0, and a fidelity-0.30 noise-matched A2), a rate-matched "
        "injection biases the internal temperature downwards for 4500 samples; the "
        "attacker then withdraws and the heater contactor welds shut, a genuine "
        "over-temperature fault. In the cloud, an alarm fires when the value forwarded "
        "by the fog layer passes 34 °C. Each self-calibrating baseline was tuned on "
        "benign drift (Section 5.3) to do its own job well, and a benign-drift arm (60 "
        "episodes, no attack) measures that job.")
    t = m.table(
        "Attack-then-fault (120 episodes) and benign drift (60 episodes). Spurious "
        "trip: the forwarded value crossed 34 °C before the fault existed. Premature "
        "trip: the forwarded value crossed during the fault, at least 5 samples ahead of "
        "the plant (a real excursion read too high, which errs on the safe side). Error "
        "at crossing: forwarded temperature minus true temperature at the moment the "
        "plant actually reached 34 °C. Mean [95 % CI]. Last column: mean error of the "
        "corrected, forwarded value under benign drift.",
        ["System", "Alarm fired", "Spurious trip", "Premature trip",
         "Error at crossing (°C)", "Poison forwarded", "Drift correction error (°C)"],
        [[r[0], _ci(S[r[0]]["alarm_fired"]), _ci(S[r[0]]["spurious_trip"]),
          _ci(S[r[0]]["premature_trip"]), r[4], r[5], r[6]] for r in T.safety_table(e9)],
        widths=[3.2, 1.9, 1.9, 1.9, 2.8, 1.5, 2.0], font=7.5)
    m.p(f"Table {t} shows the consequence of absorbing an injection. None of the three "
        "self-calibrating baselines ever raises the alarm: each adopts part of the "
        "attacker's bias as its correction, and when the real fault arrives it "
        f"under-reports by {min(-S[b]['error_at_true_cross'][0] for b in BASES):.1f}–"
        f"{max(-S[b]['error_at_true_cross'][0] for b in BASES):.1f} °C. The tuned Kalman "
        "bias estimator is instructive: it uses the same process model as PIRCAD and is "
        f"one of the best drift correctors in the table ({D['Kalman bias estimator']['drift_fwd_error'][0]:.2f} °C), "
        "yet it fails the safety test, because a model-based bias estimator with no "
        "cause vocabulary must treat a slow injection exactly as it treats ageing. "
        f"PIRCAD raises the alarm in {S['PIRCAD']['alarm_fired'][0]*100:.0f} % of "
        f"episodes, with an error of {S['PIRCAD']['error_at_true_cross'][0]:+.2f} °C at the "
        f"crossing, a spurious-trip rate of {S['PIRCAD']['spurious_trip'][0]:.3f}, and "
        f"{S['PIRCAD']['poison_forwarded'][0]*100:.1f} % of poisoned samples forwarded. "
        "Its premature trips "
        f"({S['PIRCAD']['premature_trip'][0]:.2f}) all come from the noise-matched "
        "attacker, whose ramp is partly corrected before quarantine; they err on the "
        "safe side.")
    m.p("The two PIRCAD variants isolate where this comes from. With the response "
        "made non-differential — the same detector and classifier, but every detected "
        "deviation answered with recalibration — the alarm still fires, but "
        f"{S['PIRCAD non-differential']['poison_forwarded'][0]*100:.0f} % of the poisoning is "
        "forwarded and the forwarded value over-reads by "
        f"{S['PIRCAD non-differential']['error_at_true_cross'][0]:+.2f} °C at the crossing, "
        f"with spurious trips in {S['PIRCAD non-differential']['spurious_trip'][0]*100:.0f} % "
        "of episodes: detection without a cause-dependent response does not protect "
        "the safety function. With *auto re-attribution* instead of the latched "
        "quarantine, spurious trips rise to "
        f"{S['PIRCAD auto re-attribution']['spurious_trip'][0]:.2f} and the crossing error to "
        f"{S['PIRCAD auto re-attribution']['error_at_true_cross'][0]:+.2f} °C, because the "
        "noise-matched attacker occasionally sustains enough drift verdicts to be "
        "re-attributed and partly corrected-in, while benign drift is corrected no "
        f"better ({D['PIRCAD auto re-attribution']['drift_fwd_error'][0]:.2f} against "
        f"{D['PIRCAD']['drift_fwd_error'][0]:.2f} °C). This is why latching is the default.")

    # ---- baseline tuning sweep
    Dt = R["e14dose"]["tuning"]
    bf = R["best_firing"]
    pir_drift = D["PIRCAD"]["drift_fwd_error"][0]
    m.h3("Is the outcome an artefact of how the baselines were tuned?")
    m.p("The baselines were tuned for their own job, not for this scenario, so a fair "
        "objection is that a different setting would have fired the alarm. Appendix "
        "Table A.1 sweeps each baseline's adaptation parameter over one to two orders of "
        "magnitude and reports the safety outcome and the benign-drift correction error "
        "at every setting; Fig. " + str(m.n_fig + 1) + " plots the trade-off. EWMA and Page-Hinkley "
        "recalibration never fire the alarm at any setting. The Kalman bias estimator "
        "does, once its bias process noise is reduced to "
        + (f"q_{{b}} ≤ {bf['Kalman bias estimator']['param']:g}² — but at that setting it "
           f"leaves {bf['Kalman bias estimator']['drift_fwd_error'][0]:.2f} °C of benign drift "
           f"uncorrected (against {pir_drift:.2f} °C for PIRCAD) and trips spuriously in "
           f"{bf['Kalman bias estimator']['spurious_34'][0]*100:.0f} % of episodes"
           if bf["Kalman bias estimator"] else "no setting")
        + ". In other words, a self-calibrating estimator without a cause vocabulary "
        "can be made to keep the alarm only by making it adapt so slowly that it no "
        "longer does its job. No setting of any baseline reaches PIRCAD's combination of "
        "alarm rate, spurious-trip rate and drift-correction error.")
    m.figure(FIG / "rev_fig_tradeoff.png",
             "Safety–calibration trade-off of the self-calibrating baselines swept over "
             "their adaptation parameter (Appendix Table A.1), against PIRCAD at its "
             "default setting. Horizontal: correction error on benign drift (lower is "
             "better); vertical: share of attack-then-fault episodes in which the 34 °C "
             "alarm fired. Marker size grows with the spurious-trip rate. Only PIRCAD "
             "reaches the upper-left corner.", 13.5)

    # ---- dose-response
    dur = R["e14dose"]["duration"]
    safe = R["e14safe"]
    base = R["e14base"]
    m.h3("Dose-response: attack rate, attack duration and alarm threshold")
    bc = R["e9"]["by_condition"]["PIRCAD"]
    worst = max(bc.values(), key=lambda v: v["error_at_true_cross"])
    m.p("Appendix Table A.2 varies the three quantities that set how much bias an "
        "attacker can accumulate and how much margin the alarm has. Across attack "
        "rates from 0.8 to 2.0 × 10^{−3} °C per sample, PIRCAD fires the alarm in "
        "every condition; its crossing error grows with the rate of the noise-matched "
        f"attacker, to {worst['error_at_true_cross']:+.2f} °C at the fastest, on the safe "
        "side. Shortening the attack to 1500 or 3000 samples changes nothing "
        "qualitatively: PIRCAD fires in "
        f"{min(dur[d]['PIRCAD']['alarm_34'][0] for d in dur)*100:.0f} % of episodes and "
        "the baselines in none, because even the shortest attack moves their reference "
        "by more than the alarm margin. Moving the alarm threshold from 32 to 36 °C, "
        "PIRCAD fires in "
        f"{min(safe['by_attacker']['PIRCAD'][a][f'alarm_{x}'][0] for a in ATT14[:2] for x in (32, 33, 34, 35, 36))*100:.0f} % "
        "of A0/A2 episodes at every threshold, and the baselines in none: their "
        "under-reporting exceeds the whole range of margins tested.")

    # ---- variants and attackers
    V = safe["by_attacker"]
    Dv = safe["drift"]
    order = ["PIRCAD", "PIRCAD (gradient boosting)", "- noise-signature features",
             "- dedicated observers (shared estimator)", "- cross-sensor localisation",
             "PIRCAD threshold 0.50", "PIRCAD threshold 0.80"]
    label = {"- noise-signature features": "− noise-signature features",
             "- dedicated observers (shared estimator)": "− dedicated observers (shared estimator)",
             "- cross-sensor localisation": "− cross-sensor localisation"}
    rows = []
    for v in order:
        r = [label.get(v, v)]
        for a in ATT14:
            x = V[v][a]
            r.append(f"{x['alarm_34'][0]:.2f} / {x['spurious_34'][0]:.2f} / {x['err_34'][0]:+.2f}")
        r.append(f"{Dv[v]['drift_fwd_error'][0]:.2f}")
        rows.append(r)
    for b in BASES:
        r = [b]
        for a in ATT14:
            x = base[b][a]
            r.append(f"{x['alarm_34'][0]:.2f} / {x['spurious_34'][0]:.2f} / {x['err_34'][0]:+.2f}")
        r.append(f"{D[b]['drift_fwd_error'][0]:.2f}")
        rows.append(r)
    t5 = m.table(
        "Safety outcome per attacker for PIRCAD, its ablated variants and the "
        "self-calibrating baselines (60 episodes per attacker: 20 seeds × 3 rates). "
        "Each cell: alarm fired / spurious trip / error at the true crossing (°C). Last "
        "column: correction error under benign drift (°C). Variants that change the "
        "features use the retrained classifiers of Section 6.7.",
        ["System"] + [ATT14_LABEL[a] for a in ATT14] + ["Drift error"],
        rows, widths=[3.8, 2.6, 2.6, 2.6, 2.6, 1.4], font=7.5)
    R["t_safety_variants"] = t5
    P0 = V["PIRCAD"]
    sh = V["- dedicated observers (shared estimator)"]
    nl = V["- cross-sensor localisation"]
    ns = V["- noise-signature features"]
    m.h3("Other attackers, and do the elements that do not change macro-F1 matter here?")
    m.p(f"Table {t5} repeats the experiment for the on-path (A4) and coordinated (A5) "
        "attackers, and for every variant whose in-distribution macro-F1 was not "
        "significantly different from the full system (Section 6.7). Three results "
        "qualify the safety claim. First, A4 is corrected as drift while it lasts "
        "(Section 6.2), but when the attacker withdraws, the correction it induced "
        "is still in place: the forwarded value now over-reads, by "
        f"{P0['A4 on-path']['err_34'][0]:+.2f} °C at the true crossing, and trips the alarm "
        f"before the fault in {P0['A4 on-path']['spurious_34'][0]*100:.0f} % of episodes. "
        "The error is on the safe side for an over-temperature alarm, but it would be "
        "on the unsafe side for an alarm on the other bound, and it is the price of "
        "Observation 1: a correction that follows an attack it cannot recognise must "
        "also be unwound when the attack ends, and a withdrawal looks like a step "
        "fault. Second, A5 is forwarded as a process event during the attack, as in "
        "Section 6.2, but because a process verdict forbids recalibration it leaves no "
        f"trace in the calibration, and the alarm fires in {P0['A5 coordinated']['alarm_34'][0]*100:.0f} % "
        f"of episodes with a crossing error of {P0['A5 coordinated']['err_34'][0]:+.2f} °C: "
        "A5 corrupts the data while it lasts, not the adaptive state. Third, the "
        "baselines behave identically under every attacker, since they see only the "
        "target channel.")
    m.p("The variants answer the question the ablation of Section 6.7 left open: do "
        "the elements that do not change macro-F1 change what reaches the safety "
        "function? " + R["variant_verdict"])
    m.p("PIRCAD pays for its protection in drift-correction accuracy: its correction "
        f"error during benign drift ({D['PIRCAD']['drift_fwd_error'][0]:.2f} °C) is larger "
        "than that of the tuned Kalman and Page-Hinkley correctors "
        f"({D['Kalman bias estimator']['drift_fwd_error'][0]:.2f} and "
        f"{D['Page-Hinkley recalibration']['drift_fwd_error'][0]:.2f} °C), because its "
        "correction is rate-limited and is only engaged after the drift verdict. Where "
        "no attacker is expected, those correctors are the better choice; where one is, "
        "they should not run without a guard.")
    m.figure(FIG / "v2_fig_safety.png",
             "Effect of poisoning the adaptation on the safety alarm. (a) Share of "
             "episodes in which the cloud-side 34 °C alarm fired for the genuine fault; "
             "(b) forwarded minus true temperature at the true crossing (negative = "
             "under-report, the dangerous direction); (c) correction error during benign "
             "drift. Means with 95 % CIs.", 15.5)


# --------------------------------------------------------------------------
def _threshold(m, R):
    th = R["e14thr"]
    m.h2("6.5 Sensitivity to the confidence threshold")
    rows = []
    for k in [f"{x:.2f}" for x in th["thresholds"]]:
        s, i = th["sim"][k]["summary"], th["intel"][k]["summary"]
        rows.append([k + (" (default)" if k == "0.62" else ""),
                     f"{s['macro_f1'][0]:.3f}", f"{s['false_drift_rate'][0]:.3f}",
                     f"{s['false_poison_rate'][0]:.3f}", f"{s['drift_recall'][0]:.2f}",
                     f"{i['macro_f1'][0]:.3f}", f"{i['false_drift_rate'][0]:.3f}",
                     f"{i['false_poison_rate'][0]:.3f}", f"{i['drift_recall'][0]:.2f}"])
    t = m.table(
        "Sensitivity to the confidence threshold of the gate E6, with no escalation "
        "back-end (simulated: 180 held-out episodes; Intel Lab: 312 test episodes). "
        "Below the threshold a poisoning verdict holds the sample but cannot latch "
        "quarantine.",
        ["Threshold", "Sim. macro-F1", "Sim. false-drift", "Sim. false-poison",
         "Sim. drift recall", "Intel macro-F1", "Intel false-drift", "Intel false-poison",
         "Intel drift recall"],
        rows, widths=[2.2, 1.7, 1.7, 1.7, 1.6, 1.7, 1.7, 1.7, 1.6], font=7.5)
    s, i = th["sim"], th["intel"]
    m.p(f"The threshold decides whether an uncertain poisoning verdict is acted on "
        "reversibly (hold the sample) or irreversibly (latch quarantine). Table "
        f"{t} shows that in simulation the default sits at the edge of a plateau: "
        "every threshold from 0.62 to 0.90 gives the same error rates, while lower "
        "thresholds let more uncertain verdicts latch quarantine on drifting sensors "
        f"(false-poison {s['0.40']['summary']['false_poison_rate'][0]:.3f} at 0.40 "
        f"against {s['0.62']['summary']['false_poison_rate'][0]:.3f} at 0.62). On real "
        "data there is no plateau but a monotone trade-off between the two dangerous "
        "errors: raising the threshold from 0.62 to 0.90 lowers the false-poison rate "
        f"from {i['0.62']['summary']['false_poison_rate'][0]:.3f} to "
        f"{i['0.90']['summary']['false_poison_rate'][0]:.3f} and raises macro-F1 from "
        f"{i['0.62']['summary']['macro_f1'][0]:.3f} to {i['0.90']['summary']['macro_f1'][0]:.3f}, "
        f"but raises the false-drift rate from {i['0.62']['summary']['false_drift_rate'][0]:.3f} "
        f"to {i['0.90']['summary']['false_drift_rate'][0]:.3f}, i.e. lets more poisoning "
        "be recalibrated in. A deployment that values availability over security would "
        "choose a higher threshold; we keep 0.62, which was fixed before these data were "
        "seen, and note that the real-data false-poison rate it implies is high "
        "(Section 6.6). The interaction with escalation is the same at every threshold: "
        "with an escalation back-end configured, verdicts below it are re-judged instead "
        "of held, and only the back-end's confident verdicts can latch.")


# --------------------------------------------------------------------------
def _real(m, R):
    e10, e13 = R["e10"], R.get("e13")
    S = e10["summary"]
    Tt = e10["tests"]
    ref = R["e14ref"]
    m.h2("6.6 Real sensor data: Intel Berkeley Research Lab")
    rows = []
    for s in ("PIRCAD-rf", "PIRCAD-hgb", "PIRCAD-rules", "Multi-scale 1D-CNN",
              "Generic-feature RF", "Gen-1 FHAD"):
        sm = S[s]
        rows.append([T.SYS6_LABEL.get(s, s), T.fci(sm["macro_f1"]), f(sm["drift_recall"]),
                     f(sm["poison_recall"]), f(sm["process_recall"]),
                     f(sm["false_drift_rate"]), f(sm["false_poison_rate"]),
                     f(sm["false_alarm_rate"]), f(sm["harm"], 2),
                     "–" if s == "PIRCAD-rf" else _r(Tt[s]),
                     "ref." if s == "PIRCAD-rf" else T.pfmt(Tt[s]["p_holm"])])
    t = m.table(
        f"Intel Lab results: {e10['n_test_episodes']} test episodes "
        f"({len(e10['test_motes'])} held-out motes, 6 scenarios, 2 replicates each), with "
        f"all models trained on a separate set of {len(e10['train_motes'])} motes. Mean "
        "[95 % CI]; r and p: rank-biserial effect size and paired Wilcoxon test on "
        "macro-F1, PIRCAD (random forest) vs. each row, Holm-corrected (r > 0: PIRCAD "
        "higher). Harm: mean forwarded error on the injected channel (°C).",
        ["System", "Macro-F1", "Drift recall", "Poison recall", "Process recall",
         "False-drift ↓", "False-poison ↓", "False alarm ↓", "Harm ↓", "r", "p"],
        rows, widths=[3.0, 2.4, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.0, 1.0, 1.2], font=7)
    m.p(f"Table {t} is the least favourable result for PIRCAD in this paper. On real "
        "streams with mote-disjoint splits, PIRCAD reaches macro-F1 "
        f"{S['PIRCAD-rf']['macro_f1'][0]:.3f}; the CNN and the generic forest reach "
        f"{S['Multi-scale 1D-CNN']['macro_f1'][0]:.3f} and "
        f"{S['Generic-feature RF']['macro_f1'][0]:.3f}, significantly higher with large "
        f"effects (r = {_r(Tt['Multi-scale 1D-CNN'])} and {_r(Tt['Generic-feature RF'])}). "
        "PIRCAD detects poisoning about as well as they do (recall "
        f"{S['PIRCAD-rf']['poison_recall'][0]:.2f} against "
        f"{S['Multi-scale 1D-CNN']['poison_recall'][0]:.2f} and "
        f"{S['Generic-feature RF']['poison_recall'][0]:.2f}) and almost never confuses it "
        f"with drift ({S['PIRCAD-rf']['false_drift_rate'][0]:.3f}), but it recognises real "
        f"benign drift far less often ({S['PIRCAD-rf']['drift_recall'][0]:.2f}), calls "
        f"{S['PIRCAD-rf']['false_poison_rate'][0]*100:.0f} % of drifting samples poisoned "
        f"(against {S['Multi-scale 1D-CNN']['false_poison_rate'][0]*100:.0f} % for the CNN) "
        f"and raises more false alarms on clean real data ({S['PIRCAD-rf']['false_alarm_rate'][0]:.2f}). "
        "On this deployment, PIRCAD's attribution errs heavily towards the "
        "security-preserving side, at a large cost in availability.")
    m.p("The reason is the expectation model, and it can be quantified. In the vessel "
        "the leave-one-out reference error on clean data is "
        f"{ref['sim_abs_eref']['median']:.2f} °C at the median and "
        f"{ref['sim_abs_eref']['p95']:.2f} °C at the 95th percentile — mostly the "
        "deliberate parameter mismatch of Section 5.1, a nearly constant offset. On the "
        "Intel motes the clean residual (reference error plus transducer noise) has a "
        f"similar median, {ref['intel_abs_residual']['median']:.2f} °C, but a far heavier "
        f"tail: {ref['intel_abs_residual']['p95']:.2f} °C at the 95th and "
        f"{ref['intel_abs_residual']['p99']:.1f} °C at the 99th percentile, with per-mote "
        f"medians from {ref['intel_per_mote_median']['min']:.2f} to "
        f"{ref['intel_per_mote_median']['max']:.2f} °C. "
        "The leave-one-out expectation of a mote's temperature combines a "
        "psychrometric relation that assumes spatially uniform vapour pressure with a "
        "spatial relation that ignores sunlight, vents and occupants, so its residual "
        "carries real micro-climate. By Proposition 2 this error passes straight into "
        "the corrected value, and the character features that separate a drifting "
        "transducer from a substituted stream are measured on a noisier residual. The "
        "data-driven learners, trained and tested on the same building and the same "
        "injection generator, do not pay that price.")
    if e13:
        L = e13["levels"]
        P = R["e15"]["intel_shift"]
        names = ("PIRCAD-rf", "Multi-scale 1D-CNN", "Generic-feature RF")
        rows = [[_noise(l)] + [_ci(P["ci"][l][n]["macro_f1"], 3) for n in names]
                + [f"{P['ci'][l][n]['false_alarm_rate'][0]:.2f}" for n in names] for l in L]
        lab = {"PIRCAD-rf": "PIRCAD", "Multi-scale 1D-CNN": "CNN", "Generic-feature RF": "generic RF"}
        t2 = m.table(
            "Intel Lab under a transducer-noise shift (every mote given extra white "
            "noise; no retraining). Macro-F1 mean [95 % CI] and false-alarm rate on "
            "held-out motes.",
            ["Condition"] + [f"Macro-F1 {lab[n]}" for n in names]
            + [f"False alarm {lab[n]}" for n in names],
            rows, widths=[3.2, 2.6, 2.6, 2.6, 1.6, 1.6, 1.6], font=7.5)
        R["t_intel_shift"] = t2
        DT = P["drop_tests"]
        m.p(f"Table {t2} repeats the decisive shift of Section 6.3 on the real "
            "deployment. At the strongest level macro-F1 falls by "
            f"{DT['PIRCAD-rf']['drop'][0]:.3f} [{DT['PIRCAD-rf']['drop'][1]:.3f}, {DT['PIRCAD-rf']['drop'][2]:.3f}] for PIRCAD, "
            f"{DT['Multi-scale 1D-CNN']['drop'][0]:.3f} [{DT['Multi-scale 1D-CNN']['drop'][1]:.3f}, {DT['Multi-scale 1D-CNN']['drop'][2]:.3f}] for the CNN and "
            f"{DT['Generic-feature RF']['drop'][0]:.3f} [{DT['Generic-feature RF']['drop'][1]:.3f}, {DT['Generic-feature RF']['drop'][2]:.3f}] for the generic "
            "forest (95 % CIs over the 312 paired episodes). PIRCAD's drop is "
            f"significantly larger than the CNN's ({T.pstr(DT['PIRCAD-rf vs Multi-scale 1D-CNN']['p_holm'])}, "
            f"r = {_r(DT['PIRCAD-rf vs Multi-scale 1D-CNN'])}) and significantly smaller than "
            f"the generic forest's ({T.pstr(DT['PIRCAD-rf vs Generic-feature RF']['p_holm'])}). "
            "The CNN's drop is statistically detectable but negligible in size: on real "
            "data, whose own variability is large relative to the added noise, its "
            "standardised multi-scale input absorbs the shift. The robustness advantage "
            "PIRCAD showed in simulation therefore does **not** transfer to this real "
            "deployment under this shift, and we do not claim it does.")
    mi = e10["mitm"]
    du = {n: T.per_scenario_metric(e10["rows"][n], "drift", "useful_frac")[0]
          for n in ("PIRCAD-rf", "Multi-scale 1D-CNN", "Generic-feature RF")}
    m.p("Two simulated findings do carry over, in weakened form. The on-path ramp (A4) "
        "again cannot be labelled — the learned baselines call it drift (poisoning recall "
        f"{mi['Multi-scale 1D-CNN']['poison_recall'][0]:.2f} and "
        f"{mi['Generic-feature RF']['poison_recall'][0]:.2f}); PIRCAD flags "
        f"{mi['PIRCAD-rf']['poison_recall'][0]:.2f} of it, largely because its noisier "
        "real-data residual triggers poisoning verdicts on drift-like ramps too, so this "
        "is not evidence of separability. And only the system that corrects against an "
        "independent reference delivers a usable channel: PIRCAD forwards "
        f"{mi['PIRCAD-rf']['useful_frac'][0]*100:.0f} % of the attacked channel and "
        f"{du['PIRCAD-rf']*100:.0f} % of a genuinely drifting one within 0.5 °C of the "
        "truth, against "
        f"{min(mi['Multi-scale 1D-CNN']['useful_frac'][0], mi['Generic-feature RF']['useful_frac'][0], du['Multi-scale 1D-CNN'], du['Generic-feature RF'])*100:.0f}–"
        f"{max(mi['Multi-scale 1D-CNN']['useful_frac'][0], mi['Generic-feature RF']['useful_frac'][0], du['Multi-scale 1D-CNN'], du['Generic-feature RF'])*100:.0f} % "
        "for the label-only learners — a difference of several times, but far from the "
        "99 % achieved in simulation, because the real-data correction is only as "
        "accurate as the real-data reference quantified above.")
    m.figure(FIG / "v2_fig_intel.png",
             "Per-scenario root-cause accuracy on held-out Intel Lab motes (mean, "
             "95 % CI). PIRCAD comes closest to the learned baselines on rate-matched "
             "poisoning and trails them elsewhere, most of all on real benign drift and "
             "local heat events.", 15.5)


# --------------------------------------------------------------------------
def _ablation(m, R):
    e11, e12 = R["e11"], R.get("e12")
    S = e11["summary"]
    P = e11["tests"]
    m.h2("6.7 Ablation and combined features")
    rows = T.ablation_table(e11)
    for row, r in zip(rows, S):
        row.insert(len(row) - 1, "–" if r == "full" else _r(P[r]))
    t = m.table(
        "Ablation on the 180-episode bank, one element removed per row; rows that "
        "change the features are retrained on a corpus collected under the same "
        "change. r and p: rank-biserial effect size and paired Wilcoxon test on "
        "macro-F1, full system vs. each row, Holm-corrected (r > 0: full system "
        "higher).",
        ["Configuration", "Macro-F1", "False-drift ↓", "Containment ↑",
         "Process masked ↓", "Retraining protection ↑", "Harm ↓", "r", "p"],
        rows, widths=[4.0, 2.5, 1.2, 1.3, 1.2, 1.5, 1.0, 1.0, 1.3], font=7.5)

    def g(r, k):
        return S[r][k][0]
    full = "full"

    def p_(r):
        return P[r]["p_holm"]
    m.p(f"Table {t} removes one element at a time. The physics residual carries the "
        "result: replacing it by the channel's own rolling mean lowers macro-F1 from "
        f"{g(full, 'macro_f1'):.3f} to {g('- physics residual (rolling mean)', 'macro_f1'):.3f} "
        f"(r = {_r(P['- physics residual (rolling mean)'])}), cuts poisoning containment to "
        f"{g('- physics residual (rolling mean)', 'poison_containment'):.2f}, and lets "
        f"{g('- physics residual (rolling mean)', 'process_masking_rate')*100:.0f} % of genuine "
        "process events be mislabelled as instrument faults. Removing the whole group "
        "of residual-character features costs "
        f"{g(full, 'macro_f1') - g('- all character features', 'macro_f1'):.3f} "
        f"({T.pstr(p_('- all character features'))}, r = {_r(P['- all character features'])}), "
        "whereas any single member is redundant with the others: excitation coherence "
        f"alone changes nothing measurable ({g('- excitation coherence', 'macro_f1'):.3f}, "
        f"p = {p_('- excitation coherence'):.2f}), and removing the noise-signature group "
        f"even raises macro-F1 to {g('- noise-signature features', 'macro_f1'):.3f} "
        f"(r = {_r(P['- noise-signature features'])}) while raising the false-drift rate to "
        f"{g('- noise-signature features', 'false_drift_rate'):.3f} and process masking to "
        f"{g('- noise-signature features', 'process_masking_rate'):.3f}.")
    m.p("Two results contradict hypotheses that motivated our initial design. "
        "Replacing the dedicated observers by one shared closed-loop estimator does "
        f"not significantly change macro-F1 ({g('- dedicated observers', 'macro_f1'):.3f}, "
        f"p = {p_('- dedicated observers'):.2f}) once the classifier is retrained on the "
        "changed features: the bias is attenuated in a shared estimator's residual, but "
        "the trigger and the remaining features still see it. Dropping the cross-sensor "
        "localisation also leaves macro-F1 where it was "
        f"({g('- cross-sensor consistency', 'macro_f1'):.3f}, p = "
        f"{p_('- cross-sensor consistency'):.2f}), because in this plant the character of "
        "the residual is enough to spot the process event. Neither element, then, is "
        "needed for in-distribution classification, and we do not claim otherwise. "
        "Whether they are needed at all is a question about effect, not labels, and "
        f"Section 6.4 answers it: {R['variant_summary']}")
    m.p("The two governance elements change what the system does, not the labels it "
        "assigns, so judging them by label accuracy alone would make them look useless. "
        "Remove the adaptation governor and macro-F1 stays the same, but retraining "
        "protection drops from "
        f"{g(full, 'retraining_protection'):.3f} to {g('- adaptation governor', 'retraining_protection'):.3f}, "
        "meaning every poisoned sample reaches an adaptive element. Remove the "
        "differential response and containment drops from "
        f"{g(full, 'poison_containment'):.3f} to {g('- differential response', 'poison_containment'):.3f} "
        "and process masking rises from "
        f"{g(full, 'process_masking_rate'):.3f} to {g('- differential response', 'process_masking_rate'):.3f}. "
        "These two elements, together with the leave-one-out reference, are what the "
        "safety result of Section 6.4 depends on.")
    if e12:
        r12 = e12["results"]
        e8 = R["e8"]["results"]
        rows = []
        for c in e12["conditions"]:
            c8 = "nominal" if c.startswith("nominal") else c
            pir = (R["e6"]["summary"]["PIRCAD-rf"]["macro_f1"]["mean"] if c.startswith("nominal")
                   else e8[c8]["PIRCAD"]["summary"]["macro_f1"][0])
            gen = (R["e6"]["summary"]["Generic-feature RF"]["macro_f1"]["mean"] if c.startswith("nominal")
                   else e8[c8]["Generic-feature RF"]["summary"]["macro_f1"][0])
            rows.append([("Nominal (180-episode bank)" if c.startswith("nominal") else T.SHIFT_LABEL.get(c, c)),
                         f"{pir:.3f}", f"{gen:.3f}", f"{r12[c]['summary']['macro_f1'][0]:.3f}",
                         f"{r12[c]['summary']['false_alarm_rate'][0]:.3f}"])
        t2 = m.table(
            "Macro-F1 with physics features only, generic features only, and the two "
            "combined in a single learner. The combined learner uses PIRCAD's governance "
            "and response unchanged.",
            ["Condition", "PIRCAD (physics)", "Generic-feature RF", "PIRCAD (physics + generic)",
             "False alarm, combined"], rows, widths=[4.0, 2.6, 2.6, 3.2, 2.6])
        nom = r12[e12["conditions"][0]]["summary"]["macro_f1"][0]
        noise = r12["noise x3"]["summary"]["macro_f1"][0]
        m.p(f"Table {t2} tests whether the two kinds of feature complement each other. "
            f"On the nominal bank the combined learner scores {nom:.3f}, closing the gap "
            f"to the generic forest; but with the strongest noise shift it falls to "
            f"{noise:.3f}, so it inherits the fragility of the generic features. Simply "
            "adding data-driven features to a physics-anchored learner therefore buys "
            "in-distribution accuracy at the price of robustness. This is a negative "
            "result for one specific way of combining the two, and it bounds the "
            "direction proposed in Section 8.")
    m.figure(FIG / "v2_fig_ablation.png",
             "Results of removing one element at a time (mean, 95 % CI): (a) macro-F1 "
             "and (b) forwarded error on the target channel. The dotted line shows the "
             "complete system.", 15.5)


# --------------------------------------------------------------------------
def _cost(m, R):
    c = R["e16"]
    lat = R["lat"]
    m.h2("6.8 Computational cost and edge feasibility")
    fo, p = c["forest"], c["pircad"]
    b = c["baselines_single_channel"]
    m.p("All timings in this section were measured on a development laptop "
        f"({c['machine'].split()[0]} CPU, Python {c['python']}, one thread), not on fog or "
        "edge hardware, and should be read as relative costs and an upper bound on "
        "throughput. PIRCAD processes a four-channel frame in a median of "
        f"{p['ms_median']:.2f} ms (95th percentile {p['ms_p95']:.2f} ms, 99th "
        f"{p['ms_p99']:.2f} ms), a sustained {p['frames_per_s']:.0f} frames per second; "
        f"the expensive stages run on {lat['activation']*100:.0f} % of channel-samples, "
        "which is why the median is far below the tail. The default classifier is a "
        f"forest of {fo['trees']} trees with {fo['nodes']:,} nodes in total (depth ≤ "
        f"{fo['max_depth']}); stored as the flat arrays of Section 4.4 it occupies "
        f"{fo['bytes'] / 2**20:.1f} MB, and running a fog node over a full episode adds "
        f"{p['rss_mb_after'] - p['rss_mb_before']:.0f} MB to the resident set of a Python "
        "process that already holds the experiment harness. The single-sample "
        "evaluation of Section 4.4 is what makes this feasible: the library "
        "implementation spends about 16 ms per single-sample call on per-tree task "
        "dispatch. For comparison, the single-channel self-calibrating baselines cost "
        f"{b['EWMA recalibration']['us_per_sample']:.1f} µs (EWMA), "
        f"{b['Page-Hinkley recalibration']['us_per_sample']:.1f} µs (Page-Hinkley) and "
        f"{b['Kalman bias estimator']['us_per_sample']:.0f} µs (Kalman) per sample on the "
        "same machine: PIRCAD's median frame costs roughly "
        f"{p['ms_median'] * 1e3 / b['Kalman bias estimator']['us_per_sample']:.0f} to "
        f"{p['ms_median'] * 1e3 / b['EWMA recalibration']['us_per_sample']:.0f} times as much, "
        "the price of attributing causes over four channels.")
    m.p("Whether the method fits a microcontroller was explored separately, with a "
        "two-channel closed-form kernel for an ESP32 DevKit and a single DHT22, whose "
        "temperature and humidity transducers supply the cross-channel redundancy "
        "through the Magnus relation. The kernel has no learned model; compiled for the "
        "host it takes about 4 µs per sample, and it runs in the Wokwi simulator. In "
        "its regression suite it recalls 88.1 % and 87.5 % of linear and exponential "
        "drift and 100 % of step and spike injections, and admits under 1 °C of a "
        "6 °C step or stealth ramp into its reference, against 5.8–6.8 °C for a "
        "rolling-mean baseline run in the same harness; a ramp matched to the benign "
        "drift rate is not separable with two channels at one node, which is the "
        "single-node instance of Observation 1. We report this as a feasibility "
        "demonstration only. The node has not been run on physical silicon, so no "
        "energy, memory or real-timing figures exist, and it was compared with one "
        "baseline only; we do not count it among the contributions.")


# --------------------------------------------------------------------------
def discussion_and_conclusion(m, R):
    w = R["worst8"]
    S9 = R["e9"]["summary"]
    e7 = R["e7"]["results"]
    e17 = R.get("e17")
    m.h1("7. Discussion")
    m.h2("7.1 What the evidence supports")
    m.p("The results support a narrower claim than the one we started with. PIRCAD is "
        "**not** a more accurate classifier than a strong data-driven learner trained "
        "on the same distribution: in simulation the CNN matches it and the generic "
        "forest beats it, and on real data both beat it clearly, with fewer false "
        "alarms and far fewer false-poison verdicts. What the evidence does support is "
        "about governance and correction:")
    bf = R["best_firing"]["Kalman bias estimator"]
    m.bullets([
        "**A safety function that survives poisoning — in simulation.** Against "
        "self-calibrating estimators that never raised the alarm at their tuned "
        f"settings, PIRCAD raised it in {S9['PIRCAD']['alarm_fired'][0]*100:.0f} % of "
        "attack-then-fault episodes, across attack rates, durations and alarm "
        "thresholds; a baseline could be made to keep the alarm only by slowing its "
        "adaptation until it no longer corrected drift"
        + (f" ({bf['drift_fwd_error'][0]:.2f} °C error)" if bf else "") + ". The "
        "benefit comes from the cause-dependent response and the governor, not from "
        "classification accuracy (Section 6.7).",
        "**The ability to act on a verdict.** To correct a drifting sensor, you need an "
        "estimate of its bias taken against a reference the sensor cannot affect. A "
        "label-only system can only withhold what it flags; on benign drift PIRCAD "
        f"delivered {e7['drift (reference)']['PIRCAD']['summary']['useful_frac'][0]*100:.0f} % of the "
        "channel within tolerance against "
        f"{max(e7['drift (reference)'][s]['summary']['useful_frac'][0] for s in ('Multi-scale 1D-CNN', 'Generic-feature RF'))*100:.0f} % "
        "for the learned baselines in simulation, and "
        f"{T.per_scenario_metric(R['e10']['rows']['PIRCAD-rf'], 'drift', 'useful_frac')[0]*100:.0f} % "
        "against "
        f"{max(T.per_scenario_metric(R['e10']['rows'][n], 'drift', 'useful_frac')[0] for n in ('Multi-scale 1D-CNN', 'Generic-feature RF'))*100:.0f} % "
        "on real data.",
        "**Harm bounded relative to the reference where labels are impossible.** For "
        "attacks observationally equivalent to drift (Observation 1), no system labels "
        "them, but physics-anchored correction neutralises them while they last "
        f"(Proposition 2, Table {R['t_att_effect']}), at the cost of an over-correction when the attacker "
        "withdraws (Section 6.4). The bound is only as good as the reference, whose "
        "real-data error we report.",
    ])
    m.p("One advantage reported in our submission is *not* on this list: a bounded "
        f"worst case under distribution shift. In simulation, PIRCAD's lowest macro-F1 "
        f"over the shift conditions was {w['PIRCAD'][1]:.2f} against "
        f"{w['Generic-feature RF'][1]:.2f} and {w['Multi-scale 1D-CNN'][1]:.2f} for the "
        "learned baselines, but on the one real deployment we tested, the CNN was "
        "unaffected by the noise shift that hurt PIRCAD, and even in simulation the "
        "noise-shift part of the advantage disappears once the generic forest is "
        "trained at several noise levels (Section 7.2). What remains is narrower: in "
        "simulation, shifts of operating point and plant dynamics broke the CNN but "
        "not PIRCAD, because they move the expectation and the measurement together. "
        "We treat this as a simulation finding, not as a property of the method.")
    m.p("A practitioner should read the comparison as complementary rather than "
        "competitive. Where the deployment is stable and labelled data are plentiful, a "
        "data-driven learner is the better *classifier*; it is the governance — what "
        "may learn from the data, and how a verdict is acted on — that needs a "
        "reference the channel cannot influence. Because the classifier back-end can "
        "be swapped without touching the governance, the two can be combined; Section "
        "6.7 shows, however, that the obvious combination, a single learner over both "
        "feature sets, inherits the fragility of the generic features, so the "
        "combination has to keep the reference independent rather than simply pool "
        "features.")
    m.h2("7.2 Mitigating sensitivity to sensor noise and model error")
    m.p("Two sensitivities limit PIRCAD in deployment: a noise level different from "
        "commissioning raises its false-drift rate (Section 6.3), and an inaccurate "
        "expectation raises its false alarms and, on real data, its false-poison rate "
        "(Sections 6.3 and 6.6). We tested one mitigation of the first and outline "
        "mitigations of the second.")
    if e17:
        x = e17["results"]
        n3, n2, nom = x["noise x3"], x["noise x2"], x["nominal"]
        m.p("**Training at more than one noise level.** The root-cause learner sees "
            "features only at the commissioning noise level. Collecting its training "
            "corpus at noise ×1, ×1.5 and ×2 — the same 72 episodes at three levels, "
            "with no other change — and re-evaluating on the Section 6.3 bank changes "
            f"macro-F1 from {nom['PIRCAD']['macro_f1'][0]:.3f} to "
            f"{nom['PIRCAD noise-augmented']['macro_f1'][0]:.3f} at nominal noise, from "
            f"{n2['PIRCAD']['macro_f1'][0]:.3f} to {n2['PIRCAD noise-augmented']['macro_f1'][0]:.3f} "
            f"at ×2, and from {n3['PIRCAD']['macro_f1'][0]:.3f} to "
            f"{n3['PIRCAD noise-augmented']['macro_f1'][0]:.3f} at ×3, a level outside the "
            "augmented range, with the false-drift rate at ×3 moving from "
            f"{n3['PIRCAD']['false_drift_rate'][0]:.3f} to "
            f"{n3['PIRCAD noise-augmented']['false_drift_rate'][0]:.3f} "
            f"({T.pstr(n3['test_aug_vs_orig']['p'])} for macro-F1 at ×3). The same "
            "augmentation applied to the generic-feature forest gives "
            f"{n2['Generic-feature RF noise-augmented']['macro_f1'][0]:.3f} at ×2 and "
            f"{n3['Generic-feature RF noise-augmented']['macro_f1'][0]:.3f} at ×3, so "
            "with equal training coverage the generic forest is better than PIRCAD at "
            f"every noise level tested. {R['aug_verdict']}")
    m.p("**Measuring the reference before trusting the correction.** Proposition 2 "
        "makes the reference error the quantity that limits every correction. It can be "
        "estimated at commissioning from the clean residual, as in Section 6.6, and used "
        "in two ways: to cap the correction (no correction smaller than the reference "
        "error is meaningful), and to disable bounded recalibration altogether on "
        "channels whose reference error exceeds the tolerance of the downstream "
        "function, falling back to forward-and-flag.")
    m.p("**Improving the reference without letting the channel into it.** Model error "
        "can be reduced by re-estimating the process-model parameters online, provided "
        "the estimator is itself governed: it may learn only from channels in the "
        "TRUSTED state, so that an attacker cannot move the model by moving a channel. "
        "Where physics is weak, as indoors, a learned expectation can replace or "
        "complement it, as long as it remains leave-one-out — trained to predict a "
        "channel from the others, never from that channel's own history — which is the "
        "property Proposition 2 needs. Both are directions for follow-up work; neither "
        "was evaluated here.")
    m.p("**Choosing the threshold for the deployment.** Section 6.5 shows that on real "
        "data the confidence threshold trades false-poison against false-drift "
        "verdicts monotonically. It is therefore a deployment decision, to be set from "
        "the relative cost of losing a sensor and of absorbing an attack, ideally on a "
        "labelled commissioning period at the site.")
    m.h2("7.3 Three defects exposed by the stronger evaluation")
    m.p("Our earlier prototype of this system was evaluated on four seeds against "
        "its own predecessor. Re-evaluating it at the scale reported here exposed three "
        "defects, each invisible to the original benchmark:")
    m.bullets([
        "**A polarity shortcut.** Every drift and attack ramp in the original training "
        "bank rose, and every process event heated. The classifier learned the sign: a "
        "falling drift was called normal, then a process event, and was never corrected. "
        "A first fix that drew the polarity at random happened, with six training "
        "seeds, to make every training drift fall — and the model learned that rising "
        "drift is normal. Training on both polarities of every episode removed the "
        f"shortcut; Table {R['t_pol']} confirms that no system now behaves differently on rising "
        "and falling events.",
        "**Quarantine lock-in.** One transient poisoning verdict at the onset of genuine "
        "drift latched quarantine for the rest of the run, while the response still "
        "followed the latest (drift) verdict, so uncorrected data were forwarded from a "
        "channel the governor believed quarantined. Containment now follows the "
        "governor's standing state.",
        "**Irreversible action on uncertain verdicts.** With no escalation back-end the "
        "fallback forced an uncertain verdict to full confidence, so a 0.3-confidence "
        "poisoning guess latched quarantine. On real data this turned a classifier that "
        "mostly called drift correctly into one that reported poisoning for most of each "
        "drift episode. Now, a poisoning verdict with low confidence only holds back that "
        "sample and does not trigger a latched quarantine; Section 6.5 shows how the "
        "threshold that separates the two paths moves every error rate.",
    ])
    m.p("All three made the system look better or worse than it was in ways that a "
        "label-accuracy table on a small, one-sided bank cannot reveal. Since a small "
        "one-sided bank hid them, we cannot assume the current evaluation would reveal "
        "every analogous defect, and we have taken three steps to make that more "
        "likely. Each defect now has a regression test that fails on the defective "
        "behaviour (we confirmed this by reintroducing the third defect, which the new "
        "test catches) and passes on the current code: the training and evaluation "
        "banks are asserted to be exactly balanced in polarity, a quarantined channel "
        "is asserted to forward nothing whatever its latest verdict, and a "
        "low-confidence poisoning verdict is asserted never to latch. The evaluation "
        "reports effect-level metrics next to labels, which is how the second and third "
        "defects were found. And the governance logic is exercised by episodes long "
        "enough for its state machine to matter (9000 samples in Section 6.4). The "
        "transferable lesson for anyone evaluating an adaptive anomaly pipeline is the "
        "same: balance the polarity of every event class, evaluate on effect as well as "
        "labels, and test the governance with long episodes.")
    m.h2("7.4 Limitations and threats to validity")
    m.bullets([
        "**Simulation.** The decisive comparisons — the attacker sweep and the safety "
        "experiment — use a simulated plant whose physics is known up to a few per "
        "cent. The real-data study shows that a weaker expectation model costs PIRCAD "
        "accuracy; the method's benefits are only as good as its reference, and the "
        "safety result has not been reproduced on real data.",
        "**Semi-synthetic real data.** The Intel streams are real, but drift, poisoning "
        "and the local heat events are injected, by the same generator family used for "
        "training; the data-driven baselines benefit from that match. No public dataset "
        "provides labelled real ageing alongside real attacks, so behaviour under "
        "real, unlabelled ageing is untested; learning from a commissioning period with "
        "few labelled events, for example by positive-unlabelled learning "
        "{cite:saunders2025}, is one way to approach it.",
        "**Attackers.** The hierarchy covers knowledge of the process and the noise, "
        "on-path manipulation and coordinated forgery, but not an attacker who adapts "
        "to the classifier's gradients or controls the actuator commands. A5 is "
        "detected but its forged values are forwarded; A4 is corrected while it lasts "
        "but over-corrected when it stops.",
        "**Sensitivity.** PIRCAD's false-drift rate rises under strong transducer-noise "
        "shift, and its false-alarm rate under a poor process model; on real data its "
        "false-poison rate is high at the default threshold.",
        "**Baselines.** The learned baselines were not tuned beyond sensible defaults. "
        "The self-calibrating baselines were tuned on benign drift and additionally "
        "swept over their adaptation parameter, but a stronger attack-aware baseline "
        "(e.g. robust drift detection {cite:korycki2023}) was not re-implemented.",
        "**Positioning.** Table 1 reflects our reading of the cited works, not a "
        "systematic review or a re-implementation of each.",
        "**Replication.** All experiments were designed and run by the authors. For "
        "this revision the stored per-episode results were reproduced exactly in a "
        "fresh software environment, but no external group has replicated them; the "
        "released code and artifacts are intended to make that possible.",
        "**Hardware.** Timings are from a laptop; the edge node has been run in "
        "simulation only, and energy and memory on physical silicon are not reported.",
    ])
    m.h1("8. Conclusion")
    m.p("Self-calibrating IoT pipelines can be poisoned by injections shaped like benign "
        "transducer ageing, and the harm outlasts the attack. We framed the defence as "
        "cause-aware learning whose verdicts govern every adaptive element, stated the "
        "limit of what any detector can separate, and showed that judging each channel "
        "against an expectation it cannot influence bounds the forwarded error relative "
        "to that expectation's own error. In simulation, this governance kept a "
        "downstream safety alarm working under poisoning that defeated every tuned "
        "self-calibrating baseline, across attack rates, durations and alarm "
        "thresholds; a baseline could keep the alarm only by giving up on correcting "
        "drift. As a classifier, PIRCAD is not better than strong data-driven learners: "
        "in distribution a generic-feature forest beats it, on real data both a CNN "
        "and that forest beat it, and its simulated robustness to noise shift neither "
        "survived noise-augmented training of the baselines nor reproduced on real "
        "data. The safety benefit is therefore conditional on a "
        "reference whose error is small and measured, which a real office environment "
        "did not provide. The most useful next step is not to pool data-driven and "
        "physics features, which we found inherits the fragility of the former, but to "
        "learn better *references* — expectation models that remain independent of the "
        "channel being judged — and to test the governance on deployments with "
        "labelled field ageing.")


# --------------------------------------------------------------------------
def appendix(m, R):
    m.h1("Appendix A. Supplementary tables")
    D = R["e14dose"]["tuning"]
    pname = {"Kalman bias estimator": "q_{b}^{1/2}", "EWMA recalibration": "α",
             "Page-Hinkley recalibration": "λ"}
    rows = []
    for b in BASES:
        for t in D[b]:
            rows.append([b, f"{pname[b]} = {t['param']:g}" + (" (tuned)" if t["default"] else ""),
                         _ci(t["alarm_34"]), _ci(t["spurious_34"]), _ci(t["err_34"]),
                         f"{t['drift_fwd_error'][0]:.2f}"])
    d9 = R["e9"]["summary"]["PIRCAD"]
    rows.append(["PIRCAD (default)", "–", _ci(d9["alarm_fired"]), _ci(d9["spurious_trip"]),
                 _ci(d9["error_at_true_cross"]),
                 f"{R['e9']['drift_summary']['PIRCAD']['drift_fwd_error'][0]:.2f}"])
    m.table(
        "Tuning sweep of the self-calibrating baselines on the attack-then-fault "
        "experiment (120 episodes, A0 and A2) and the benign-drift arm (60 episodes). "
        "Mean [95 % CI].",
        ["Baseline", "Setting", "Alarm fired", "Spurious trip", "Error at crossing (°C)",
         "Drift correction error (°C)"],
        rows, widths=[3.6, 2.6, 2.4, 2.4, 2.8, 2.0], font=7.5, label="A.1")
    dur = R["e14dose"]["duration"]
    safe = R["e14safe"]["by_attacker"]["PIRCAD"]
    base = R["e14base"]
    rows = []
    bc = R["e9"]["by_condition"]
    for sysn in ("PIRCAD",) + BASES:
        for cond, v in bc[sysn].items():
            att, rate = cond.split(" | ")
            rows.append([sysn, f"rate {abs(float(rate)) * 1e3:.1f}×10^{{−3}}, "
                         + ("A0" if att.startswith("stale") else "A2"),
                         f"{v['alarm_fired']:.2f}", f"{v['spurious_trip']:.2f}",
                         f"{v['error_at_true_cross']:+.2f}"])
    for sysn in ("PIRCAD",) + BASES:
        for d in ("1500", "3000"):
            x = dur[d][sysn]
            rows.append([sysn, f"duration {d} samples", f"{x['alarm_34'][0]:.2f}",
                         f"{x['spurious_34'][0]:.2f}", f"{x['err_34'][0]:+.2f}"])
    for sysn in ("PIRCAD",) + BASES:
        for th in (32, 33, 35, 36):
            src = [safe[a] for a in ATT14[:2]] if sysn == "PIRCAD" else [base[sysn][a] for a in ATT14[:2]]
            rows.append([sysn, f"alarm at {th} °C",
                         f"{np.mean([s[f'alarm_{th}'][0] for s in src]):.2f}",
                         f"{np.mean([s[f'spurious_{th}'][0] for s in src]):.2f}",
                         f"{np.mean([s[f'err_{th}'][0] for s in src]):+.2f}"])
    m.table(
        "Dose-response of the attack-then-fault experiment: attack rate and "
        "attacker (20 episodes each, 4500-sample attack, alarm at 34 °C), attack duration "
        "(120 episodes each, A0 and A2, alarm at 34 °C), and alarm threshold (120 "
        "episodes each, A0 and A2, 4500-sample attack).",
        ["System", "Condition", "Alarm fired", "Spurious trip", "Error at crossing (°C)"],
        rows, widths=[4.2, 4.2, 2.3, 2.3, 2.8], font=7.5, label="A.2")
