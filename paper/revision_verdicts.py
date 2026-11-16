"""Narrative conclusions drawn from experiments 14 and 17.

Kept separate because they are judgements on the numbers, written after the
numbers were seen; every figure they quote is still read from the artifacts.
"""


def _g(R):
    return R["e14safe"]["by_attacker"], R["e14safe"]["drift"], R["e11"]["summary"]


def variant_verdict(R):
    V, D, A = _g(R)
    P = V["PIRCAD"]
    sh, nl, ns = (V["- dedicated observers (shared estimator)"],
                  V["- cross-sensor localisation"], V["- noise-signature features"])
    gb = V["PIRCAD (gradient boosting)"]
    return (
        "The answer differs by element. The shared estimator does not change the "
        "safety outcome in this plant either: every variant fires the alarm under A0 "
        "and A2, its crossing errors stay within a few tenths of a degree of the full "
        f"system's, and its drift-correction error ({D['- dedicated observers (shared estimator)']['drift_fwd_error'][0]:.2f} °C) "
        "is no worse. Remark 1 therefore did not materialise here, and we say so: the "
        "case for the leave-one-out design rests on Proposition 2 and on plants where "
        "a shared estimator weights the channel more heavily, not on a measured benefit "
        "in this one. Cross-sensor localisation, by contrast, matters for effect even "
        "though it does not matter for labels: without it, "
        f"{nl['fidelity 0.3/matched']['poison_forwarded'][0]*100:.0f} % of the noise-matched "
        f"A2 poisoning is forwarded (full system {P['fidelity 0.3/matched']['poison_forwarded'][0]*100:.0f} %) "
        "and the benign-drift correction error doubles "
        f"({D['- cross-sensor localisation']['drift_fwd_error'][0]:.2f} against "
        f"{D['PIRCAD']['drift_fwd_error'][0]:.2f} °C), because more drift episodes "
        "are held instead of corrected. Localisation is also what lets A5 through: "
        "without it the coordinated forgery is withheld "
        f"({nl['A5 coordinated']['poison_forwarded'][0]*100:.0f} % forwarded against "
        f"{P['A5 coordinated']['poison_forwarded'][0]*100:.0f} %), since nothing can then call it "
        "a process event — the same mechanism that protects genuine process events. "
        "Removing the noise-signature features does not hurt the safety outcome "
        f"(A2 crossing error {ns['fidelity 0.3/matched']['err_34'][0]:+.2f} against "
        f"{P['fidelity 0.3/matched']['err_34'][0]:+.2f} °C); its cost is the higher "
        "false-drift and process-masking rates of Section 6.7. Gradient boosting keeps "
        "the alarm, but in the 9000-sample drift episodes it calls "
        f"{D['PIRCAD (gradient boosting)']['drift_as_poison'][0]*100:.0f} % of drifting "
        f"samples poisoned (random forest {D['PIRCAD']['drift_as_poison'][0]*100:.0f} %), "
        "which confirms the choice of default. Raising the confidence threshold to 0.80 "
        "lets more of the noise-matched attacker be corrected in (A2 crossing error "
        f"{V['PIRCAD threshold 0.80']['fidelity 0.3/matched']['err_34'][0]:+.2f} °C, spurious "
        f"trips {V['PIRCAD threshold 0.80']['fidelity 0.3/matched']['spurious_34'][0]:.2f}).")


def variant_summary(R):
    V, D, A = _g(R)
    return (
        "the dedicated observers make no measurable difference to the safety outcome "
        "in this plant either, whereas localisation roughly halves the drift-correction "
        "error and keeps most noise-matched poisoning from being forwarded. On the "
        "deployment question raised by the noise-signature result we keep the "
        "noise-signature group in the default: without it macro-F1 is slightly higher "
        "and the safety outcome is unchanged, but process masking rises from "
        f"{A['full']['process_masking_rate'][0]:.3f} to "
        f"{A['- noise-signature features']['process_masking_rate'][0]:.3f} and the "
        f"false-drift rate from {A['full']['false_drift_rate'][0]:.3f} to "
        f"{A['- noise-signature features']['false_drift_rate'][0]:.3f}; a process event "
        "masked as an instrument fault is calibrated away, which is the one error the "
        "differential response exists to prevent. Where the deployment's process "
        "events are rare and labels matter more, the reduced configuration is a "
        "defensible alternative.")


def aug_verdict(R):
    if not R.get("e17"):
        return ""
    x = R["e17"]["results"]
    n3 = x["noise x3"]
    gain = n3["PIRCAD noise-augmented"]["macro_f1"][0] - n3["PIRCAD"]["macro_f1"][0]
    nom = x["nominal"]
    dnom = nom["PIRCAD noise-augmented"]["macro_f1"][0] - nom["PIRCAD"]["macro_f1"][0]
    if gain > 0.03 and x["noise x3"]["Generic-feature RF noise-augmented"]["macro_f1"][0] > \
            n3["PIRCAD noise-augmented"]["macro_f1"][0]:
        head = ("Training coverage therefore explains most of PIRCAD's noise weakness, and "
                "also most of its apparent advantage in Section 6.3")
    elif gain > 0.03:
        head = ("The noise weakness is therefore largely a training-coverage problem "
                "rather than a property of the residual features")
    elif gain > 0.0:
        head = "Augmentation helps, but only partly"
    else:
        head = "Augmentation does not help at this level, so the weakness is not only one of training coverage"
    cost = (f", at a cost of {-dnom:.3f} in nominal macro-F1" if dnom < -0.005 else
            ", at no measurable cost in distribution")
    return head + cost + "."
