# Revision notes — journal version (Machine Learning with Applications)

## Title

**Old:** Physics-Informed Fog-Assisted Root-Cause Disambiguation of Sensor Drift and
Data Poisoning in IoT Sensor Networks with Differential Response Control

**New:** Drift or Deception? Physics-Anchored Root-Cause Learning for
Poisoning-Resilient Self-Calibration in IoT Sensor Networks

The new title states the ML problem (cause-aware learning that protects
self-calibrating elements from poisoning) rather than the system architecture.
MLWA reviewers read for the learning contribution.

## What changed in the paper

| Area | Before | After |
|---|---|---|
| Framing | A fog-layer system with a pipeline of elements | Cause-aware learning under adversarial distribution shift, with a formal problem, an attacker-capability hierarchy (A0–A5), and two propositions that state what can and cannot be separated |
| Baselines | The authors' own Generation-1 pipeline only | A supervised multi-scale 1D-CNN, a generic-feature random forest, Generation-1, four unsupervised detectors (Isolation Forest, PCA-SPE, autoencoder, Kalman χ²), and three self-calibrating estimators tuned on benign drift |
| Statistics | 4 seeds, mean ± SD | 30 held-out seeds (180 episodes), 95% bootstrap CIs, paired Wilcoxon tests with Holm correction, rank-biserial effect sizes |
| Attacker | One (stale surrogate) | Sweep A0 → A5, including an on-path attacker that is provably indistinguishable from drift, and a coordinated multi-sensor forgery |
| Generalisation | none | Eleven distribution-shift conditions with no retraining |
| Real data | none (all simulated) | Intel Berkeley Research Lab: real T/RH streams from 51 motes, semi-synthetic injections, mote-disjoint train/test |
| Effect metrics | label accuracy, containment | adds forwarded error ("harm") and useful-data rate, so a system cannot look safe by withholding everything |
| Removed | quantum escalation appendix, patent language, pin tables, MQTT topic tables, "expected vs actual" targets | the edge node is kept as a short section |

## Defects found by the stronger evaluation (fixed, and reported in the paper)

1. **Polarity shortcut.** The original training bank had every drift and attack
   ramp going upward and every process event heating. The classifier learned the
   sign as a shortcut: a downward drift was called NORMAL, then PROCESS, and
   never corrected. A first attempt to fix this by drawing the sign at random
   happened to make all six training drift episodes downward, and the model then
   learned "upward drift is normal". **Fix:** every training episode appears in
   both polarities, and the evaluation bank alternates polarity by seed.
   (`experiments/v2_common.py::_sign`, `train_specs`)
2. **Quarantine lock-in.** One transient POISON verdict at the onset of genuine
   drift quarantined the channel permanently, because by design a drift verdict
   could never release it. Worse, the action still followed the latest verdict,
   so the node forwarded *uncorrected* data while the governor believed the
   channel was quarantined. **Fix:**
   - containment follows the governor's standing state, not the latest verdict;
   - an evidence-gated re-attribution path (freeze hold served, plus 200
     consecutive drift verdicts) moves a quarantined channel to DRIFTING.

   Re-attribution trades security for availability, so it is a policy switch
   (`FogConfig.allow_reattribution`). Once fix 3 was in, experiment 9 showed
   that latched quarantine has the same drift-correction error and far fewer
   spurious safety trips, so **latched is the default** and re-attribution is
   reported as an alternative. (`pircad/governor.py`, `pircad/fognode.py`)

3. **Uncertain verdicts latched quarantine.** Below the confidence threshold a
   verdict is meant to be escalated, with "no irreversible action on ambiguous
   samples". With no escalation backend configured, the fallback *forced the
   confidence to 0.999* and acted on the guess. A 0.3-confidence POISON verdict
   therefore latched quarantine. On real data this turned a classifier that called
   drift correctly most of the time into one that reported "poisoning" for most of
   each drift episode. **Fix:** an unresolved POISON verdict holds that sample
   (not forwarded, no adaptation, no calibration), but the governor records it
   as UNKNOWN, so it cannot latch. A confident POISON verdict still quarantines
   immediately. (`pircad/fognode.py`, escalation block)

All three were found by the new experiments, not by the original 4-seed
benchmark, and each one made the original numbers look better or worse than the
method deserved. The paper reports them in Section 7.2.

## Engineering changes that affect reproducibility, not results

- `pircad/fastforest.py`: single-sample random-forest inference that is
  numerically identical to scikit-learn (asserted in tests) and about 200× faster.
  scikit-learn spends ~16 ms of joblib overhead per single-sample call, which was
  97% of a fog-node run.
- One BLAS/OpenMP thread per worker process (`experiments/v2_common.py`).
- `pircad/scenarios.py`: with default settings, the extended scenario factory
  reproduces the original bank sample for sample (asserted in tests).

## Reproduce

```
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt torch python-docx
python experiments/run_all_v2.py      # fits models, runs exps 6-12, draws figures
python paper/build_manuscript.py      # writes paper/PIRCAD_MLWA_manuscript.docx
python tests/test_pircad.py && python tests/test_v2.py
```
The Intel Lab data (34 MB) is downloaded on first use into
`data/external/intel_lab/` (gitignored).
