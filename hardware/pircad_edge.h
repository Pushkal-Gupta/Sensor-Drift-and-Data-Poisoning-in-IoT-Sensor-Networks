/* ============================================================================
 *  pircad_edge.h - edge-side PIRCAD kernel, free of Arduino dependencies.
 *
 *  Deliberately pure C++: sketch.ino includes it on the ESP32, and
 *  test_edge.cpp includes the same file on a host compiler, so the decision
 *  logic is regression-tested without flashing hardware.  Nothing here touches
 *  Serial, WiFi or any board API.
 *
 *  ---------------------------------------------------------------------------
 *  This is the two-channel edge form of the fog pipeline in ../pircad/.
 *  A DHT22 reports temperature AND relative humidity, and over short horizons
 *  in a given air mass the absolute humidity is conserved, so the two channels
 *  are tied together by the Magnus relation:
 *
 *          RH = 100 * e / e_s(T),     e_s(T) = 610.94 exp(17.625 T /(T+243.04))
 *
 *  That is a genuine physical constraint between two independently transduced
 *  quantities, and it is what makes the edge node a real miniature of the fog
 *  architecture rather than a degraded single-stream detector:
 *
 *      E1  physics model      -> expected response   (Newton's law of cooling)
 *      E2  residual + character                      (slope, monotonicity)
 *      E3  statistical trigger on the residual       (two-sided CUSUM)
 *      E4  cross-channel consistency                 (Magnus T-RH constraint)
 *      E5  root-cause disambiguation                 (rule form, no training)
 *      E7  adaptation governor                       (quarantine latch)
 *      E8  differential response                     (bounded recalibration)
 *
 *  What each stage separates:
 *      E4 separates an INSTRUMENT-side bias on the temperature channel from a
 *         GENUINE ambient change.  A real thermal excursion moves T and RH
 *         together and leaves the constraint satisfied; a bias on the
 *         temperature transducer alone breaks it.
 *      E2/E3/E5 then separate BENIGN AGEING from INJECTION within the
 *         instrument-side case, by rate and by reachability.
 *
 *  Known limit, asserted in the test suite so it cannot be quietly overclaimed:
 *  an injection that ramps at exactly the benign ageing rate is NOT separable
 *  here.  Both hypotheses predict the same thing on both channels.  Separating
 *  that case needs the redundancy of the multi-sensor fog node.
 * ==========================================================================*/
#ifndef PIRCAD_EDGE_H
#define PIRCAD_EDGE_H

#include <math.h>
#include <stdint.h>

namespace pircad {

// ----------------------------------------------------------------- verdicts
enum Verdict : uint8_t { V_CLEAN = 0, V_DRIFT = 1, V_POISON = 2, V_UNKNOWN = 3 };

inline const char* verdict_name(Verdict v) {
  switch (v) {
    case V_CLEAN:  return "CLEAN";
    case V_DRIFT:  return "DRIFT";
    case V_POISON: return "POISON";
    default:       return "UNKNOWN";
  }
}

// ------------------------------------------------------- psychrometrics (E4)
// Saturation vapour pressure over water, Magnus form, in pascal.
inline float magnus_es(float t_c) {
  return 610.94f * expf(17.625f * t_c / (t_c + 243.04f));
}
// d e_s / dT, in Pa per kelvin.
inline float magnus_des_dt(float t_c) {
  const float d = t_c + 243.04f;
  return magnus_es(t_c) * 17.625f * 243.04f / (d * d);
}
// Partial pressure of water vapour implied by a (T, RH) pair, in pascal.
inline float vapour_pressure(float t_c, float rh_pct) {
  return 0.01f * rh_pct * magnus_es(t_c);
}
// Inverse of magnus_es: the temperature at which e_s takes the given value.
inline float magnus_inv_t(float es_pa) {
  if (es_pa < 1.0f) es_pa = 1.0f;
  const float z = logf(es_pa / 610.94f);
  const float d = 17.625f - z;
  return (fabsf(d) < 1e-4f) ? 0.0f : 243.04f * z / d;
}

// --------------------------------------------------------------- parameters
static const int PIRCAD_MAX_WINDOW = 32;

struct EdgeParams {
  // --- E1 thermal model: y_hat(k) = y_hat(k-1) + (dt/tau)(T_amb - y_hat(k-1))
  float dt_s              = 2.0f;    // sampling interval, s
  float tau_s             = 30.0f;   // thermal time constant of the sensor mass
  float amb_alpha         = 0.02f;   // long-window EMA for the ambient estimate
  float model_gain        = 0.50f;   // state re-anchoring gain on the residual

  // --- E2 residual statistics (learned on a clean window, then held)
  float sigma_alpha       = 0.05f;   // noise-scale learning rate
  int   scale_learn_until = 40;      // sample after which the residual scales
                                     // are HELD.  An adaptive scale is exactly
                                     // the handle a slow injection uses to
                                     // normalise itself away: let it keep
                                     // learning and the threshold chases the
                                     // very bias it is supposed to catch.
  float sigma_floor       = 0.05f;   // degC, resolution floor of the transducer
  float bias_sigma_floor  = 0.08f;   // degC, floor for the cross-channel estimate
  int   rate_window       = 16;      // samples used for the slope estimate
  int   warmup            = 12;      // samples before any verdict but CLEAN

  // --- E3 trigger
  float cusum_k           = 0.15f;   // slack, degC
  float cusum_h           = 1.20f;   // decision interval, degC

  // --- E4 cross-channel consistency
  float vapour_alpha      = 0.02f;   // learning rate for the vapour-pressure
                                     // estimate; updated only while CLEAN

  // --- E5 decision rules
  float kappa             = 3.0f;    // reachability / consistency margin, in sigma
  float drift_max_rate    = 0.030f;  // degC/sample: datasheet ageing bound
  float rate_tol          = 1.25f;   // tolerance multiple on the ageing bound
  int   drift_min_persist = 18;      // near-monotonic samples before DRIFT.
                                     // MUST exceed rate_window: until the slope
                                     // window lies wholly inside the event the
                                     // rate test cannot see the true rate, and
                                     // a premature DRIFT verdict would start
                                     // correcting an injection.

  // --- E7 adaptation governor
  int   poison_hold       = 12;      // samples of enforced quarantine

  // --- E8 bounded recalibration
  float compensating_dead = 0.15f;   // degC.  While a correction larger than
                                     // this is in force the channel stays
                                     // classified as drifting: its bias reads
                                     // zero BECAUSE the correction is working,
                                     // which is evidence the correction works,
                                     // not evidence the sensor recovered.
  float recal_gain        = 0.25f;   // fraction of the standing bias applied
  float recal_rate_limit  = 0.02f;   // degC per sample
  float recal_total_cap   = 3.0f;    // degC, authority before exhaustion

  // --- configuration switches (used for the baseline comparison)
  bool  use_physics_model = true;    // false -> rolling mean of the own signal
  bool  use_consistency   = true;    // E4
  bool  use_reachability  = true;
  bool  use_rate_envelope = true;
  bool  use_poison_latch  = true;    // E7
  bool  use_bounded_recal = true;    // E8
  int   mean_window       = 12;      // rolling-mean span when the model is off
};

inline EdgeParams preset_proposed() { return EdgeParams(); }

// The edge preview as first drafted: physics residual, CUSUM and reachability,
// but no cross-channel constraint, no ageing envelope, no quarantine latch and
// an unbounded correction.
inline EdgeParams preset_draft() {
  EdgeParams p;
  p.use_consistency   = false;
  p.use_rate_envelope = false;
  p.use_poison_latch  = false;
  p.use_bounded_recal = false;
  return p;
}

// Generation-1 / prior-art behaviour: the reference is the signal's own recent
// mean, there is one response to every alarm, and it absorbs without limit.
inline EdgeParams preset_baseline() {
  EdgeParams p;
  p.use_physics_model = false;
  p.use_consistency   = false;
  p.use_reachability  = false;
  p.use_rate_envelope = false;
  p.use_poison_latch  = false;
  p.use_bounded_recal = false;
  return p;
}

// -------------------------------------------------------------------- state
struct EdgeState {
  float y_hat;              // thermal model state (degC)
  float t_amb;              // ambient estimate (degC)
  float e_est;              // vapour-pressure estimate (Pa)
  float sigma;              // residual scale (degC)
  float bias_sigma;         // scale of the cross-channel bias estimate (degC)
  float calib;              // standing calibration correction (degC)
  float c_pos, c_neg;       // CUSUM accumulators
  float last_corr;          // previous corrected temperature
  int   mono_run;           // consecutive same-sign bias samples
  int   last_sign;
  int   n;                  // samples seen
  int   hold;               // remaining quarantine samples
  bool  frozen;             // adaptation inhibited this sample
  bool  bootstrapped;
  float bias_buf[PIRCAD_MAX_WINDOW];  // ring of the cross-channel bias estimate
  int   bias_n, bias_i;
  float mean_buf[PIRCAD_MAX_WINDOW];  // ring for the baseline rolling mean
  int   mean_n, mean_i;
  int   calib_saturated;    // times the correction hit its cap
};

// ------------------------------------------------------------------- output
struct EdgeOut {
  float residual;      // thermal model residual, degC
  float rmax;          // largest physically reachable change this interval
  float cusum;         // trigger statistic
  float bias;          // temperature bias implied by the humidity channel, degC
  float slope;         // rate of change of that bias, degC/sample
  float slope_limit;   // rate above which benign ageing is excluded
  float calib;         // standing correction now in force
  float corrected;     // the temperature the node would forward
  Verdict verdict;
  bool  frozen;        // adaptation inhibited
  bool  triggered;     // the cheap stage fired
};

// ----------------------------------------------------------------- internals
inline void edge_reset(EdgeState& s, const EdgeParams& p) {
  s.y_hat = 0; s.t_amb = 0; s.e_est = 0;
  s.sigma = p.sigma_floor * 4.0f; s.bias_sigma = p.bias_sigma_floor * 2.0f;
  s.calib = 0; s.c_pos = 0; s.c_neg = 0; s.last_corr = 0;
  s.mono_run = 0; s.last_sign = 0; s.n = 0; s.hold = 0;
  s.frozen = false; s.bootstrapped = false;
  s.bias_n = 0; s.bias_i = 0; s.mean_n = 0; s.mean_i = 0;
  s.calib_saturated = 0;
  for (int i = 0; i < PIRCAD_MAX_WINDOW; i++) { s.bias_buf[i] = 0; s.mean_buf[i] = 0; }
}

inline int clamp_window(int w) {
  if (w < 4) return 4;
  if (w > PIRCAD_MAX_WINDOW) return PIRCAD_MAX_WINDOW;
  return w;
}

// Least-squares slope of a ring buffer, in units per sample.
inline float ring_slope(const float* buf, int idx, int count, int cap) {
  if (count < 4) return 0.0f;
  float sx = 0, sy = 0, sxx = 0, sxy = 0;
  for (int j = 0; j < count; j++) {
    const int pos = (idx - count + j + 2 * cap) % cap;
    const float x = (float)j, y = buf[pos];
    sx += x; sy += y; sxx += x * x; sxy += x * y;
  }
  const float nf = (float)count;
  const float den = nf * sxx - sx * sx;
  if (fabsf(den) < 1e-6f) return 0.0f;
  return (nf * sxy - sx * sy) / den;
}

inline float ring_mean(const float* buf, int idx, int count, int cap) {
  if (count <= 0) return 0.0f;
  float acc = 0;
  for (int j = 0; j < count; j++) acc += buf[(idx - count + j + 2 * cap) % cap];
  return acc / (float)count;
}

/* --------------------------------------------------------------------------
 * One sample through the edge pipeline.
 *   t_raw  : temperature as reported by the transducer, before correction
 *   rh_raw : relative humidity as reported, %
 * ------------------------------------------------------------------------ */
inline EdgeOut edge_step(EdgeState& s, const EdgeParams& p,
                         float t_raw, float rh_raw) {
  EdgeOut o;
  const int win  = clamp_window(p.rate_window);
  const int mwin = clamp_window(p.mean_window);

  // --- E8 (input side): apply the standing correction ----------------------
  const float t_corr = t_raw - s.calib;
  o.corrected = t_corr;
  o.calib = s.calib;
  o.slope_limit = p.drift_max_rate * p.rate_tol;

  if (!s.bootstrapped) {
    s.y_hat = t_corr; s.t_amb = t_corr; s.last_corr = t_corr;
    s.e_est = vapour_pressure(t_corr, rh_raw);
    s.bootstrapped = true; s.n = 1;
    s.mean_buf[s.mean_i] = t_corr; s.mean_i = (s.mean_i + 1) % mwin;
    if (s.mean_n < mwin) s.mean_n++;
    o.residual = 0; o.rmax = 0; o.cusum = 0; o.bias = 0; o.slope = 0;
    o.verdict = V_CLEAN; o.frozen = false; o.triggered = false;
    return o;
  }
  s.n++;

  // --- E1: expected response ----------------------------------------------
  float t_pred;
  if (p.use_physics_model) {
    // Newton's law of cooling, driven by a long-window ambient estimate.
    s.t_amb += p.amb_alpha * (t_corr - s.t_amb);
    t_pred = s.y_hat + (p.dt_s / p.tau_s) * (s.t_amb - s.y_hat);
  } else {
    // Baseline: the reference is the signal's own recent mean (Generation 1).
    t_pred = (s.mean_n > 0) ? ring_mean(s.mean_buf, s.mean_i, s.mean_n, mwin) : t_corr;
  }

  // --- E2: residual --------------------------------------------------------
  o.residual = t_corr - t_pred;

  /* --- E4: cross-channel consistency ------------------------------------
   * The humidity channel is transduced independently of the temperature
   * channel, and the two are tied by the Magnus relation at constant vapour
   * pressure.  Expressing the inconsistency as the temperature bias that
   * would explain it gives a quantity in degC that is directly comparable
   * with the ageing envelope -- and, unlike the thermal residual, it is NOT
   * absorbed by the model state, so its slope really is the rate at which the
   * temperature channel is departing from physics.
   *
   *   dRH/dT < 0, so a positive temperature bias makes the predicted RH too
   *   low and leaves a positive humidity residual; the implied bias is
   *   therefore positive, as it should be.                                */
  if (p.use_consistency) {
    // Temperature at which the measured humidity and the stored vapour
    // pressure would be consistent.  Solved exactly rather than linearised:
    // a first-order expansion about the BIASED temperature under-reads the
    // bias by ~20% at a few degC, which is enough to hide a ramp just over
    // the ageing envelope.
    const float rh = (rh_raw > 1.0f) ? rh_raw : 1.0f;
    const float t_from_rh = magnus_inv_t(100.0f * s.e_est / rh);
    o.bias = t_corr - t_from_rh;                              // degC
  } else {
    o.bias = 0.0f;
  }

  // Buffer the departure INCLUDING whatever the correction has already taken
  // up.  Buffering the bare bias would let a correction that is tracking an
  // injection flatten the slope and so erase the evidence that it is one.
  s.bias_buf[s.bias_i] = o.bias + s.calib;
  s.bias_i = (s.bias_i + 1) % win;
  if (s.bias_n < win) s.bias_n++;
  o.slope = ring_slope(s.bias_buf, s.bias_i, s.bias_n, win);
  const bool window_full = (s.bias_n >= win);

  // --- E3: two-sided CUSUM on the thermal residual -------------------------
  s.c_pos = fmaxf(0.0f, s.c_pos + o.residual - p.cusum_k);
  s.c_neg = fmaxf(0.0f, s.c_neg - o.residual - p.cusum_k);
  o.cusum = fmaxf(s.c_pos, s.c_neg);

  // Reachability: the largest change physics permits in one interval.
  o.rmax = (p.dt_s / p.tau_s) * fabsf(s.t_amb - s.last_corr) + p.kappa * s.sigma;
  const float observed_change = fabsf(t_corr - s.last_corr);
  const bool unreachable = p.use_reachability && (observed_change > o.rmax);

  // Cross-channel inconsistency beyond its own noise scale.
  const bool inconsistent = p.use_consistency
                          && (fabsf(o.bias) > p.kappa * s.bias_sigma);
  // Departing faster than the transducer can legitimately age.
  const bool rate_illegal = p.use_rate_envelope && window_full
                          && (fabsf(o.slope) > o.slope_limit);

  o.triggered = (o.cusum >= p.cusum_h) || unreachable || inconsistent || rate_illegal;

  // Sign persistence of the implied bias: the smoothness signature of ageing.
  const int sgn = (o.bias > 0.5f * p.bias_sigma_floor)
                - (o.bias < -0.5f * p.bias_sigma_floor);
  s.mono_run = (sgn != 0 && sgn == s.last_sign) ? s.mono_run + 1 : (sgn != 0 ? 1 : 0);
  s.last_sign = sgn;

  /* --- E7: a channel under quarantine is not re-litigated ----------------
   * Without this latch a held injection stops looking "unreachable" after its
   * first sample, falls into the benign branch a few samples later, and is
   * absorbed by the very correction that exists to serve honest ageing.   */
  if (p.use_poison_latch && s.hold > 0) {
    if (o.triggered) s.hold = p.poison_hold;   // still misbehaving: re-arm
    else             s.hold--;                 // counting down a clean run
    s.frozen = true;
    o.verdict = V_POISON; o.frozen = true;
    s.last_corr = t_corr;                      // state is NOT advanced
    o.calib = s.calib;
    return o;
  }

  // --- E5: root-cause disambiguation --------------------------------------
  if (s.n <= p.warmup || !o.triggered) {
    // A channel under active compensation keeps its standing classification.
    // Only poisoning may displace it, and poisoning arrives through the
    // triggered path above.
    const bool compensating = p.use_bounded_recal
                            && (fabsf(s.calib) > p.compensating_dead);
    o.verdict = compensating ? V_DRIFT : V_CLEAN;
    s.frozen = false;
    // scales and the vapour-pressure reference are learned only while nothing
    // is suspect -- an adaptive scale is the handle a slow injection uses to
    // normalise itself away
    if (s.n <= p.scale_learn_until && o.verdict == V_CLEAN) {
      s.sigma += p.sigma_alpha * (fabsf(o.residual) - s.sigma);
      if (s.sigma < p.sigma_floor) s.sigma = p.sigma_floor;
      s.bias_sigma += p.sigma_alpha * (fabsf(o.bias) - s.bias_sigma);
      if (s.bias_sigma < p.bias_sigma_floor) s.bias_sigma = p.bias_sigma_floor;
    }
    if (p.use_consistency && o.verdict == V_CLEAN) {
      const float e_obs = vapour_pressure(t_corr, rh_raw);
      s.e_est += p.vapour_alpha * (e_obs - s.e_est);
    }
  } else if (unreachable) {
    // No physical process could move the sensed quantity this far this fast.
    o.verdict = V_POISON;
  } else if (rate_illegal) {
    // Drift-shaped, but growing faster than the transducer can legitimately
    // age: outside the datasheet envelope, so not benign ageing.
    o.verdict = V_POISON;
  } else if (s.mono_run >= p.drift_min_persist && window_full) {
    // Smooth, one-directional, inside the envelope, and corroborated by the
    // humidity channel as an instrument-side bias rather than a real change.
    o.verdict = V_DRIFT;
  } else {
    o.verdict = V_UNKNOWN;
  }

  // --- E8: differential response ------------------------------------------
  switch (o.verdict) {
    case V_POISON:
      s.frozen = true;
      if (p.use_poison_latch) s.hold = p.poison_hold;
      s.c_pos = s.c_neg = 0;
      break;

    case V_DRIFT: {
      s.frozen = false;
      // correct towards the bias the humidity channel implies, or towards the
      // thermal residual when the consistency stage is disabled
      const float target = p.use_consistency ? o.bias : o.residual;
      float step = p.recal_gain * target;
      if (p.use_bounded_recal) {
        if (step >  p.recal_rate_limit) step =  p.recal_rate_limit;
        if (step < -p.recal_rate_limit) step = -p.recal_rate_limit;
        float next = s.calib + step;
        if (next >  p.recal_total_cap) { next =  p.recal_total_cap; s.calib_saturated++; }
        if (next < -p.recal_total_cap) { next = -p.recal_total_cap; s.calib_saturated++; }
        s.calib = next;
      } else {
        s.calib += step;                      // prior-art behaviour: no limit
      }
      s.c_pos = s.c_neg = 0;
      break;
    }

    case V_UNKNOWN:
      s.frozen = true;                        // hold, escalate, do not learn
      s.c_pos *= 0.5f; s.c_neg *= 0.5f;
      break;

    default:
      break;
  }

  // --- E1/E7: the model state advances only on data we accept --------------
  if (!s.frozen) s.y_hat = t_pred + p.model_gain * o.residual;

  s.mean_buf[s.mean_i] = t_corr; s.mean_i = (s.mean_i + 1) % mwin;
  if (s.mean_n < mwin) s.mean_n++;
  s.last_corr = t_corr;

  o.calib  = s.calib;
  o.frozen = s.frozen;
  return o;
}

// =========================== injection layer ================================
// Not part of the node's function: this is the experiment apparatus that
// produces labelled drift and poisoning so the node can be scored.  It touches
// the TEMPERATURE channel only -- the humidity channel keeps reporting the real
// air, which is precisely why the constraint between them is informative.

enum Inject : uint8_t {
  INJ_CLEAN = 0, INJ_DRIFT_LIN, INJ_DRIFT_EXP,
  INJ_POISON_STEP, INJ_POISON_SPIKE, INJ_POISON_STEALTH, INJ_POISON_MATCHED,
  INJ_COUNT
};

inline const char* inject_name(Inject m) {
  static const char* N[] = { "CLEAN", "DRIFT_LIN", "DRIFT_EXP", "POISON_STEP",
                             "POISON_SPIKE", "POISON_STEALTH", "POISON_MATCHED" };
  return N[m];
}

// Ground-truth class of each injection.
inline const char* inject_truth(Inject m) {
  static const char* T[] = { "CLEAN", "DRIFT", "DRIFT", "POISON",
                             "POISON", "POISON", "POISON" };
  return T[m];
}

struct InjectParams {
  float drift_lin_rate   = 0.020f;  // degC/sample, inside the ageing envelope
  float drift_exp_tau    = 100.0f;  // samples.  Ageing is modelled as an
                                    // exponential SETTLING towards a final
                                    // offset, which is the realistic shape and,
                                    // unlike runaway growth, keeps every
                                    // per-sample increment (cap/tau at most)
                                    // inside the ageing envelope for the whole
                                    // of its life
  float drift_exp_cap    = 2.5f;    // degC, final offset
  float poison_step_mag  = 6.0f;    // degC, held
  float poison_spike_mag = 9.0f;    // degC, isolated
  uint8_t spike_every    = 6;       // samples
  float stealth_rate     = 0.045f;  // degC/sample, drift-shaped but over bound
  float matched_rate     = 0.020f;  // degC/sample, EXACTLY the benign rate
};

struct InjectState { uint32_t k; float accum; };

inline void inject_reset(InjectState& s) { s.k = 0; s.accum = 0.0f; }

inline float inject_step(InjectState& s, const InjectParams& p, Inject m,
                         float clean, float& offset_out) {
  float off = 0.0f;
  switch (m) {
    case INJ_CLEAN: break;
    case INJ_DRIFT_LIN:      s.accum += p.drift_lin_rate; off = s.accum; break;
    case INJ_DRIFT_EXP:
      s.accum = p.drift_exp_cap * (1.0f - expf(-(float)s.k / p.drift_exp_tau));
      off = s.accum; break;
    case INJ_POISON_STEP:    off = p.poison_step_mag; break;
    case INJ_POISON_SPIKE:   off = (s.k % p.spike_every == 0) ? p.poison_spike_mag : 0.0f; break;
    case INJ_POISON_STEALTH: s.accum += p.stealth_rate; off = s.accum; break;
    case INJ_POISON_MATCHED: s.accum += p.matched_rate; off = s.accum; break;
    default: break;
  }
  s.k++;
  offset_out = off;
  return clean + off;
}

}  // namespace pircad
#endif  // PIRCAD_EDGE_H
