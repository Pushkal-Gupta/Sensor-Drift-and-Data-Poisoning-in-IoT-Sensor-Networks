/* ============================================================================
 *  test_edge.cpp - host-side regression test for pircad_edge.h
 *
 *  Compiles the same header the ESP32 firmware uses, on the host, and drives
 *  it with synthetic streams for every injection mode.  This is how the edge
 *  decision logic is verified without flashing hardware, and how the baseline
 *  and the proposed method are compared on byte-identical input.
 *
 *      g++ -O2 -std=c++11 -o test_edge test_edge.cpp && ./test_edge
 *      ./test_edge --csv > edge_runs.csv        (per-sample trace for plotting)
 * ==========================================================================*/
#include "pircad_edge.h"
#include <cstdio>
#include <cstring>
#include <cstdlib>

using namespace pircad;

// Deterministic pseudo-noise so runs are reproducible across machines.
static uint32_t rng_state = 12345u;
static float noise(float sd) {
  rng_state = rng_state * 1664525u + 1013904223u;
  float u1 = ((rng_state >> 8) & 0xFFFFFF) / 16777216.0f;
  rng_state = rng_state * 1664525u + 1013904223u;
  float u2 = ((rng_state >> 8) & 0xFFFFFF) / 16777216.0f;
  if (u1 < 1e-7f) u1 = 1e-7f;
  return sd * sqrtf(-2.0f * logf(u1)) * cosf(6.2831853f * u2);
}

struct Score { int clean, drift, poison, unknown, n; float final_calib;
               float ref_departure; int detect_k; };

/* How far the system's own reference has moved away from the truth by the end
 * of the run -- the honest measure of how much of the injection it absorbed.
 * The correction offset alone understates it: a rolling-mean reference absorbs
 * by tracking the injected signal, without ever recording a correction. */
static float reference_departure(const EdgeState& s, const EdgeParams& p,
                                 float last_clean) {
  const float ref = p.use_physics_model
                  ? s.y_hat
                  : ring_mean(s.mean_buf, s.mean_i, s.mean_n,
                              clamp_window(p.mean_window));
  return (ref + s.calib) - last_clean;
}

/* Run one injection mode through one configuration.
 * `onset` is the sample at which the injection starts; everything before it is
 * clean and is used to let the model settle.                                */
static Score run(const EdgeParams& p, Inject mode, int n_samples, int onset,
                 float noise_sd, bool csv, const char* cfg_name) {
  EdgeState s; edge_reset(s, p);
  InjectState inj; inject_reset(inj);
  InjectParams ip;

  Score sc = {0,0,0,0,0,0.0f,0.0f,-1};
  float last_clean = 0.0f;
  const float ambient = 24.0f;
  // 45 %RH at 24 degC -> a fixed vapour pressure for the whole run
  static const float E_VAPOUR = 0.45f * 610.94f * expf(17.625f * 24.0f / (24.0f + 243.04f));

  for (int k = 0; k < n_samples; k++) {
    // A gentle real thermal excursion, so "clean" is not a flat line.
    float clean = ambient + 0.8f * sinf(k * 0.03f) + noise(noise_sd);
    // The air holds a fixed vapour pressure, so the humidity the sensor really
    // sees is set by the REAL temperature.  The injection touches only the
    // temperature channel, which is what makes the two channels disagree.
    float rh = 100.0f * E_VAPOUR / magnus_es(clean) + noise(noise_sd);
    last_clean = clean;
    float offset = 0.0f;
    float y = (k >= onset) ? inject_step(inj, ip, mode, clean, offset)
                           : (inject_reset(inj), clean);

    EdgeOut o = edge_step(s, p, y, rh);

    if (k >= onset) {
      sc.n++;
      switch (o.verdict) {
        case V_CLEAN:  sc.clean++;  break;
        case V_DRIFT:  sc.drift++;  break;
        case V_POISON: sc.poison++; break;
        default:       sc.unknown++;break;
      }
      const char* truth = inject_truth(mode);
      if (sc.detect_k < 0 && strcmp(verdict_name(o.verdict), truth) == 0
          && strcmp(truth, "CLEAN") != 0)
        sc.detect_k = k - onset;
    }
    if (csv)
      printf("%s,%s,%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%s,%s,%d\n",
             cfg_name, inject_name(mode), k, clean, offset, y, rh,
             o.residual, o.cusum, o.bias, o.slope, o.calib,
             verdict_name(o.verdict), inject_truth(mode), o.frozen ? 1 : 0);
  }
  sc.final_calib = s.calib;
  sc.ref_departure = reference_departure(s, p, last_clean);
  return sc;
}

static int failures = 0;
static void check(bool ok, const char* what) {
  printf("  %s  %s\n", ok ? "PASS " : "FAIL ", what);
  if (!ok) failures++;
}

/* Fraction of post-onset samples whose verdict equals the ground-truth class. */
static float correct_fraction(const Score& s, Inject m) {
  const char* t = inject_truth(m);
  int hit = 0;
  if      (!strcmp(t, "CLEAN"))  hit = s.clean;
  else if (!strcmp(t, "DRIFT"))  hit = s.drift;
  else                            hit = s.poison;
  return s.n ? (float)hit / (float)s.n : 0.0f;
}

int main(int argc, char** argv) {
  const bool csv = (argc > 1 && !strcmp(argv[1], "--csv"));
  const int  N = 220, ONSET = 60;
  const float NOISE = 0.05f;

  struct Cfg { const char* name; EdgeParams p; };
  Cfg cfgs[3] = { {"baseline", preset_baseline()},
                  {"draft",    preset_draft()},
                  {"proposed", preset_proposed()} };

  if (csv) {
    printf("config,mode,k,temp_clean,offset,temp_reported,hum,residual,cusum,bias,slope,calib,verdict,truth,frozen\n");
    for (int c = 0; c < 3; c++)
      for (int m = 0; m < INJ_COUNT; m++)
        run(cfgs[c].p, (Inject)m, N, ONSET, NOISE, true, cfgs[c].name);
    return 0;
  }

  printf("PIRCAD edge kernel - host regression test\n");
  printf("%d samples per run, injection from sample %d, sensor noise %.3f degC\n\n",
         N, ONSET, NOISE);

  // ------------------------------------------------- per-configuration table
  Score S[3][INJ_COUNT];
  printf("%-16s", "mode");
  for (int c = 0; c < 3; c++) printf("%22s", cfgs[c].name);
  printf("\n%s\n", "--------------------------------------------------------------------------------");
  for (int m = 0; m < INJ_COUNT; m++) {
    printf("%-16s", inject_name((Inject)m));
    for (int c = 0; c < 3; c++) {
      S[c][m] = run(cfgs[c].p, (Inject)m, N, ONSET, NOISE, false, cfgs[c].name);
      printf("%21.2f%%", 100.0f * correct_fraction(S[c][m], (Inject)m));
    }
    printf("\n");
  }

  printf("\nreference departure at end of run (degC) - how far the system's own\n"
         "reference has been pulled from the truth, i.e. how much of the\n"
         "injection it absorbed:\n");
  printf("%-16s", "mode");
  for (int c = 0; c < 3; c++) printf("%22s", cfgs[c].name);
  printf("\n%s\n", "--------------------------------------------------------------------------------");
  for (int m = 0; m < INJ_COUNT; m++) {
    printf("%-16s", inject_name((Inject)m));
    for (int c = 0; c < 3; c++) printf("%22.3f", S[c][m].ref_departure);
    printf("\n");
  }

  // ------------------------------------------------------------- assertions
  printf("\nassertions\n");
  const EdgeParams pp = preset_proposed();

  check(correct_fraction(S[2][INJ_CLEAN], INJ_CLEAN) > 0.95f,
        "proposed: a clean stream stays CLEAN (>95% of samples)");

  check(correct_fraction(S[2][INJ_DRIFT_LIN], INJ_DRIFT_LIN) > 0.60f,
        "proposed: linear ageing is identified as DRIFT");
  check(S[2][INJ_DRIFT_LIN].poison == 0,
        "proposed: benign ageing is never called POISON");

  check(correct_fraction(S[2][INJ_DRIFT_EXP], INJ_DRIFT_EXP) > 0.50f,
        "proposed: accelerating ageing inside the envelope is DRIFT");

  check(correct_fraction(S[2][INJ_POISON_STEP], INJ_POISON_STEP) > 0.90f,
        "proposed: a HELD step stays POISON (the latch defect)");
  check(S[2][INJ_POISON_STEP].drift == 0,
        "proposed: a held step is never re-labelled DRIFT");
  // The reference is not perfectly untouched: a few samples pass before the
  // latch engages.  What matters is the size of what gets through relative to
  // a system with no latch at all.
  check(fabsf(S[2][INJ_POISON_STEP].ref_departure) < 1.0f,
        "proposed: under 1 degC of the 6 degC step reaches the reference");
  check(fabsf(S[0][INJ_POISON_STEP].ref_departure)
        > 5.0f * fabsf(S[2][INJ_POISON_STEP].ref_departure),
        "proposed absorbs >5x less of the step than the baseline");

  check(correct_fraction(S[2][INJ_POISON_SPIKE], INJ_POISON_SPIKE) > 0.50f,
        "proposed: isolated spikes are POISON");

  check(correct_fraction(S[2][INJ_POISON_STEALTH], INJ_POISON_STEALTH) > 0.60f,
        "proposed: the stealth ramp is POISON (the envelope defect)");
  check(fabsf(S[2][INJ_POISON_STEALTH].ref_departure) < 1.0f,
        "proposed: under 1 degC of the stealth ramp reaches the reference");
  check(fabsf(S[0][INJ_POISON_STEALTH].ref_departure)
        > 5.0f * fabsf(S[2][INJ_POISON_STEALTH].ref_departure),
        "proposed absorbs >5x less of the stealth ramp than the baseline");

  // The two defects, demonstrated against the draft configuration.
  check(correct_fraction(S[1][INJ_POISON_STEP], INJ_POISON_STEP) < 0.30f,
        "draft: held step is largely missed  (defect 1 reproduced)");
  check(fabsf(S[0][INJ_POISON_STEP].ref_departure) > 3.0f,
        "baseline: absorbs most of the held step into its reference");
  check(fabsf(S[0][INJ_POISON_STEALTH].ref_departure) > 3.0f,
        "baseline: absorbs the stealth ramp entirely");
  check(correct_fraction(S[1][INJ_POISON_STEALTH], INJ_POISON_STEALTH) < 0.05f,
        "draft: stealth ramp is invisible    (defect 2 reproduced)");
  check(correct_fraction(S[2][INJ_POISON_STEP], INJ_POISON_STEP)
        > correct_fraction(S[1][INJ_POISON_STEP], INJ_POISON_STEP) + 0.5f,
        "proposed beats draft on the held step by >50 points");
  check(correct_fraction(S[2][INJ_POISON_STEALTH], INJ_POISON_STEALTH)
        > correct_fraction(S[1][INJ_POISON_STEALTH], INJ_POISON_STEALTH) + 0.5f,
        "proposed beats draft on the stealth ramp by >50 points");

  // Bounded recalibration must respect its cap.
  for (int m = 0; m < INJ_COUNT; m++)
    if (fabsf(S[2][m].final_calib) > pp.recal_total_cap + 1e-3f) {
      check(false, "proposed: correction exceeded its cap"); break;
    }
  check(true, "proposed: correction never exceeded the total cap");

  // The honest limitation: an injection at exactly the benign ageing rate is
  // NOT separable at a single node.  Asserted so it cannot regress silently
  // into an unearned claim.
  check(correct_fraction(S[2][INJ_POISON_MATCHED], INJ_POISON_MATCHED) < 0.30f,
        "proposed: a rate-matched injection is NOT separable single-channel "
        "(documented limitation, needs the multi-sensor fog node)");

  printf("\n%s (%d failure%s)\n", failures ? "FAILED" : "ALL CHECKS PASSED",
         failures, failures == 1 ? "" : "s");
  return failures ? 1 : 0;
}
