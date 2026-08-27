/* ============================================================================
 *  PIRRD Edge Node - Physics-Informed Residual Root-cause Disambiguation
 *  Assessment-4: Fog-Assisted Disambiguation of Sensor Drift vs Data Poisoning
 *
 *  Team: Shridhar M (23BCT0201), Vanka Kaushik, Pushkal Gupta
 *
 *  Role of this firmware (Section 5.1 / 7.1 of the proposal):
 *    - sample the DHT22 at a fixed interval and timestamp each reading
 *    - inject synthetic DRIFT (linear / exponential ramp) and POISONING
 *      (step, spike, slow stealth ramp) directly into the reported value,
 *      with the ground-truth label carried alongside so the fog process can
 *      score itself
 *    - publish every sample over Wi-Fi / MQTT to the fog process
 *    - mirror everything on the serial monitor (the only UI at this stage)
 *
 *  The fog process (Python, Section 9) is the authoritative PIRRD pipeline.
 *  This sketch additionally runs a compact "edge preview" of the same
 *  physics model + residual CUSUM + reachability test so the mechanism is
 *  visible on the serial monitor inside Wokwi without a broker attached.
 *  The preview never touches the reported value; it is diagnostic only.
 *
 *  Serial commands (type in the monitor, then Enter):
 *    c  CLEAN          d  DRIFT linear      e  DRIFT exponential
 *    s  POISON step    p  POISON spike      t  POISON stealth ramp
 *    a  AUTO schedule (default) - cycles through every mode on a timer
 *    r  reset injection state and baseline
 *    ?  print this help
 * ==========================================================================*/

#include <Arduino.h>
#include <WiFi.h>
#include <PubSubClient.h>
#include <DHTesp.h>

// ------------------------------- hardware ----------------------------------
#define PIN_DHT         4          // GPIO4 (board pin D4), 10k pull-up to 3V3

// ------------------------------- network -----------------------------------
static const char* WIFI_SSID   = "Wokwi-GUEST";
static const char* WIFI_PASS   = "";
static const char* MQTT_HOST   = "broker.hivemq.com";   // public test broker
static const uint16_t MQTT_PORT = 1883;
static const char* NODE_ID     = "pirrd-node1";
static const char* TOPIC_DATA  = "pirrd/node1/data";   // what the fog sees
static const char* TOPIC_TRUTH = "pirrd/node1/truth";  // ground truth, separate
static const char* TOPIC_EDGE  = "pirrd/node1/edge";   // edge-preview verdict
static const char* TOPIC_CMD   = "pirrd/node1/cmd";    // remote injection control

// ------------------------------- timing ------------------------------------
static const uint32_t SAMPLE_MS      = 2000;   // DHT22 needs >= 2 s between reads
static const uint32_t AUTO_PHASE_MS  = 40000;  // each injection phase in AUTO mode
static const float    DT_S           = SAMPLE_MS / 1000.0f;

// ------------------------------- injection ---------------------------------
enum Inject : uint8_t { CLEAN, DRIFT_LIN, DRIFT_EXP, POISON_STEP, POISON_SPIKE, POISON_STEALTH };
static const char* INJECT_NAME[] = { "CLEAN", "DRIFT_LIN", "DRIFT_EXP",
                                     "POISON_STEP", "POISON_SPIKE", "POISON_STEALTH" };
// Ground-truth class for each injection (Section 4.1 three-class outcome)
static const char* GT_CLASS[]    = { "CLEAN", "DRIFT", "DRIFT", "POISON", "POISON", "POISON" };

static const float DRIFT_LIN_RATE   = 0.020f;  // degC per sample  (slow, benign)
static const float DRIFT_EXP_RATE   = 0.035f;  // growth factor per sample
static const float DRIFT_EXP_CAP    = 3.0f;    // degC
static const float POISON_STEP_MAG  = 6.0f;    // degC, held
static const float POISON_SPIKE_MAG = 9.0f;    // degC, single sample every N
static const uint8_t SPIKE_EVERY    = 6;       // samples
static const float STEALTH_RATE     = 0.045f;  // degC per sample - shaped to look
                                               // like drift but exceeds sensor
                                               // degradation bounds

// ------------------------------- edge preview ------------------------------
// Physics model, Section 9.1:   y_hat(t) = y_hat(t-1) + (dt/tau)*(Tamb - y_hat(t-1)) + b
static const float TAU_S            = 30.0f;   // thermal time constant of DHT22 mass
static const float AMB_ALPHA        = 0.02f;   // long-window EMA for Tamb (several tau)
static const float BASE_ALPHA       = 0.01f;   // slow baseline b(t) learning rate
// CUSUM on the residual, Section 9.2 step 4
static const float CUSUM_K          = 0.15f;   // slack
static const float CUSUM_H          = 1.20f;   // decision threshold
// Reachability, Section 9.1:   Rmax = (dt/tau)*|Tamb - y(t-1)| + kappa*sigma
static const float KAPPA            = 3.0f;
static const float SIGMA_ALPHA      = 0.05f;
static const float DRIFT_MAX_RATE   = 0.030f;  // degC/sample - sensor's known degradation bound
static const int   DRIFT_MIN_PERSIST = 8;      // near-monotonic residual samples

// --------------------------------- state -----------------------------------
DHTesp dht;
WiFiClient net;
PubSubClient mqtt(net);

Inject   mode         = CLEAN;
bool     autoMode     = true;
uint32_t phaseStart   = 0;
uint32_t lastSample   = 0;
uint32_t seq          = 0;
uint32_t injSamples   = 0;      // samples since the current injection began
float    driftAccum   = 0.0f;   // accumulated ramp offset

// edge preview state
float yHat = NAN, tAmb = NAN, base = 0.0f, sigma = 0.3f;
float cPos = 0, cNeg = 0;
float lastReported = NAN;
int   monoRun = 0;  int lastResSign = 0;
bool  frozen = false;           // freeze-learning interlock (Section 4.1)

// ------------------------------- helpers -----------------------------------
static const char* modeChar(char c, Inject& out) {
  switch (c) {
    case 'c': out = CLEAN;          return "CLEAN";
    case 'd': out = DRIFT_LIN;      return "DRIFT_LIN";
    case 'e': out = DRIFT_EXP;      return "DRIFT_EXP";
    case 's': out = POISON_STEP;    return "POISON_STEP";
    case 'p': out = POISON_SPIKE;   return "POISON_SPIKE";
    case 't': out = POISON_STEALTH; return "POISON_STEALTH";
    default:  return nullptr;
  }
}

void setMode(Inject m, const char* source) {
  mode = m; injSamples = 0; driftAccum = 0.0f; phaseStart = millis();
  Serial.printf("# [%s] injection -> %s (gt=%s)\n", source, INJECT_NAME[m], GT_CLASS[m]);
}

void printHelp() {
  Serial.println(F("# cmds: c=CLEAN d=DRIFT_LIN e=DRIFT_EXP s=POISON_STEP p=POISON_SPIKE"));
  Serial.println(F("#       t=POISON_STEALTH a=AUTO r=reset ?=help"));
}

void resetState() {
  yHat = NAN; tAmb = NAN; base = 0; sigma = 0.3f; cPos = cNeg = 0;
  lastReported = NAN; monoRun = 0; lastResSign = 0; frozen = false;
  setMode(CLEAN, "reset");
}

// --------------------------- injection (Section 7.1) -----------------------
// Returns the value the node REPORTS. The clean reading is never modified in
// place; the injection is a separable additive layer so the truth topic can
// carry the offset exactly.
float inject(float clean, float& offsetOut) {
  float off = 0.0f;
  switch (mode) {
    case CLEAN:
      break;
    case DRIFT_LIN:
      driftAccum += DRIFT_LIN_RATE;                 off = driftAccum; break;
    case DRIFT_EXP:
      driftAccum = fminf(DRIFT_EXP_CAP,
                   (driftAccum < 0.05f ? 0.05f : driftAccum) * (1.0f + DRIFT_EXP_RATE));
      off = driftAccum; break;
    case POISON_STEP:
      off = POISON_STEP_MAG; break;
    case POISON_SPIKE:
      off = (injSamples % SPIKE_EVERY == 0) ? POISON_SPIKE_MAG : 0.0f; break;
    case POISON_STEALTH:
      driftAccum += STEALTH_RATE;                   off = driftAccum; break;
  }
  injSamples++;
  offsetOut = off;
  return clean + off;
}

// ----------------------- edge preview of PIRRD (Section 9.2) ---------------
// Returns one of "CLEAN","DRIFT","POISON","UNKNOWN" and fills residual/Rmax.
const char* edgePirrd(float y, float& residual, float& rmax, float& cusum) {
  // bootstrap
  if (isnan(yHat)) { yHat = y; tAmb = y; lastReported = y; residual = 0; rmax = 0; cusum = 0; return "CLEAN"; }

  // step 2: physics prediction. Tamb is a long-window EMA of the stream (9.1)
  tAmb = tAmb + AMB_ALPHA * (y - tAmb);
  float yPred = yHat + (DT_S / TAU_S) * (tAmb - yHat) + base;

  // step 3: residual
  residual = y - yPred;

  // step 4: two-sided CUSUM on the residual
  cPos = fmaxf(0.0f, cPos + residual - CUSUM_K);
  cNeg = fmaxf(0.0f, cNeg - residual - CUSUM_K);
  cusum = fmaxf(cPos, cNeg);

  // reachability bound (9.1) against the previous reported value
  rmax = (DT_S / TAU_S) * fabsf(tAmb - lastReported) + KAPPA * sigma;
  float observedChange = fabsf(y - lastReported);

  // near-monotonic residual tracking for the drift signature
  int sgn = (residual > 0.02f) - (residual < -0.02f);
  monoRun = (sgn != 0 && sgn == lastResSign) ? monoRun + 1 : (sgn != 0 ? 1 : 0);
  lastResSign = sgn;

  const char* verdict;
  if (cusum < CUSUM_H) {
    // step 5: CLEAN - slow baseline update permitted, noise estimate updated
    verdict = "CLEAN";
    frozen = false;
    base  += BASE_ALPHA * residual;
    sigma  = sigma + SIGMA_ALPHA * (fabsf(residual) - sigma);
    if (sigma < 0.05f) sigma = 0.05f;
  } else if (observedChange > rmax) {
    // step 7a: physically unreachable in one interval -> POISON regardless
    verdict = "POISON";
    frozen = true;                               // freeze-learning interlock
    cPos = cNeg = 0;                             // re-arm after alert
  } else if (monoRun >= DRIFT_MIN_PERSIST &&
             fabsf(y - lastReported) <= DRIFT_MAX_RATE * 1.5f) {
    // step 7b: slow, near-monotonic, inside sensor degradation bound -> DRIFT
    verdict = "DRIFT";
    frozen = false;
    base += 0.25f * residual;                    // recalibrate, keep using data
    cPos = cNeg = 0;
  } else {
    // step 8: not confidently either -> UNKNOWN (escalate, do not learn)
    verdict = "UNKNOWN";
    frozen = true;
    cPos *= 0.5f; cNeg *= 0.5f;
  }

  // step 9 (data path only): state advance is skipped when frozen so a poisoned
  // sample can never move y_hat or b(t) - structurally unreachable, Section 9.3
  if (!frozen) yHat = yPred + 0.5f * residual;
  lastReported = y;
  return verdict;
}

// ------------------------------- MQTT --------------------------------------
void onMqtt(char* topic, byte* payload, unsigned int len) {
  if (len == 0) return;
  char c = (char)payload[0];
  Inject m;
  if (c == 'a') { autoMode = true;  Serial.println(F("# [mqtt] AUTO schedule on")); }
  else if (c == 'r') { resetState(); }
  else if (modeChar(c, m)) { autoMode = false; setMode(m, "mqtt"); }
}

void ensureMqtt() {
  if (WiFi.status() != WL_CONNECTED) return;
  if (mqtt.connected()) return;
  static uint32_t lastTry = 0;
  if (millis() - lastTry < 3000) return;
  lastTry = millis();
  Serial.print(F("# MQTT connecting... "));
  if (mqtt.connect(NODE_ID)) {
    Serial.println(F("ok"));
    mqtt.subscribe(TOPIC_CMD);
  } else {
    Serial.printf("failed rc=%d\n", mqtt.state());
  }
}

// --------------------------------- setup -----------------------------------
void setup() {
  Serial.begin(115200);
  delay(300);
  Serial.println(F("# PIRRD edge node - ESP32 + DHT22"));
  Serial.println(F("# Assessment-4: drift vs poisoning root-cause disambiguation"));

  dht.setup(PIN_DHT, DHTesp::DHT22);

  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  Serial.print(F("# WiFi"));
  for (int i = 0; i < 60 && WiFi.status() != WL_CONNECTED; i++) { delay(250); Serial.print('.'); }
  if (WiFi.status() == WL_CONNECTED) Serial.printf(" connected %s\n", WiFi.localIP().toString().c_str());
  else Serial.println(F(" not connected - running serial-only"));

  mqtt.setServer(MQTT_HOST, MQTT_PORT);
  mqtt.setCallback(onMqtt);
  mqtt.setBufferSize(512);

  printHelp();
  Serial.println(F("seq,ms,mode,gt,temp_clean,offset,temp_reported,hum,residual,rmax,cusum,edge_verdict,frozen"));
  setMode(CLEAN, "boot");
  lastSample = millis();
}

// --------------------------------- loop ------------------------------------
void loop() {
  // serial command handling
  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\r' || c == '\n' || c == ' ') continue;
    Inject m;
    if (c == 'a')      { autoMode = true; phaseStart = millis(); Serial.println(F("# AUTO schedule on")); }
    else if (c == 'r') { resetState(); }
    else if (c == '?') { printHelp(); }
    else if (modeChar(c, m)) { autoMode = false; setMode(m, "serial"); }
  }

  ensureMqtt();
  mqtt.loop();

  // AUTO schedule: CLEAN -> DRIFT_LIN -> CLEAN -> POISON_STEP -> CLEAN -> POISON_SPIKE
  //                -> CLEAN -> DRIFT_EXP -> CLEAN -> POISON_STEALTH -> ...
  if (autoMode && millis() - phaseStart >= AUTO_PHASE_MS) {
    static const Inject SCHEDULE[] = { CLEAN, DRIFT_LIN, CLEAN, POISON_STEP, CLEAN,
                                       POISON_SPIKE, CLEAN, DRIFT_EXP, CLEAN, POISON_STEALTH };
    static uint8_t idx = 0;
    idx = (idx + 1) % (sizeof(SCHEDULE) / sizeof(SCHEDULE[0]));
    setMode(SCHEDULE[idx], "auto");
  }

  if (millis() - lastSample < SAMPLE_MS) return;
  lastSample += SAMPLE_MS;

  // step 1: acquire
  TempAndHumidity th = dht.getTempAndHumidity();
  if (dht.getStatus() != DHTesp::ERROR_NONE || isnan(th.temperature)) {
    Serial.printf("# DHT error: %s\n", dht.getStatusString());
    return;
  }
  float tClean = th.temperature;
  float hum    = th.humidity;

  // injection layer (what an ageing sensor / an attacker would do to the value)
  float offset;
  float tReported = inject(tClean, offset);

  // edge preview of the fog pipeline
  float residual, rmax, cusum;
  const char* verdict = edgePirrd(tReported, residual, rmax, cusum);

  seq++;
  uint32_t now = millis();

  // serial CSV (the node-side UI, Section 5.3)
  Serial.printf("%lu,%lu,%s,%s,%.2f,%.2f,%.2f,%.1f,%.3f,%.3f,%.3f,%s,%d\n",
                (unsigned long)seq, (unsigned long)now, INJECT_NAME[mode], GT_CLASS[mode],
                tClean, offset, tReported, hum, residual, rmax, cusum, verdict, frozen ? 1 : 0);

  // MQTT: data topic carries ONLY what a real node would send. Truth and edge
  // verdict go on separate topics so the fog process can be scored blind.
  if (mqtt.connected()) {
    char buf[256];
    snprintf(buf, sizeof(buf),
      "{\"node\":\"%s\",\"seq\":%lu,\"ms\":%lu,\"temp\":%.2f,\"hum\":%.1f}",
      NODE_ID, (unsigned long)seq, (unsigned long)now, tReported, hum);
    mqtt.publish(TOPIC_DATA, buf);

    snprintf(buf, sizeof(buf),
      "{\"seq\":%lu,\"mode\":\"%s\",\"gt\":\"%s\",\"offset\":%.3f,\"temp_clean\":%.2f}",
      (unsigned long)seq, INJECT_NAME[mode], GT_CLASS[mode], offset, tClean);
    mqtt.publish(TOPIC_TRUTH, buf);

    snprintf(buf, sizeof(buf),
      "{\"seq\":%lu,\"residual\":%.3f,\"rmax\":%.3f,\"cusum\":%.3f,\"verdict\":\"%s\",\"frozen\":%d}",
      (unsigned long)seq, residual, rmax, cusum, verdict, frozen ? 1 : 0);
    mqtt.publish(TOPIC_EDGE, buf);
  }
}
