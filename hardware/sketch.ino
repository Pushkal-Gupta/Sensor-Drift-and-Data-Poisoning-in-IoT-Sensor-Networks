/* ============================================================================
 *  PIRCAD edge node - ESP32 DevKit V1 + DHT22
 *  Physics-Informed Root-Cause disambiguation of sensor drift vs data poisoning
 *
 *  Team: Shridhar M (23BCT0201), Vanka Kaushik, Pushkal Gupta
 *
 *  What this firmware does
 *  ----------------------
 *   1. samples the DHT22 (temperature AND humidity) at a fixed interval
 *   2. injects labelled drift / poisoning into the reported TEMPERATURE only,
 *      carrying the ground truth on a separate topic so the fog process can be
 *      scored blind
 *   3. runs the edge PIRCAD kernel (pircad_edge.h) over both channels and
 *      applies the differential response: recalibrate benign ageing, quarantine
 *      an injection, hold anything ambiguous
 *   4. drives three status LEDs so the differential response is visible on the
 *      board itself, and mirrors everything as CSV on the serial monitor
 *   5. publishes over Wi-Fi / MQTT to the fog process
 *
 *  All decision logic lives in pircad_edge.h, which has no Arduino
 *  dependencies and is regression-tested on the host by test_edge.cpp.  This
 *  file is only I/O: sensor, LEDs, serial, network.
 *
 *  Serial commands (type in the monitor, then Enter):
 *    c CLEAN        d DRIFT linear     e DRIFT exponential-settling
 *    s POISON step  p POISON spike     t POISON stealth ramp
 *    m POISON matched to the benign drift rate  (the documented hard case)
 *    a AUTO schedule (default)   r reset   ? help
 * ==========================================================================*/

#include <Arduino.h>
#include <WiFi.h>
#include <PubSubClient.h>
#include <DHTesp.h>

#include "pircad_edge.h"
using namespace pircad;

// ------------------------------- hardware ----------------------------------
#define PIN_DHT        4     // GPIO4  (board pin D4), 10k pull-up to 3V3
#define PIN_LED_CLEAN  25    // GPIO25 (D25) green  - forwarding normally
#define PIN_LED_DRIFT  26    // GPIO26 (D26) yellow - drift, correcting, still forwarding
#define PIN_LED_POISON 27    // GPIO27 (D27) red    - quarantined, adaptation frozen

// ------------------------------- network -----------------------------------
static const char* WIFI_SSID    = "Wokwi-GUEST";
static const char* WIFI_PASS    = "";
static const char* MQTT_HOST    = "broker.hivemq.com";   // public test broker
static const uint16_t MQTT_PORT = 1883;
static const char* NODE_ID      = "pircad-node1";
static const char* TOPIC_DATA   = "pircad/node1/data";   // what the fog sees
static const char* TOPIC_TRUTH  = "pircad/node1/truth";  // ground truth, separate
static const char* TOPIC_EDGE   = "pircad/node1/edge";   // edge verdict
static const char* TOPIC_CMD    = "pircad/node1/cmd";    // remote control

// ------------------------------- timing ------------------------------------
static const uint32_t SAMPLE_MS     = 2000;   // DHT22 needs >= 2 s between reads
static const uint32_t AUTO_PHASE_MS = 120000; // each injection phase in AUTO mode
static const uint32_t WIFI_WAIT_MS  = 15000;

// --------------------------------- state -----------------------------------
DHTesp        dht;
WiFiClient    net;
PubSubClient  mqtt(net);

EdgeParams  edgeParams;      // preset_proposed() by default
EdgeState   edgeState;
InjectParams injParams;
InjectState  injState;

Inject   mode        = INJ_CLEAN;
bool     autoMode    = true;
uint32_t phaseStart  = 0;
uint32_t lastSample  = 0;
uint32_t seq         = 0;

// scoreboard, printed periodically
uint32_t nTotal = 0, nCorrect = 0, nPoisonForwarded = 0;

// ------------------------------- helpers -----------------------------------
static bool modeFromChar(char c, Inject& out) {
  switch (c) {
    case 'c': out = INJ_CLEAN;           return true;
    case 'd': out = INJ_DRIFT_LIN;       return true;
    case 'e': out = INJ_DRIFT_EXP;       return true;
    case 's': out = INJ_POISON_STEP;     return true;
    case 'p': out = INJ_POISON_SPIKE;    return true;
    case 't': out = INJ_POISON_STEALTH;  return true;
    case 'm': out = INJ_POISON_MATCHED;  return true;
    default:                             return false;
  }
}

static void setMode(Inject m, const char* source) {
  mode = m;
  inject_reset(injState);
  phaseStart = millis();
  Serial.printf("# [%s] injection -> %s (ground truth = %s)\n",
                source, inject_name(m), inject_truth(m));
}

static void printHelp() {
  Serial.println(F("# cmds: c=CLEAN  d=DRIFT_LIN  e=DRIFT_EXP  s=POISON_STEP"));
  Serial.println(F("#       p=POISON_SPIKE  t=POISON_STEALTH  m=POISON_MATCHED"));
  Serial.println(F("#       a=AUTO  r=reset  ?=help"));
}

static void resetAll() {
  edge_reset(edgeState, edgeParams);
  nTotal = nCorrect = nPoisonForwarded = 0;
  seq = 0;
  setMode(INJ_CLEAN, "reset");
}

// The differential response, made visible on the board.
static void showVerdict(Verdict v) {
  digitalWrite(PIN_LED_CLEAN,  v == V_CLEAN);
  digitalWrite(PIN_LED_DRIFT,  v == V_DRIFT);
  // a held or ambiguous sample both mean "not forwarded"; the red LED is the
  // quarantine indication
  digitalWrite(PIN_LED_POISON, v == V_POISON || v == V_UNKNOWN);
}

// E8: only CLEAN and DRIFT are forwarded. POISON is quarantined, UNKNOWN is
// held pending escalation. This is the whole point of the architecture.
static bool shouldForward(Verdict v) { return v == V_CLEAN || v == V_DRIFT; }

// ------------------------------- MQTT --------------------------------------
static void onMqtt(char* topic, byte* payload, unsigned int len) {
  if (len == 0) return;
  const char c = (char)payload[0];
  Inject m;
  if      (c == 'a') { autoMode = true; phaseStart = millis();
                       Serial.println(F("# [mqtt] AUTO schedule on")); }
  else if (c == 'r') { resetAll(); }
  else if (modeFromChar(c, m)) { autoMode = false; setMode(m, "mqtt"); }
}

static void ensureMqtt() {
  if (WiFi.status() != WL_CONNECTED || mqtt.connected()) return;
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

static void publishSample(uint32_t now, float tReported, float rh,
                          float tClean, float offset, const EdgeOut& o) {
  if (!mqtt.connected()) return;
  char buf[288];

  // The data topic carries only what a real node would send, and only when the
  // node has decided the sample is fit to forward.
  if (shouldForward(o.verdict)) {
    snprintf(buf, sizeof(buf),
      "{\"node\":\"%s\",\"seq\":%lu,\"ms\":%lu,\"temp\":%.2f,\"hum\":%.1f}",
      NODE_ID, (unsigned long)seq, (unsigned long)now, o.corrected, rh);
    mqtt.publish(TOPIC_DATA, buf);
  }

  // Ground truth on its own topic, so the fog pipeline is never shown labels.
  snprintf(buf, sizeof(buf),
    "{\"seq\":%lu,\"mode\":\"%s\",\"gt\":\"%s\",\"offset\":%.3f,"
    "\"temp_clean\":%.2f,\"temp_reported\":%.2f}",
    (unsigned long)seq, inject_name(mode), inject_truth(mode),
    offset, tClean, tReported);
  mqtt.publish(TOPIC_TRUTH, buf);

  snprintf(buf, sizeof(buf),
    "{\"seq\":%lu,\"residual\":%.3f,\"bias\":%.3f,\"slope\":%.4f,"
    "\"cusum\":%.3f,\"calib\":%.3f,\"verdict\":\"%s\",\"frozen\":%d,"
    "\"forwarded\":%d}",
    (unsigned long)seq, o.residual, o.bias, o.slope, o.cusum, o.calib,
    verdict_name(o.verdict), o.frozen ? 1 : 0, shouldForward(o.verdict) ? 1 : 0);
  mqtt.publish(TOPIC_EDGE, buf);
}

// --------------------------------- setup -----------------------------------
void setup() {
  Serial.begin(115200);
  delay(300);
  Serial.println(F("# PIRCAD edge node - ESP32 + DHT22"));
  Serial.println(F("# root-cause disambiguation: drift vs data poisoning"));

  pinMode(PIN_LED_CLEAN,  OUTPUT);
  pinMode(PIN_LED_DRIFT,  OUTPUT);
  pinMode(PIN_LED_POISON, OUTPUT);
  showVerdict(V_CLEAN);

  dht.setup(PIN_DHT, DHTesp::DHT22);

  edgeParams = preset_proposed();
  edgeParams.dt_s = SAMPLE_MS / 1000.0f;
  edge_reset(edgeState, edgeParams);
  inject_reset(injState);

  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  Serial.print(F("# WiFi"));
  const uint32_t t0 = millis();
  while (WiFi.status() != WL_CONNECTED && millis() - t0 < WIFI_WAIT_MS) {
    delay(250); Serial.print('.');
  }
  if (WiFi.status() == WL_CONNECTED)
    Serial.printf(" connected %s\n", WiFi.localIP().toString().c_str());
  else
    Serial.println(F(" not connected - running serial-only"));

  mqtt.setServer(MQTT_HOST, MQTT_PORT);
  mqtt.setCallback(onMqtt);
  mqtt.setBufferSize(512);

  printHelp();
  Serial.println(F("seq,ms,mode,truth,temp_clean,offset,temp_reported,hum,"
                   "residual,bias,slope,cusum,calib,verdict,frozen,forwarded"));
  setMode(INJ_CLEAN, "boot");
  lastSample = millis();
}

// --------------------------------- loop ------------------------------------
static void handleSerial() {
  while (Serial.available()) {
    const char c = Serial.read();
    if (c == '\r' || c == '\n' || c == ' ') continue;
    Inject m;
    if      (c == 'a') { autoMode = true; phaseStart = millis();
                         Serial.println(F("# AUTO schedule on")); }
    else if (c == 'r') { resetAll(); }
    else if (c == '?') { printHelp(); }
    else if (modeFromChar(c, m)) { autoMode = false; setMode(m, "serial"); }
  }
}

static void handleAutoSchedule() {
  if (!autoMode || millis() - phaseStart < AUTO_PHASE_MS) return;
  static const Inject SCHEDULE[] = {
    INJ_CLEAN, INJ_DRIFT_LIN,      INJ_CLEAN, INJ_POISON_STEP,
    INJ_CLEAN, INJ_POISON_SPIKE,   INJ_CLEAN, INJ_DRIFT_EXP,
    INJ_CLEAN, INJ_POISON_STEALTH, INJ_CLEAN, INJ_POISON_MATCHED };
  static uint8_t idx = 0;
  idx = (idx + 1) % (sizeof(SCHEDULE) / sizeof(SCHEDULE[0]));
  setMode(SCHEDULE[idx], "auto");
}

void loop() {
  handleSerial();
  ensureMqtt();
  mqtt.loop();
  handleAutoSchedule();

  if (millis() - lastSample < SAMPLE_MS) return;
  // If the loop stalled (a long Wi-Fi or broker retry), resynchronise instead
  // of firing a burst of catch-up samples the DHT22 cannot service.
  lastSample = (millis() - lastSample > 2 * SAMPLE_MS) ? millis()
                                                       : lastSample + SAMPLE_MS;

  // --- step 1: acquire both channels ---------------------------------------
  TempAndHumidity th = dht.getTempAndHumidity();
  if (dht.getStatus() != DHTesp::ERROR_NONE || isnan(th.temperature)) {
    Serial.printf("# DHT error: %s\n", dht.getStatusString());
    return;
  }
  const float tClean = th.temperature;
  const float rh     = th.humidity;

  // --- step 2: injection layer (experiment apparatus, temperature only) ----
  float offset = 0.0f;
  const float tReported = inject_step(injState, injParams, mode, tClean, offset);

  // --- step 3: the edge PIRCAD kernel --------------------------------------
  const EdgeOut o = edge_step(edgeState, edgeParams, tReported, rh);

  // --- step 4: differential response ---------------------------------------
  showVerdict(o.verdict);
  const bool forwarded = shouldForward(o.verdict);

  seq++;
  const uint32_t now = millis();

  // running score against the ground truth we happen to know here
  nTotal++;
  if (strcmp(verdict_name(o.verdict), inject_truth(mode)) == 0) nCorrect++;
  if (strcmp(inject_truth(mode), "POISON") == 0 && forwarded) nPoisonForwarded++;

  // --- step 5: report -------------------------------------------------------
  Serial.printf("%lu,%lu,%s,%s,%.2f,%.2f,%.2f,%.1f,"
                "%.3f,%.3f,%.4f,%.3f,%.3f,%s,%d,%d\n",
                (unsigned long)seq, (unsigned long)now,
                inject_name(mode), inject_truth(mode),
                tClean, offset, tReported, rh,
                o.residual, o.bias, o.slope, o.cusum, o.calib,
                verdict_name(o.verdict), o.frozen ? 1 : 0, forwarded ? 1 : 0);

  if (seq % 30 == 0)
    Serial.printf("# score: root-cause %lu/%lu = %.1f%%   "
                  "poisoned samples forwarded: %lu\n",
                  (unsigned long)nCorrect, (unsigned long)nTotal,
                  100.0 * nCorrect / (nTotal ? nTotal : 1),
                  (unsigned long)nPoisonForwarded);

  publishSample(now, tReported, rh, tClean, offset, o);
}
