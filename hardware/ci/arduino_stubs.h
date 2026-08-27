/* Minimal stand-ins for the Arduino/ESP32 APIs used by sketch.ino.
 * They exist only so the firmware can be type-checked on a host compiler in
 * CI; none of them talk to hardware.  See ci/compile_check.cpp. */
#pragma once
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <string>

#define F(x) (x)
#define OUTPUT 1
#define HIGH 1
#define LOW 0
inline void pinMode(int, int) {}
inline void digitalWrite(int, int) {}
inline uint32_t millis() { return 0; }
inline void delay(uint32_t) {}

struct SerialStub {
  void begin(unsigned long) {}
  template <typename... A> void printf(const char* f, A... a) { (void)std::snprintf(nullptr, 0, f, a...); }
  void print(const char*) {}
  void print(char) {}
  void println(const char*) {}
  void println() {}
  int  available() { return 0; }
  int  read() { return -1; }
};
static SerialStub Serial;

struct IPAddr { std::string toString() const { return std::string("0.0.0.0"); } };
#define WL_CONNECTED 3
#define WIFI_STA 1
struct WiFiStub {
  void mode(int) {}
  void begin(const char*, const char*) {}
  int  status() { return WL_CONNECTED; }
  IPAddr localIP() { return IPAddr(); }
};
static WiFiStub WiFi;
struct WiFiClient {};

struct PubSubClient {
  PubSubClient(WiFiClient&) {}
  void setServer(const char*, uint16_t) {}
  void setCallback(void (*)(char*, uint8_t*, unsigned int)) {}
  void setBufferSize(uint16_t) {}
  bool connect(const char*) { return true; }
  bool connected() { return true; }
  int  state() { return 0; }
  void subscribe(const char*) {}
  void publish(const char*, const char*) {}
  void loop() {}
};
typedef uint8_t byte;

struct TempAndHumidity { float temperature; float humidity; };
struct DHTesp {
  enum DHT_MODEL_t { DHT22 };
  enum DHT_ERROR_t { ERROR_NONE };
  void setup(int, DHT_MODEL_t) {}
  TempAndHumidity getTempAndHumidity() { TempAndHumidity t; t.temperature = 24.0f; t.humidity = 45.0f; return t; }
  DHT_ERROR_t getStatus() { return ERROR_NONE; }
  const char* getStatusString() { return "OK"; }
};
