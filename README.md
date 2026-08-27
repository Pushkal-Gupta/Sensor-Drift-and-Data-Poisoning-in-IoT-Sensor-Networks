# PIRRD Edge Node — Wokwi simulation (Assessment-4)

ESP32 DevKit V1 + DHT22 node from Section 5 of the proposal. Samples the sensor,
injects labelled drift / poisoning into the reported value, and publishes over
MQTT to the fog process. Serial monitor is the UI (no OLED/LEDs, per Section 5.3).

## Run
1. <https://wokwi.com/projects/new/esp32> → replace the sketch with `sketch.ino`.
2. Replace `diagram.json` with this one (board type is `wokwi-esp32-devkit-v1`).
3. Add a `libraries.txt` tab with the two lines from this repo.
4. ▶ — Wi-Fi joins `Wokwi-GUEST`, MQTT connects to `broker.hivemq.com:1883`.

## Wiring
| DHT22 pin | ESP32 DevKit V1 pin |
|---|---|
| VCC | `3V3` |
| GND | `GND.1` |
| SDA (data) | `D4` (GPIO4), with 10 kΩ pull-up to VCC |
| NC | — |

DevKit V1 uses `D`-prefixed pin names in diagram.json (`D4`, `3V3`, `GND.1`,
`TX0`/`RX0`) — this differs from the DevKit-C v4 board.

## MQTT topics
| Topic | Content | Consumer |
|---|---|---|
| `pirrd/node1/data` | `{node,seq,ms,temp,hum}` — the *reported* value only | fog process (blind) |
| `pirrd/node1/truth` | `{seq,mode,gt,offset,temp_clean}` — ground truth | fog scorer |
| `pirrd/node1/edge` | `{seq,residual,rmax,cusum,verdict,frozen}` — edge preview | optional |
| `pirrd/node1/cmd` | single char: `c d e s p t a r` | remote injection control |

Truth is deliberately on a separate topic so the fog pipeline never sees labels.

## Injection modes
| Key | Mode | Ground truth | Shape |
|---|---|---|---|
| c | CLEAN | CLEAN | none |
| d | DRIFT_LIN | DRIFT | +0.02 °C/sample linear ramp |
| e | DRIFT_EXP | DRIFT | exponential ramp, capped +3 °C |
| s | POISON_STEP | POISON | +6 °C held |
| p | POISON_SPIKE | POISON | +9 °C every 6th sample |
| t | POISON_STEALTH | POISON | +0.045 °C/sample — drift-shaped, exceeds sensor bound |
| a | AUTO | — | cycles all modes, 40 s each (default) |

## Fog-side quick listener
```python
import paho.mqtt.client as mqtt, json
def on_msg(c, u, m): print(m.topic, json.loads(m.payload))
c = mqtt.Client(); c.on_message = on_msg
c.connect("broker.hivemq.com", 1883); c.subscribe("pirrd/node1/#"); c.loop_forever()
```

Change the DHT22 value in Wokwi (click the sensor) to produce a *legitimate*
transient — the edge preview should stay CLEAN because the physics model
absorbs it, which is the residual-domain advantage of Section 4.1.
