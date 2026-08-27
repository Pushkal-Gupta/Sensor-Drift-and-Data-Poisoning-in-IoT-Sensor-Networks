# PIRCAD edge node — Wokwi prototype

ESP32 DevKit V1 + DHT22 node for the fog-assisted root-cause disambiguation
work. It samples the sensor, injects labelled drift and poisoning into the
reported temperature, decides *why* the stream deviated, and acts differently on
each cause — recalibrate benign ageing, quarantine an injection, hold anything
ambiguous — then publishes to the fog process over MQTT.

The parent repository holds the multi-sensor fog pipeline (`../pircad/`) and the
patent material (`../docs/`). This directory is the hardware tier.

## Run it in Wokwi

1. <https://wokwi.com/projects/new/esp32> → replace the sketch with `sketch.ino`.
2. Add a second tab named `pircad_edge.h` and paste that file in — the sketch
   includes it and will not compile without it.
3. Replace `diagram.json` with this one (board is `wokwi-esp32-devkit-v1`).
4. Add a `libraries.txt` tab with the two lines from this repo.
5. ▶ — Wi-Fi joins `Wokwi-GUEST`, MQTT connects to `broker.hivemq.com:1883`.

Wi-Fi and MQTT are optional. With no broker the node still runs and prints
everything to the serial monitor; only the publishing is skipped.

There is deliberately no `wokwi.toml`. That file configures the Wokwi CLI and
VS Code extension, which simulate a **pre-compiled** binary and ignore the web
IDE flow above. Making it work would need `arduino-cli`, plus the sketch moved
into a folder named after itself (Arduino requires the main `.ino` to match its
parent directory, and this one lives in `hardware/`). If you ever want that
route:

```bash
arduino-cli core install esp32:esp32
arduino-cli compile --fqbn esp32:esp32:esp32doit-devkit-v1 --output-dir build .
```

then point `firmware` at `build/sketch.ino.bin` and `elf` at
`build/sketch.ino.elf`. Until someone needs CI builds, the web IDE is the
supported path and a config pointing at binaries that do not exist is worse
than none.

## Verify it without hardware

```bash
./run_checks.sh
```

Four things, none of which need a board:

| check | what it proves |
|---|---|
| `ci/compile_check.cpp` | `sketch.ino` type-checks under `-Wall -Wextra` against stubbed board APIs |
| `test_edge.cpp` | the decision kernel behaves correctly on every injection mode |
| `ci/check_diagram.py` | every pin the firmware drives is actually wired in `diagram.json` |
| `fog_bridge.py --replay` | the node's stream runs end-to-end through the fog pipeline |

## Circuit

| DHT22 pin | ESP32 DevKit V1 | note |
|---|---|---|
| VCC | `3V3` | |
| GND | `GND.1` | |
| SDA (data) | `D4` (GPIO4) | 10 kΩ pull-up to VCC |
| NC | — | |

| Indicator | ESP32 pin | Meaning |
|---|---|---|
| green LED | `D25` (GPIO25) | CLEAN — forwarding normally |
| yellow LED | `D26` (GPIO26) | SENSOR DRIFT — correcting, still forwarding |
| red LED | `D27` (GPIO27) | POISON or UNKNOWN — quarantined, adaptation frozen |

Each LED sits behind a 220 Ω resistor to `GND.2`. The three LEDs *are* the
differential response: the same deviation size lights a different lamp and
produces a different data-path behaviour depending on the cause found.

DevKit V1 uses `D`-prefixed pin names in `diagram.json` (`D4`, `D25`, `3V3`,
`GND.1`, `TX0`/`RX0`) — this differs from the DevKit-C v4 board.

## How it decides

Both channels of the DHT22 are used. Over short horizons the absolute humidity
of a given air mass is conserved, so temperature and relative humidity are tied
together by the Magnus relation:

```
RH = 100 · e / e_s(T),     e_s(T) = 610.94 · exp(17.625 T / (T + 243.04))
```

That is a real physical constraint between two independently transduced
quantities, and it is what makes this node a miniature of the fog architecture
rather than a single-stream detector:

```
   DHT22 ──┬── temperature ──► standing correction ──► Newton thermal model ──► residual ──► CUSUM
           │                                                     │
           └── humidity ──────► Magnus inversion ──► implied temperature bias
                                                                 │
                          reachability + ageing-envelope + persistence tests
                                                                 │
                     ┌───────────────────────┬───────────────────┴─────────┐
                  DRIFT                   POISON                        UNKNOWN
           bounded recalibration      quarantine, freeze              hold, escalate
           keep forwarding            adaptation, red LED             no adaptation
```

The humidity channel separates an **instrument-side bias** from a **genuine
thermal change**: a real temperature change moves both channels together and
leaves the constraint satisfied, while a bias on the temperature transducer
alone breaks it. Reachability and the ageing envelope then separate **benign
ageing** from **injection** within the instrument-side case.

## MQTT topics

| Topic | Content | Consumer |
|---|---|---|
| `pircad/node1/data` | `{node,seq,ms,temp,hum}` — corrected value, **only when the node decided to forward it** | fog process (blind) |
| `pircad/node1/truth` | `{seq,mode,gt,offset,temp_clean,temp_reported}` | scorer only |
| `pircad/node1/edge` | `{seq,residual,bias,slope,cusum,calib,verdict,frozen,forwarded}` | analysis |
| `pircad/node1/cmd` | single char: `c d e s p t m a r` | remote injection control |

Ground truth is deliberately on a separate topic so the fog pipeline is never
shown a label. `fog_bridge.py` keeps the two apart in code as well.

## Injection modes

The injection layer touches the **temperature channel only** — the humidity
channel keeps reporting the real air, which is exactly why the constraint
between them is informative.

| Key | Mode | Truth | Shape |
|---|---|---|---|
| `c` | CLEAN | CLEAN | none |
| `d` | DRIFT_LIN | DRIFT | +0.020 °C/sample, inside the ageing envelope |
| `e` | DRIFT_EXP | DRIFT | exponential settling toward +2.5 °C |
| `s` | POISON_STEP | POISON | +6 °C, held |
| `p` | POISON_SPIKE | POISON | +9 °C every 6th sample |
| `t` | POISON_STEALTH | POISON | +0.045 °C/sample — drift-shaped, above the envelope |
| `m` | POISON_MATCHED | POISON | +0.020 °C/sample — **exactly** the benign drift rate |
| `a` | AUTO | — | cycles all modes, 120 s each (default) |

`m` exists to be failed. See *Known limit* below.

## Running the fog tier

```bash
# live, against the node (needs: pip install paho-mqtt)
python3 fog_bridge.py --host broker.hivemq.com

# offline, no broker and no hardware
./test_edge --csv > runs.csv
python3 fog_bridge.py --replay runs.csv --config proposed
```

`fog_bridge.py` imports the residual engine, trigger, adaptation governor,
calibration manager and response controller from `../pircad/` rather than
reimplementing them. Only the root-cause *rules* are specialised: the repo's
`RuleRootCauseClassifier` is weighted for the redundant multi-sensor case and
mislabels a held injection when given a single node's evidence, so
`SingleNodeRule` in that file states the two-channel rules explicitly.

## Simulating a genuine thermal change

Moving only the Wokwi temperature slider is *not* a physically consistent
change — it is an inconsistent one, and the node will correctly flag it. To
simulate real warming or cooling, move both sliders together along a line of
constant vapour pressure (1340 Pa, the 24 °C / 45 %RH reference):

| Temperature | Humidity |
|---|---|
| 18 °C | 65 % |
| 20 °C | 57 % |
| 22 °C | 51 % |
| **24 °C** | **45 %** |
| 26 °C | 40 % |
| 28 °C | 36 % |
| 30 °C | 32 % |

Do that and the verdict stays CLEAN throughout — the physics model absorbs a
real change. Move temperature alone and you are, by definition, presenting a
temperature reading the humidity channel contradicts.

## Known limit

An injection that ramps at **exactly** the benign ageing rate (`m`,
POISON_MATCHED) is **not** separable at this node, and the test suite asserts
that it is not, so the limitation cannot regress into an unearned claim. Both
hypotheses predict the same thing on both channels: a smooth, one-directional
departure inside the ageing envelope. Separating that case needs the redundancy
of the multi-sensor fog node in `../pircad/`, where several instruments
constrained by independent physical relations disagree in a cause-specific
pattern.

## Files

```
pircad_edge.h    the decision kernel - pure C++, no Arduino dependencies
sketch.ino       firmware: sensor, LEDs, serial, MQTT.  I/O only
diagram.json     circuit netlist
libraries.txt    Wokwi library manifest
test_edge.cpp    host regression test for the kernel
fog_bridge.py    fog tier: live MQTT or offline replay, using ../pircad/
run_checks.sh    everything verifiable without hardware
ci/              board-API stubs and the netlist validator
dataset/         generated dataset and its data card
```
