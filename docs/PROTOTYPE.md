# Prototype: implementation, modelling and results

The hardware tier of the PIRCAD work — an ESP32 + DHT22 node that decides *why*
a sensor stream deviated and acts differently on each cause. Everything below is
reproduced by `cd hardware && ./run_checks.sh`, which needs no board.

Companion documents: `ARCHITECTURE.md` (the fog pipeline), `PATENT_PACKAGE.md`
(claims and prior art), `hardware/README.md` (how to run it in Wokwi).

---

## 1. Circuit

### Netlist

| From | To | Via | Purpose |
|---|---|---|---|
| `dht:VCC` | `esp:3V3` | — | sensor supply, 3.3 V |
| `dht:GND` | `esp:GND.1` | — | sensor return |
| `dht:SDA` | `esp:D4` (GPIO4) | 10 kΩ pull-up to VCC | 1-wire data |
| `esp:D25` (GPIO25) | `ledClean:A` | 220 Ω | green — CLEAN, forwarding |
| `esp:D26` (GPIO26) | `ledDrift:A` | 220 Ω | yellow — DRIFT, correcting |
| `esp:D27` (GPIO27) | `ledPoison:A` | 220 Ω | red — POISON/UNKNOWN, quarantined |
| `led*:C` | `esp:GND.2` | — | LED return |
| `esp:TX0` / `esp:RX0` | serial monitor | — | console and command input |

Board: `wokwi-esp32-devkit-v1`. The DevKit V1 uses `D`-prefixed pin names in
`diagram.json`, unlike the DevKit-C v4.

### Power

Everything runs from the ESP32's on-board 3.3 V regulator. The DHT22 draws
about 1.5 mA while converting; each LED draws (3.3 − 2.0) / 220 ≈ 6 mA and at
most one is lit at a time, since the three verdicts are mutually exclusive. Peak
draw outside the Wi-Fi radio is therefore under 10 mA, well inside the
regulator's budget.

### Why these components

The DHT22 was not chosen only as a temperature source. It reports temperature
**and** relative humidity from two separate transducers, and those two
quantities are tied together by physics (§3.2). That gives a single-node
analogue of the cross-sensor redundancy the fog pipeline relies on — the
smallest sensor that makes the architecture demonstrable rather than merely
described.

The three LEDs are the differential response made physical: the same deviation
size lights a different lamp, and drives a different data-path behaviour,
according to the cause found.

### Validation

`hardware/ci/check_diagram.py` parses `diagram.json`, checks every connection
endpoint against a declared part, and cross-checks the `#define PIN_*` lines in
`sketch.ino` so the firmware cannot drive a pin that is not wired.

```
OK: 11 parts, 16 connections
    ESP32 pins wired: 3V3, D25, D26, D27, D4, GND.1, GND.2, RX0, TX0
    firmware pins   : PIN_DHT=GPIO4, PIN_LED_CLEAN=GPIO25,
                      PIN_LED_DRIFT=GPIO26, PIN_LED_POISON=GPIO27
```

---

## 2. Dataset

Full data card: `hardware/dataset/README.md`.

| | |
|---|---|
| File | `hardware/dataset/edge_dataset.csv` |
| Samples | 4 620 = 3 configurations × 7 injection modes × 220 samples |
| Interval | 2 s (DHT22 minimum) — 220 samples ≈ 7.3 min per episode |
| **Input features** | `temp_reported` (°C), `hum` (%RH) — nothing else reaches the decision path |
| **Target** | `truth` ∈ {CLEAN, DRIFT, POISON} |
| Ground truth (scoring only) | `temp_clean`, `offset` |
| Format | CSV, 15 columns |
| Reproducible | fixed-seed generator; byte-identical across machines |

### Split

| Samples | Role |
|---|---|
| 0 – 39 | commissioning: residual scales are learned, then **frozen** |
| 40 – 59 | clean hold-out, model settling |
| 60 – 219 | injection active; these are the samples scored |

Freezing the scales at sample 40 is not a detail. An adaptive threshold is
exactly the handle a slow injection uses to normalise itself away — during
development the drift case sat at 0.6 % recall for precisely this reason, with
the threshold chasing upward the very bias it was meant to catch.

### Class balance

| Class | Samples | Share |
|---|---|---|
| CLEAN | 660 | 14.3 % |
| DRIFT | 1 320 | 28.6 % |
| POISON | 2 640 | 57.1 % |

Deliberately attack-heavy — four of seven modes are attacks. Per-class recall is
reported rather than plain accuracy for that reason.

### Generation

The clean signal is ambient 24 °C with a slow ±0.8 °C excursion and 0.05 °C RMS
transducer noise. The air holds a fixed vapour pressure of 1 340 Pa, so humidity
is derived from the *true* temperature through the Magnus relation. The
injection is added to the **temperature channel only** — the humidity channel
keeps reporting the real air, which is what makes the constraint between them
informative.

Data is produced by the same `pircad_edge.h` kernel the ESP32 runs, driven on
the host by `test_edge.cpp`. There is no separate simulation model to drift out
of step with the firmware.

---

## 3. Mathematical model

No learned model is used at the node. The decision rules are closed-form
physical statements, which is what lets the node run with no training corpus and
makes each rule auditable. The fog tier (`../pircad/`) is where the supervised
model lives.

### 3.1 Thermal model (E1)

The sensor mass exchanges heat with its surroundings by Newton's law of cooling:

```
    dT/dt = (1/τ)(T_amb − T)
```

discretised at the sampling interval Δt, with the ambient estimated as a
long-window EMA of the stream itself:

```
    T̂(k)   = T̂(k−1) + (Δt/τ)( T̂_amb(k) − T̂(k−1) )
    T̂_amb(k) = T̂_amb(k−1) + α ( T_corr(k) − T̂_amb(k−1) )
    r(k)   = T_corr(k) − T̂(k)                                    (residual)
```

with τ = 30 s, Δt = 2 s, α = 0.02, and `T_corr = T_reported − c(k)` where `c` is
the standing calibration correction.

**Reachability.** The most the sensed temperature can legitimately move in one
interval:

```
    R_max(k) = (Δt/τ) | T̂_amb(k) − T_corr(k−1) | + κ σ,     κ = 3
```

A one-step change exceeding `R_max` is not something the process can do.

### 3.2 Cross-channel consistency (E4)

Over short horizons a given air mass conserves its vapour pressure `e`, so
temperature and relative humidity are tied by the Magnus relation:

```
    e_s(T) = 610.94 exp( 17.625 T / (T + 243.04) )        [Pa]
    RH     = 100 e / e_s(T)                               [%]
```

Inverting it gives the temperature the **humidity channel** implies, and the
difference is the temperature bias the two channels disagree about:

```
    T_RH(k) = e_s^{-1}( 100 ê / RH(k) )
            = 243.04 z / (17.625 − z),   z = ln( 100 ê / (RH · 610.94) )

    b(k) = T_corr(k) − T_RH(k)                            (implied bias, °C)
```

`ê` is estimated as an EMA of the observed vapour pressure and is frozen the
moment anything is suspect.

This is the single most important equation in the node. A **genuine** thermal
change moves both channels together and leaves `b ≈ 0`. A bias on the
temperature transducer alone drives `b` to the size of that bias. Sensitivity at
24 °C / 45 %RH is `dRH/dT ≈ −2.7 %RH per °C`, so a 1 °C bias produces a 2.7 %RH
inconsistency — far above the sensor's resolution.

The inversion is solved **exactly** rather than linearised: a first-order
expansion about the *biased* temperature under-reads the bias by roughly 20 % at
a few °C, which was enough, when measured, to hide a ramp sitting just above the
ageing envelope.

### 3.3 Ageing envelope (E5)

A transducer's specified degradation rate bounds what benign ageing can look
like. With `ρ` the least-squares slope of the implied bias over a 16-sample
window:

```
    ρ(k) = slope over window of [ b(i) + c(i) ]
    benign ageing  requires   |ρ| ≤ ρ_max · tol,    ρ_max = 0.030 °C/sample, tol = 1.25
```

The slope is taken on `b + c`, **including** whatever the correction has already
absorbed. Taking it on `b` alone lets a correction that is tracking an injection
flatten the slope and so erase the evidence that it is one — a failure that
appeared in testing before it was fixed.

### 3.4 Trigger (E3)

Two-sided CUSUM on the residual, with slack `κ_c = 0.15` °C and decision
interval `h = 1.20` °C:

```
    C⁺(k) = max(0, C⁺(k−1) + r(k) − κ_c)
    C⁻(k) = max(0, C⁻(k−1) − r(k) − κ_c)
    fire  when  max(C⁺, C⁻) ≥ h
```

The stage is engaged when **any** of these holds: the CUSUM fires, the one-step
change exceeds `R_max`, the implied bias exceeds `κ·σ_b`, or the slope exceeds
the envelope. CUSUM alone is not sufficient — a ramp of 0.045 °C/sample produces
a residual below the slack and never accumulates at all, which is exactly how
the stealth attack went undetected in the first draft.

### 3.5 Decision rules (E5)

Evaluated in this order; the order is load-bearing.

| # | Condition | Verdict | Reasoning |
|---|---|---|---|
| 0 | channel already quarantined | POISON | a quarantined channel is not re-litigated |
| 1 | \|ΔT_corr\| > R_max | POISON | no physical process moves the quantity this fast |
| 2 | \|ρ\| > ρ_max·tol | POISON | drift-shaped, but outside the ageing envelope |
| 3 | persistence ≥ 18 and window full | DRIFT | smooth, one-directional, inside the envelope |
| 4 | otherwise | UNKNOWN | hold, escalate, do not learn |

Rule 3 requires more persistence (18 samples) than the slope window is long
(16). Until the window lies wholly inside the event the rate test cannot see the
true rate, and a premature DRIFT verdict would start *correcting an injection*.

### 3.6 Response (E8)

| Cause | Action |
|---|---|
| CLEAN | forward; scales and vapour reference may adapt |
| DRIFT | correct and **keep forwarding**: `c ← clamp(c + clamp(g·b, ±0.02), ±3.0)`, `g = 0.25` |
| POISON | quarantine, freeze every adaptive element, latch for 12 samples, red LED |
| UNKNOWN | hold; no forwarding, no adaptation |

The correction is a fraction of the estimated bias, rate-limited per sample and
capped in total. A capped, rate-limited corrector cannot absorb an attack faster
than the detector can classify it.

**Standing classification.** While a correction above a deadband is in force the
channel stays classified as drifting. Its bias reads zero *because the
correction is working*, which is evidence the correction works — not evidence
the sensor recovered.

---

## 4. Algorithms: baseline vs proposed

### 4.1 Baseline (existing method)

The prior-art / Generation-1 behaviour: reference the signal against its own
recent mean, threshold, and treat every alarm the same way.

```
for each sample y:
    ŷ   ← mean of the last W reported values      # reference from the signal itself
    r   ← y − ŷ
    C   ← max(0, C + r − k)
    if C ≥ h:
        raise an anomaly                          # one response for every cause
        absorb r into the reference               # unbounded
    append y to the rolling window                # unconditional adaptation
```

Two structural weaknesses, both measured in §5: the reference is derived from
the signal it is meant to judge, so a slow injection drags it; and there is no
cause vocabulary, so there is nothing to act differently on.

### 4.2 Proposed (PIRCAD edge)

```
for each sample (T_raw, RH):
    T_corr ← T_raw − c                                       # E8 standing correction

    # E1 expected response from physics, not from recent history
    T̂_amb  ← T̂_amb + α (T_corr − T̂_amb)
    T̂      ← T̂ + (Δt/τ)(T̂_amb − T̂)
    r      ← T_corr − T̂

    # E4 what the OTHER channel says the temperature is
    T_RH   ← magnus⁻¹( 100 ê / RH )
    b      ← T_corr − T_RH
    ρ      ← slope over window of (b + c)                    # includes absorbed part

    # E3 cheap trigger
    C±     ← cusum(r)
    R_max  ← (Δt/τ)|T̂_amb − T_corr(k−1)| + κσ
    fired  ← C ≥ h  or  |ΔT_corr| > R_max  or  |b| > κσ_b  or  |ρ| > ρ_max·tol

    # E7 a quarantined channel is not re-litigated
    if hold > 0:
        hold ← fired ? HOLD : hold − 1
        emit POISON; freeze; do not advance T̂                # ← fixes defect 1
        continue

    # E5 root cause, in this order
    if not fired:            verdict ← DRIFT if |c| > dead else CLEAN
    elif |ΔT_corr| > R_max:  verdict ← POISON                # unreachable
    elif |ρ| > ρ_max·tol:    verdict ← POISON                # over the envelope ← fixes defect 2
    elif persistence ≥ P:    verdict ← DRIFT
    else:                    verdict ← UNKNOWN

    # E8 differential response
    case CLEAN:   forward;                 adapt σ, σ_b, ê
    case DRIFT:   forward;  c ← clamp(c + clamp(g·b, ±rate), ±cap)
    case POISON:  quarantine; hold ← HOLD; freeze everything
    case UNKNOWN: hold;                    freeze everything

    if not frozen: T̂ ← T̂ + gain·r                            # gated state advance
```

The two lines marked *fixes defect N* are the substantive difference from the
first draft of this firmware; §5.2 gives the measurements that motivated them.

---

## 5. Results

`cd hardware && ./run_checks.sh`. All figures are deterministic.

### 5.1 Edge kernel — per-mode recall

Fraction of injected samples given the correct root cause. Identical input to
all three configurations.

| Injection mode | Truth | Baseline | Draft | **Proposed** |
|---|---|---|---|---|
| CLEAN | CLEAN | 100.0 % | 100.0 % | **100.0 %** |
| DRIFT_LIN (+0.020 °C/sample) | DRIFT | 0.0 % | 0.0 % | **88.1 %** |
| DRIFT_EXP (settling to +2.5 °C) | DRIFT | 0.0 % | 0.0 % | **87.5 %** |
| POISON_STEP (+6 °C held) | POISON | 0.0 % | 0.6 % | **100.0 %** |
| POISON_SPIKE (+9 °C isolated) | POISON | 0.0 % | 33.8 % | **100.0 %** |
| POISON_STEALTH (+0.045 °C/sample) | POISON | 0.0 % | 0.0 % | **93.1 %** |
| POISON_MATCHED (+0.020 °C/sample) | POISON | 0.0 % | 0.0 % | 0.0 % † |

† not separable by design — see §6.

### 5.2 Reference departure — how much of the attack was absorbed

How far each system's own reference was pulled from the truth by the end of the
run. This is the technical effect the architecture exists to produce.

| Injection mode | Baseline | Draft | **Proposed** |
|---|---|---|---|
| CLEAN | −0.11 °C | −0.11 °C | −0.05 °C |
| DRIFT_LIN | +3.03 °C | +3.01 °C | +3.12 °C ‡ |
| POISON_STEP (+6 °C) | **+5.82 °C** | +0.46 °C | **+0.55 °C** |
| POISON_SPIKE | +1.41 °C | −0.01 °C | +0.57 °C |
| POISON_STEALTH | **+6.81 °C** | +6.72 °C | **+0.76 °C** |

‡ absorbing benign ageing is the *correct* behaviour — that is what
recalibration is for. Absorbing an attack is not.

The baseline takes essentially the whole 6 °C step and the whole stealth ramp
into its reference. The proposed configuration admits under 1 °C of either — a
10× and 9× reduction. That is the same failure, and the same fix, as the
adaptation-poisoning experiment in the fog pipeline.

### 5.3 Two defects found and fixed during implementation

The first draft of this firmware was tested by porting its decision function to
the host and driving it with the scenarios above. It failed two of them:

| | Draft behaviour | Cause | Fix |
|---|---|---|---|
| Held +6 °C step | 1 sample of 50 called POISON; by sample 17 it flipped to DRIFT and absorbed the attack into its baseline | after the first sample a held step is no longer an unreachable *change*, so it fell through to the benign branch | quarantine latch (E7): a channel under quarantine is not re-litigated |
| Stealth ramp | 60 of 60 samples called CLEAN | the residual (0.045–0.136 °C) never exceeds the CUSUM slack (0.15 °C), so the statistic never accumulates | ageing-envelope test on the cross-channel bias slope, evaluated independently of the CUSUM |

Both are worth recording because both are the *precise* failure this research
describes, occurring inside our own implementation. A detector whose adaptive
path absorbs the attack is the thing the paper is about.

### 5.4 Fog tier

The edge stream replayed through `fog_bridge.py`, which imports the residual
engine, trigger, adaptation governor, calibration manager and response
controller from `../pircad/`.

| Mode | Root-cause accuracy | Poisoning contained |
|---|---|---|
| CLEAN | 1.000 | — |
| DRIFT_LIN | 0.856 | — |
| DRIFT_EXP | 0.845 | — |
| POISON_STEP | 0.884 | 0.884 |
| POISON_SPIKE | 0.884 | 0.884 |
| POISON_STEALTH | 0.823 | 0.823 |
| POISON_MATCHED | 0.000 † | 0.000 |

Overall root-cause accuracy 0.756 across all seven modes; **drift kept flowing
0.997**, and no benign drift was ever quarantined (the DRIFT→POISON cell of the
confusion matrix is zero). Excluding the deliberately unseparable matched case,
accuracy is 0.88.

---

## 6. Scope and limitations

Stated plainly, and asserted in the test suite so they cannot regress into
unearned claims.

1. **A rate-matched injection is not separable at this node.** An attacker who
   ramps at exactly the benign ageing rate produces a smooth, one-directional
   departure inside the envelope — which is what benign ageing produces. Both
   hypotheses predict the same thing on both channels. `test_edge.cpp` asserts
   that this case *fails*. Separating it requires the redundancy of the
   multi-sensor fog node, where several instruments constrained by independent
   physical relations disagree in a cause-specific pattern.
2. **The consistency test assumes conserved vapour pressure.** True over
   minutes in a given air mass; not true across ventilation events or
   condensation. A deployment would gate the test on a stability check.
3. **The ageing envelope is a datasheet number.** `ρ_max = 0.030 °C/sample` is
   the parameter the drift/poison boundary rests on, and it must come from the
   transducer's specification.
4. **Detection of a slow ramp is delayed by construction** — roughly 18 samples
   (36 s), because the rate test needs its window to lie inside the event. This
   is physics-limited, not a defect, but it is real.
5. **Not yet run on physical hardware.** Verified in Wokwi and by host
   regression tests. Noise on a real DHT22 is larger than the 0.05 °C modelled
   here, which will widen `σ` and push the detectable drift rate up.

---

## 7. Code map

| File | Role | Lines |
|---|---|---|
| `hardware/pircad_edge.h` | decision kernel — pure C++, no Arduino dependencies | ~490 |
| `hardware/sketch.ino` | firmware: sensor, LEDs, serial, MQTT. I/O only | ~310 |
| `hardware/diagram.json` | circuit netlist | 11 parts, 16 nets |
| `hardware/test_edge.cpp` | host regression test, 20 assertions | ~210 |
| `hardware/fog_bridge.py` | fog tier: live MQTT or offline replay | ~380 |
| `hardware/ci/` | board-API stubs, netlist validator | — |
| `hardware/run_checks.sh` | everything verifiable without hardware | — |

The split between `pircad_edge.h` and `sketch.ino` is what makes the firmware
testable: all decision logic lives in the header with no board dependencies, so
the same code that runs on the ESP32 is compiled and exercised on the host. The
sketch contains no decision logic at all.
