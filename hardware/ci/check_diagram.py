#!/usr/bin/env python3
"""Validate diagram.json: JSON well-formed, every connection endpoint refers to
a declared part, and every ESP32 pin used by the firmware is actually wired."""
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
diagram = json.loads((HERE.parent / "diagram.json").read_text())
sketch = (HERE.parent / "sketch.ino").read_text()

parts = {p["id"]: p["type"] for p in diagram["parts"]}
used = set()
for conn in diagram["connections"]:
    for end in conn[:2]:
        if not end or end.startswith("$"):
            continue
        pid, _, pin = end.partition(":")
        if pid not in parts:
            sys.exit(f"FAIL: connection refers to unknown part {pid!r}")
        used.add((pid, pin))

esp_pins = {pin for pid, pin in used if "esp32" in parts[pid]}

# every GPIO the firmware drives or reads must appear in the netlist
declared = {int(m.group(2)): m.group(1)
            for m in re.finditer(r"#define\s+(PIN_\w+)\s+(\d+)", sketch)}
missing = [f"{name} (GPIO{gpio})" for gpio, name in declared.items()
           if f"D{gpio}" not in esp_pins]
if missing:
    sys.exit("FAIL: firmware uses pins that are not wired: " + ", ".join(missing))

for required in ("3V3", "GND.1", "TX0", "RX0"):
    if required not in esp_pins:
        sys.exit(f"FAIL: {required} is not connected")

print(f"OK: {len(parts)} parts, {len(diagram['connections'])} connections")
print(f"    ESP32 pins wired: {', '.join(sorted(esp_pins))}")
print(f"    firmware pins   : " +
      ", ".join(f"{n}=GPIO{g}" for g, n in sorted(declared.items())))
