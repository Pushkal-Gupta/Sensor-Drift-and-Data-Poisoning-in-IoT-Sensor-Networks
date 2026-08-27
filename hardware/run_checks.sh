#!/usr/bin/env bash
# Everything that can be verified without hardware.
#   ./run_checks.sh
set -euo pipefail
cd "$(dirname "$0")"

echo "== 1. type-check the firmware against stubbed board APIs =="
g++ -std=c++11 -Wall -Wextra -Wno-unused-parameter -I. -Ici \
    -o ci/compile_check ci/compile_check.cpp
./ci/compile_check

echo
echo "== 2. regression-test the edge decision kernel =="
g++ -O2 -std=c++11 -Wall -o test_edge test_edge.cpp
./test_edge

echo
echo "== 3. validate the circuit netlist =="
python3 ci/check_diagram.py

echo
echo "== 4. replay the edge stream through the fog pipeline =="
./test_edge --csv > ci/edge_runs.csv
python3 fog_bridge.py --replay ci/edge_runs.csv --config proposed

echo
echo "all hardware-side checks passed"
