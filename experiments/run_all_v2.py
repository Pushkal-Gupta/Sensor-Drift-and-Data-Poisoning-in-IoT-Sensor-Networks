"""Run the journal evaluation end to end (experiments 6-11 + figures).

    python experiments/run_all_v2.py            # everything
    python experiments/run_all_v2.py --only exp7 exp9

Experiments 1-5 and their runner (run_all.py) are unchanged.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
STEPS = ["fit_models.py", "exp6_statistical_benchmark.py", "exp7_adaptive_attacker.py",
         "exp8_distribution_shift.py", "exp9_adaptation_poisoning.py",
         "exp10_intel_lab.py", "exp11_ablation.py", "exp12_hybrid_features.py", "exp13_intel_shift.py",
         "make_figures_v2.py"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    failed = []
    for s in STEPS:
        if a.only and not any(o in s for o in a.only):
            continue
        print(f"\n$ python {s}", flush=True)
        rc = subprocess.call([sys.executable, "-u", str(HERE / s)], cwd=HERE)
        if rc:
            failed.append(s)
    print("\nFAILED: " + ", ".join(failed) if failed else "\nAll steps completed.")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
