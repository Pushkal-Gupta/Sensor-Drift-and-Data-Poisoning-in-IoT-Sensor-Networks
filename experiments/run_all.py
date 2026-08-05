"""Run the whole evidence suite and write artifacts/ and figures/.

    python experiments/run_all.py            # everything except the quantum stage
    python experiments/run_all.py --quantum  # include the quantum-kernel backend
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
STEPS = [
    ("exp1_root_cause_benchmark.py", []),
    ("exp2_poisoned_adaptation.py", []),
    ("exp3_ablation.py", []),
    ("exp4_escalation.py", []),
    ("exp5_legacy_dataset.py", []),
    ("make_figures.py", []),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quantum", action="store_true")
    ap.add_argument("--only", nargs="*", help="run only these scripts")
    args = ap.parse_args()

    failures = []
    for script, extra in STEPS:
        if args.only and not any(o in script for o in args.only):
            continue
        cmd = [sys.executable, str(HERE / script), *extra]
        if args.quantum and script == "exp4_escalation.py":
            cmd.append("--quantum")
        print(f"\n$ {' '.join(cmd)}", flush=True)
        rc = subprocess.call(cmd, cwd=HERE)
        if rc != 0:
            failures.append((script, rc))
            print(f"!! {script} exited {rc}", file=sys.stderr)

    if failures:
        print("\nFAILED:", ", ".join(f"{s} ({rc})" for s, rc in failures))
        return 1
    print("\nAll experiments completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
