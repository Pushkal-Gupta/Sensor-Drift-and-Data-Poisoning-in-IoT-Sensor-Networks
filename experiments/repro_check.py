"""Re-run the 180-episode PIRCAD-rf bank and compare with the stored artifact."""
import json

from v2_common import ARTIFACTS, RunCfg, eval_specs, pmap, run_pircad

KEYS = ["macro_f1", "root_cause_accuracy", "false_drift_rate", "false_poison_rate",
        "false_alarm_rate"]

if __name__ == "__main__":
    rows = pmap(run_pircad, [(s, RunCfg()) for s in eval_specs()], "repro")
    ref = {(r["seed"], r["scenario"]): r for r in json.load(
        open(ARTIFACTS / "exp6_statistical_benchmark.json"))["rows"]["PIRCAD-rf"]}
    bad = sum(1 for r in rows for k in KEYS
              if not ((r[k] != r[k] and ref[(r["seed"], r["scenario"])][k] != ref[(r["seed"], r["scenario"])][k])
                      or r[k] == ref[(r["seed"], r["scenario"])][k]))
    print(f"REPRO: {len(rows)} episodes, {len(rows) * len(KEYS)} values, mismatches: {bad}")
