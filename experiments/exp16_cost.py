"""
Experiment 16 -- resource figures for the fog tier (first-round review).

Single process, single thread.  Reports, for PIRCAD with the default random
forest, the memory held by the flattened forest, the resident-set growth of a
fog node over a full episode, per-frame latency and sustained throughput, and
the same for the three self-calibrating baselines on the same machine.  The
machine is a development laptop, not a fog gateway; the figures are an upper
bound on throughput and are reported as such.

Run:  python experiments/exp16_cost.py
"""

from __future__ import annotations

import platform
import resource
import time

import numpy as np

from v2_common import eval_specs, make_classifier, save_json, worker_models


def rss_mb():
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 2**20 if platform.system() == "Darwin" else r / 2**10


def main():
    from pircad.fognode import PIRCADFogNode
    from pircad.simulator import frames
    blob = worker_models()
    ff = blob["backends"]["rf"].model
    forest_bytes = sum(a.nbytes for a in (ff.roots, ff.left, ff.right, ff.feat, ff.thr, ff.val))
    specs = [s for s in eval_specs() if s.scenario in ("drift", "stealth_poison", "clean")][:30]
    eps = [s.build() for s in specs]
    rss0 = rss_mb()
    lat, n_frames, wall = [], 0, 0.0
    for ep in eps:
        node = PIRCADFogNode(classifier=make_classifier("rf"))
        fr = list(frames(ep))
        t0 = time.perf_counter()
        res = node.run(iter(fr), truth_labels=ep.labels)
        dt = time.perf_counter() - t0
        wall += dt; n_frames += len(fr)
    rss1 = rss_mb()
    # per-frame latency distribution, from a step-wise run of one episode
    ep = eps[1]
    node = PIRCADFogNode(classifier=make_classifier("rf"))
    t_frames = []

    def timed(it):
        for f in it:
            t0 = time.perf_counter()
            yield f                       # resumes when the node asks for the next frame
            t_frames.append(time.perf_counter() - t0)
    for e in eps:
        PIRCADFogNode(classifier=make_classifier("rf")).run(timed(frames(e)), truth_labels=e.labels)
    from pircad.baselines import AdaptiveRecalibratingBaseline
    from pircad.baselines_adaptive import KalmanBiasEstimator, PageHinkleyRecalibrator
    base = {}
    for name, fnc in (
            ("EWMA recalibration", lambda e: AdaptiveRecalibratingBaseline(
                alpha=0.002, recal_gain=0.3, warmup=600).run(e.obs["T_in"])),
            ("Kalman bias estimator", lambda e: KalmanBiasEstimator(q_b=0.02 ** 2).run(
                e.obs["T_in"], e.actuation["u_heater"], e.obs["T_amb"])),
            ("Page-Hinkley recalibration", lambda e: PageHinkleyRecalibrator(
                delta=0.01, lam=5.0).run(e.obs["T_in"]))):
        t0 = time.perf_counter(); n = 0
        for e in eps:
            fnc(e); n += len(e.labels)
        base[name] = {"us_per_sample": (time.perf_counter() - t0) / n * 1e6}
    out = {"machine": f"{platform.machine()} {platform.system()} {platform.release()}",
           "python": platform.python_version(),
           "forest": {"trees": int(ff.n_trees), "nodes": int(len(ff.feat)),
                      "max_depth": int(ff.depth), "bytes": int(forest_bytes)},
           "pircad": {"frames": n_frames, "frames_per_s": n_frames / wall,
                      "ms_per_frame_mean": wall / n_frames * 1e3,
                      "rss_mb_before": rss0, "rss_mb_after": rss1},
           "baselines_single_channel": base}
    if t_frames:
        t = np.array(t_frames) * 1e3
        out["pircad"].update({"ms_median": float(np.median(t)), "ms_p95": float(np.quantile(t, .95)),
                              "ms_p99": float(np.quantile(t, .99)), "ms_max": float(t.max())})
    import json
    print(json.dumps(out, indent=1))
    save_json("exp16_cost.json", out)


if __name__ == "__main__":
    main()
