"""
Real-data embodiment: indoor temperature/humidity motes (Intel Berkeley Lab).

The same fog pipeline, with a different expected-response model and a different
constraint set, applied to the public Intel Berkeley Research Lab deployment
(54 Mica2Dot motes, Feb-Apr 2004).  There is no actuator here; the physical
redundancy comes from two places:

  * psychrometrics -- each mote's temperature and relative-humidity transducers
    are tied by the Magnus relation through the vapour pressure e of the air,
    and e is spatially near-uniform in a room, so neighbouring motes supply it;
  * spatial correlation -- neighbouring motes see the same room.

Dedicated observers, as in the vessel twin:

    expected(T)  <- Magnus inversion of this mote's RH at the neighbours' e,
                    fused with the neighbours' mean temperature + offset
    expected(RH) <- this mote's T and the neighbours' e

Constraint set (E4) and hypothesis signatures, derived from the physics at the
commissioning operating point -- no labelled fault data:

    g_mag  = T - T_Magnus(RH, e_nb)     moves for a T or RH bias; NOT for a
                                        genuine local heating, which moves T and
                                        RH consistently at constant e
    g_sp   = T - T_nb - a               moves for a T bias and for local heating
    g_rhsp = RH - RH_nb - b             moves for an RH bias and (negatively)
                                        for local heating

The data loader bins the raw ~31 s epochs into 2-minute samples and
interpolates gaps up to 30 minutes.  Days 0-10 are used: from day 11 on, many
motes' batteries fail and their readings become meaningless.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .consistency import PROCESS_HYPOTHESIS, Localisation
from .simulator import Episode, magnus_es
from .twin import ProcessModel, _as_weight
from .types import Frame, RootCause

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "external" / "intel_lab"
URL = "https://db.csail.mit.edu/labdata/data.txt.gz"
LOCS_URL = "https://db.csail.mit.edu/labdata/mote_locs.txt"

BIN_EPOCHS = 4          # 4 x 31 s ~ 2 minutes
N_DAYS = 11
SAMPLES_PER_DAY = 2880 // BIN_EPOCHS
INDOOR_CHANNELS = ("T", "RH")


def magnus_inv(e: np.ndarray | float) -> np.ndarray | float:
    z = np.log(np.maximum(e, 1e-6) / 610.94)
    return 243.04 * z / (17.625 - z)


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------
def download(force: bool = False) -> None:
    import urllib.request
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for url, name in ((URL, "data.txt.gz"), (LOCS_URL, "mote_locs.txt")):
        p = DATA_DIR / name
        if force or not p.exists():
            print(f"[intel] downloading {url}")
            urllib.request.urlretrieve(url, p)


def load_intel(cache: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (T, RH, locs): binned per-mote series and mote coordinates."""
    cp = DATA_DIR / "binned.pkl"
    if cache and cp.exists():
        return pd.read_pickle(cp)
    download()
    df = pd.read_csv(DATA_DIR / "data.txt.gz", sep=r"\s+", header=None,
                     names=["date", "time", "epoch", "mote", "T", "RH",
                            "light", "V"], on_bad_lines="skip")
    df = df.dropna(subset=["epoch", "mote", "T", "RH"])
    df = df[(df.mote >= 1) & (df.mote <= 54)]
    # physically valid range for an occupied office; out-of-range readings in
    # this dataset are battery-failure artefacts (e.g. T = 122 C)
    df = df[df["T"].between(5, 45) & df["RH"].between(5, 95)]
    df["bin"] = (df.epoch // BIN_EPOCHS).astype(int)
    n = SAMPLES_PER_DAY * N_DAYS
    df = df[df.bin < n]
    T = df.pivot_table(index="bin", columns="mote", values="T").reindex(range(n))
    H = df.pivot_table(index="bin", columns="mote", values="RH").reindex(range(n))
    T = T.interpolate(limit=15, limit_area="inside")
    H = H.interpolate(limit=15, limit_area="inside")
    T.columns = T.columns.astype(int); H.columns = H.columns.astype(int)
    locs = pd.read_csv(DATA_DIR / "mote_locs.txt", sep=r"\s+", header=None,
                       names=["mote", "x", "y"]).set_index("mote")
    out = (T, H, locs)
    if cache:
        pd.to_pickle(out, cp)
    return out


def usable_motes(T: pd.DataFrame, H: pd.DataFrame, max_missing: float = 0.02) -> List[int]:
    ok = (T.isna().mean() <= max_missing) & (H.isna().mean() <= max_missing)
    return sorted(int(m) for m in T.columns[ok])


def neighbours(m: int, locs: pd.DataFrame, pool: Sequence[int], k: int = 4) -> List[int]:
    d = np.hypot(locs.x - locs.x[m], locs.y - locs.y[m])
    order = [int(i) for i in d.sort_values().index if int(i) != m and int(i) in pool]
    return order[:k]


# --------------------------------------------------------------------------
# episodes
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class IndoorSpec:
    mote: int
    scenario: str
    seed: int
    onset: int
    rate: float
    attack: str = "substitute"   # substitute | mitm


WARMUP_INDOOR = 2 * SAMPLES_PER_DAY      # two days of commissioning


def make_indoor_episode(spec: IndoorSpec, T: pd.DataFrame, H: pd.DataFrame,
                        locs: pd.DataFrame, pool: Sequence[int]) -> Episode:
    rng = np.random.default_rng(spec.seed)
    m = spec.mote
    nb = neighbours(m, locs, pool)
    Tm = T[m].ffill().bfill().to_numpy(float)
    Hm = H[m].ffill().bfill().to_numpy(float)
    Tn = T[nb].ffill().bfill().to_numpy(float)
    Hn = H[nb].ffill().bfill().to_numpy(float)
    e_nb = (Hn / 100.0 * magnus_es(Tn)).mean(axis=1)
    T_nb = Tn.mean(axis=1)
    RH_nb = Hn.mean(axis=1)
    n = len(Tm)

    obs = {"T": Tm.copy(), "RH": Hm.copy()}
    truth = {"T": Tm.copy(), "RH": Hm.copy()}
    labels = np.full(n, int(RootCause.NORMAL))
    s = spec.onset
    tau = np.arange(n - s, dtype=float)
    tgt: Optional[str] = "T"
    # event polarity: drift/attack direction, local heating or cooling
    sgn = 1.0 if rng.random() < 0.5 else -1.0
    rate = sgn * spec.rate
    # the mote's own sample-to-sample noise, for a noise-matched attacker
    sig = float(np.std(np.diff(Tm[:WARMUP_INDOOR]))) / np.sqrt(2.0)

    if spec.scenario == "clean":
        tgt = None
    elif spec.scenario == "drift":
        obs["T"][s:] += rate * tau
        labels[s:] = RootCause.SENSOR_DRIFT
    elif spec.scenario == "stealth_poison":
        if spec.attack == "mitm":
            obs["T"][s:] += rate * tau
        else:
            sur = np.empty(n); v = Tm[max(0, s - 1)]
            for k in range(n):
                v = 0.98 * v + 0.02 * Tm[k]
                sur[k] = v
            obs["T"][s:] = sur[s:] + rate * tau + rng.normal(0, sig, n - s)
        labels[s:] = RootCause.DATA_POISONING
    elif spec.scenario == "spike_poison":
        idx = rng.choice(np.arange(s, n), size=max(3, (n - s) // 200), replace=False)
        obs["T"][idx] += rng.choice([-1, 1], idx.size) * rng.uniform(3.0, 6.0, idx.size)
        labels[idx] = RootCause.DATA_POISONING
    elif spec.scenario == "replay_poison":
        lag = SAMPLES_PER_DAY      # replay yesterday: right diurnal shape, wrong day
        obs["T"][s:] = Tm[s - lag:n - lag]
        labels[s:] = RootCause.DATA_POISONING
    elif spec.scenario == "process_event":
        # a genuine local heat source (or cold draught) next to the mote: T
        # moves, and RH moves the other way exactly as psychrometrics requires
        # at constant vapour pressure
        A = sgn * rng.uniform(1.5, 4.0); tc = rng.uniform(30, 120)
        dT = A * (1.0 - np.exp(-tau / tc))
        e_true = Hm[s:] / 100.0 * magnus_es(Tm[s:])
        T_new = Tm[s:] + dT
        for d in (obs, truth):
            d["T"][s:] = T_new
            d["RH"][s:] = 100.0 * e_true / magnus_es(T_new)
        labels[s:] = RootCause.PROCESS_EVENT
        tgt = None
    else:
        raise ValueError(spec.scenario)

    return Episode(t=np.arange(n, dtype=float) * 120.0, truth=truth, obs=obs,
                   actuation={"T_nb": T_nb, "e_nb": e_nb, "RH_nb": RH_nb},
                   labels=labels, target_channel=tgt, scenario=spec.scenario,
                   params=None,
                   meta={"onset": float(s), "rate": float(spec.rate),
                         "seed": float(spec.seed), "mote": float(m),
                         "signals": ["T", "RH", "T_nb", "RH_nb"],
                         "pairs": [("T", "RH"), ("T", "T_nb"), ("RH", "RH_nb"),
                                   ("T_nb", "RH_nb")],
                         "warm": WARMUP_INDOOR})


def indoor_frames(ep: Episode):
    for k in range(len(ep.labels)):
        yield Frame(k=k, t=float(ep.t[k]),
                    values={c: float(ep.obs[c][k]) for c in INDOOR_CHANNELS},
                    actuation={a: float(v[k]) for a, v in ep.actuation.items()})


# --------------------------------------------------------------------------
# expected response
# --------------------------------------------------------------------------
class IndoorTwin(ProcessModel):
    """Leave-one-out psychrometric + spatial expected response."""

    channels = INDOOR_CHANNELS

    def __init__(self, warmup: int = WARMUP_INDOOR, adapt: float = 1.0 / 2000.0):
        self.warmup = warmup
        self.adapt = adapt
        self.n = 0
        self.a_T = 0.0      # T - T_nb offset
        self.c_e = 0.0      # e - e_nb offset
        self.w_mag = 0.5    # fusion weight of the Magnus estimate
        self._acc: Dict[str, List[float]] = {"dT": [], "de": [], "rmag": [], "rsp": []}

    def _e(self, T: float, RH: float) -> float:
        return RH / 100.0 * float(magnus_es(T))

    def T_magnus(self, RH: float, e_nb: float) -> float:
        return float(magnus_inv(100.0 * (e_nb + self.c_e) / max(RH, 1e-3)))

    def predict(self, frame: Frame) -> Dict[str, float]:
        v, a = frame.values, frame.actuation
        T_sp = a["T_nb"] + self.a_T
        T_mag = self.T_magnus(v["RH"], a["e_nb"])
        exp_T = self.w_mag * T_mag + (1.0 - self.w_mag) * T_sp
        exp_RH = 100.0 * (a["e_nb"] + self.c_e) / float(magnus_es(v["T"]))
        return {"T": exp_T, "RH": exp_RH}

    def update(self, frame: Frame, trust: Dict[str, float]) -> None:
        v, a = frame.values, frame.actuation
        dT = v["T"] - a["T_nb"]
        de = self._e(v["T"], v["RH"]) - a["e_nb"]
        self.n += 1
        if self.n <= self.warmup:
            self._acc["dT"].append(dT); self._acc["de"].append(de)
            self.a_T = float(np.mean(self._acc["dT"]))
            self.c_e = float(np.mean(self._acc["de"]))
            self._acc["rmag"].append(v["T"] - self.T_magnus(v["RH"], a["e_nb"]))
            self._acc["rsp"].append(dT - self.a_T)
            if self.n == self.warmup // 2:
                # inverse-variance fusion of the two leave-one-out estimates,
                # fixed half-way through commissioning so that the residual
                # statistics (learned on the second half) see the final weights
                vm = float(np.var(self._acc["rmag"][self.warmup // 8:])) + 1e-6
                vs = float(np.var(self._acc["rsp"][self.warmup // 8:])) + 1e-6
                self.w_mag = (1 / vm) / (1 / vm + 1 / vs)
            return
        # governed re-anchoring: only while BOTH transducers are trusted, since
        # each offset is a joint property of the pair
        w = min(_as_weight(trust.get("T", 1.0)), _as_weight(trust.get("RH", 1.0)))
        if w >= 0.999:
            self.a_T += self.adapt * (dT - self.a_T)
            self.c_e += self.adapt * (de - self.c_e)


class IndoorConsistencyMonitor:
    """Psychrometric/spatial constraint evaluation and hypothesis localisation."""

    CONSTRAINTS = ("g_mag", "g_sp", "g_rhsp")

    def __init__(self, twin: IndoorTwin, window: int = 120):
        self.twin = twin
        self.window = window
        self.buf: Dict[str, Deque[float]] = {c: deque(maxlen=window) for c in self.CONSTRAINTS}
        self._warm: Dict[str, List[float]] = {c: [] for c in self.CONSTRAINTS}
        self.mean = {c: 0.0 for c in self.CONSTRAINTS}
        self.sigma = {c: 1.0 for c in self.CONSTRAINTS}
        self.ready = False
        self.signatures: Dict[str, np.ndarray] = {}
        self._rh_off: List[float] = []
        self.b_RH = 0.0
        self._last_act: Dict[str, float] = {}

    def evaluate(self, values, expected, actuation) -> Dict[str, float]:
        tw = self.twin
        T, RH = values["T"], values["RH"]
        self._last_act = dict(actuation)
        if not self.ready:
            self._rh_off.append(RH - actuation["RH_nb"])
            self.b_RH = float(np.mean(self._rh_off))
        return {
            "g_mag": T - tw.T_magnus(RH, actuation["e_nb"]),
            "g_sp": T - actuation["T_nb"] - tw.a_T,
            "g_rhsp": RH - actuation["RH_nb"] - self.b_RH,
        }

    def push(self, g) -> None:
        for c in self.CONSTRAINTS:
            self.buf[c].append(g[c])
            if not self.ready:
                self._warm[c].append(g[c])

    def finalise_baseline(self, values) -> None:
        for c in self.CONSTRAINTS:
            a = np.asarray(self._warm[c][len(self._warm[c]) // 4:], float)
            med = float(np.median(a)); mad = float(np.median(np.abs(a - med)))
            self.mean[c] = med
            self.sigma[c] = max(1.4826 * mad, 1e-6)
        T, RH = values["T"], values["RH"]
        e_nb = self._last_act.get("e_nb", RH / 100 * float(magnus_es(T)))
        s = self.sigma
        h = 1e-3
        dTmag_dRH = (self.twin.T_magnus(RH + h, e_nb) - self.twin.T_magnus(RH, e_nb)) / h
        es = float(magnus_es(T)); des = es * 17.625 * 243.04 / (T + 243.04) ** 2
        dRH_dT_const_e = -RH * des / es
        sig = {
            "T": np.array([1.0 / s["g_mag"], 1.0 / s["g_sp"], 0.0]),
            "RH": np.array([-dTmag_dRH / s["g_mag"], 0.0, 1.0 / s["g_rhsp"]]),
            PROCESS_HYPOTHESIS: np.array([0.0, 1.0 / s["g_sp"],
                                          dRH_dT_const_e / s["g_rhsp"]]),
        }
        self.signatures = {k: w / (np.linalg.norm(w) + 1e-18) for k, w in sig.items()}
        self.ready = True

    def localise(self, span=None) -> Localisation:
        if not self.ready:
            return Localisation()
        span = span or self.window
        g = np.array([(np.mean(list(self.buf[c])[-span:]) - self.mean[c]) / self.sigma[c]
                      for c in self.CONSTRAINTS])
        norm = float(np.linalg.norm(g))
        viol = {c: float(x) for c, x in zip(self.CONSTRAINTS, g)}
        if norm <= 0:
            return Localisation("", 0.0, "", 0.0, 0.0, viol)
        scored = sorted(((k, float(np.dot(w, g) / norm) ** 2 * np.sign(np.dot(w, g)))
                         for k, w in self.signatures.items()),
                        key=lambda t: -t[1])
        best, second = scored[0], scored[1]
        return Localisation(best[0], max(0.0, best[1]), second[0],
                            max(0.0, second[1]), norm, viol)


def indoor_config():
    """FogConfig with indoor per-channel envelopes and caps.

    The ageing envelope (2.5e-3 C per 2-min sample, ~1.8 C/day) is an
    accelerated-test value -- real thermistor ageing is far slower -- chosen so
    that injected events reach several sigma within the 11-day record.
    """
    from .config import FogConfig
    cfg = FogConfig()
    cfg.envelope.max_rate = {"T": 2.5e-3, "RH": 1.0e-2}
    cfg.recal_total_cap = {"T": 10.0, "RH": 20.0}
    cfg.recal_rate_limit = 5.0e-3
    cfg.residual.baseline_window = WARMUP_INDOOR
    cfg.freeze_hold = 300
    return cfg
