"""Fit (or refit) every model used by experiments 6-11.

    python experiments/fit_models.py          # fit whatever is not cached
    python experiments/fit_models.py --force  # refit everything
"""

from __future__ import annotations

import argparse

from v2_common import get_ml_baselines, get_v2_models

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    get_v2_models(force=a.force)
    get_ml_baselines(force=a.force)
