"""
Monte-Carlo validation of the v2 pipeline (no files are written, nothing is stored).

Run from anywhere:
    python tools/monte_carlo_validation.py                 # 200 seeds (seeds 0..199)
    python tools/monte_carlo_validation.py --seeds 50 --n-rx 6

For every seed it regenerates the dataset in memory (person-a/generate_dataset.py), runs
Person B's noise-aware rank + basis, then Person C's projection + least squares, and compares
with the legacy / failing variants. This reproduces the numbers quoted in docs/DATASET_FIX.md.

Reported (mean +- std over seeds, averaged over the x/y/z axes):
  * fraction of seeds where the noise-aware rank equals the true rank (3)
  * RMSE vs the hidden true path for: raw GPS, legacy all-channel basis, IMU-only basis,
    projection, least squares, projection + least squares, and the fused pipeline
"""

import argparse
import importlib.util
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def load_module(name, rel):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, rel))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


gen = load_module("generate_dataset", "person-a/generate_dataset.py")
B = load_module("person_B", "person-b/person_B.py")
C = load_module("person_C", "person-c/person_C.py")


def one_seed(seed, n_rx):
    cols, sigma = gen.generate(seed=seed, n_rx=n_rx)
    t = cols["time"]
    chans = [c for c in cols if c != "time" and not c.startswith("true_")]
    M = np.column_stack([cols[c] for c in chans])
    sig = np.array([sigma[c] for c in chans])

    # ---- Person B v2 ----
    Q, r = B.signal_basis(M, sig)

    # ---- legacy B on the noisy matrix: keeps every column -> Q spans all channels ----
    keep = B.find_independent_columns(M)
    Q_all = B.gram_schmidt(M[:, keep])

    # ---- IMU-only basis (the failed "Option A") ----
    imu = [i for i, c in enumerate(chans) if c.startswith(("acc_", "gyro_", "vel_"))]
    Q_imu = B.gram_schmidt(M[:, imu])

    out = {"rank_ok": float(r == 3), "legacy_cols_kept": len(keep) / len(chans)}
    res = C.run_stages(cols, Q)
    keys = list(res["x"]["rmse"])
    for k in keys:
        out[k] = np.mean([res[a]["rmse"][k] for a in "xyz"])

    for label, Qx in (("legacy all-channel basis, projected only", Q_all),
                      ("IMU-only basis, projected only", Q_imu)):
        out[label] = np.mean([C.rmse(C.project_onto_basis(Qx, cols[f"gps_{a}"]), cols[f"true_{a}"])
                              for a in "xyz"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=200)
    ap.add_argument("--n-rx", type=int, default=3)
    args = ap.parse_args()

    runs = [one_seed(s, args.n_rx) for s in range(args.seeds)]
    keys = list(runs[0])
    stat = {k: (np.mean([r[k] for r in runs]), np.std([r[k] for r in runs])) for k in keys}

    print(f"Monte Carlo: {args.seeds} seeds, {args.n_rx} GPS receivers/axis, 101 timestamps\n")
    print(f"noise-aware rank == 3 in {stat['rank_ok'][0] * 100:.1f}% of seeds")
    print(f"legacy find_independent_columns keeps {stat['legacy_cols_kept'][0] * 100:.0f}% of columns "
          "(i.e. never removes anything)\n")
    print(f"{'RMSE vs true path (mean over x,y,z)':<66s}{'mean':>9s}{'std':>9s}")
    order = [k for k in keys if k not in ("rank_ok", "legacy_cols_kept")]
    for k in order:
        print(f"{k:<66s}{stat[k][0]:9.4f}{stat[k][1]:9.4f}")


if __name__ == "__main__":
    main()
