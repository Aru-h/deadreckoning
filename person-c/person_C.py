"""
PERSON C — UAV Sensor Fusion
Projection (first-pass denoising) + Least-Squares trajectory fit + Prediction

Run (after person-a/generate_dataset.py and person-b/person_B.py):
    python person_C.py
    python person_C.py --data ../person-a/uav_sensor_data.csv --basis ../person-b/person_B_orthonormal_basis.csv

Input : ../person-a/uav_sensor_data.csv         (shared data; rows = timestamps)
        ../person-b/person_B_orthonormal_basis.csv  (Q: n_timestamps x r, orthonormal columns)
Output: person_C_coefficients.csv   quadratic coefficients + residual norm per axis  -> Person D
        person_C_output.csv         time, projected, fitted trajectory               -> Person D
        person_C_prediction.csv     extrapolated positions beyond the data           -> Person D
        person_C_trajectory.png, person_C_axes.png

PIPELINE (Concept -> Purpose -> Outcome)
  1. Projection:  p = P x = Q (Q^T x),  P = Q Q^T.
       Valid because Q has orthonormal columns (Q^T Q = I), so the general projector
       A (A^T A)^-1 A^T collapses to Q Q^T -- no inversion. Each GPS channel is a vector in
       R^(n_timestamps); projecting removes the part of it that is NOT in the signal subspace.
       The projected receivers are averaged (fusion) before fitting.
  2. Least squares:  A = [1, t, t^2] (n x 3, n >> 3: overdetermined), b = projected positions.
       x_hat from numpy.linalg.lstsq (same as the normal equations A^T A x = A^T b, but stable).
       The residual e = b - A x_hat satisfies A^T e = 0 (orthogonal to the column space of A).
  3. Prediction: evaluate the fitted polynomial for t beyond the observed range.

HONEST RESULT (docs/DATASET_FIX.md): projection alone cuts GPS error about 2x; the least-squares
quadratic then does most of the rest, and projection + least squares ties least squares on a single
raw channel. The benefit of redundancy is FUSION across receivers. This script prints that table.

IMPORTANT: Q must contain a position-type direction (it comes from Person B's whitened SVD of ALL
channels). A basis built from IMU channels only spans {1, t} and destroys the position signal.
"""

import argparse
import csv
import os
import sys
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA = os.path.join(HERE, "..", "person-a", "uav_sensor_data.csv")
DEFAULT_BASIS = os.path.join(HERE, "..", "person-b", "person_B_orthonormal_basis.csv")
T_EXTRAPOLATE = 2.0     # seconds of prediction beyond the last timestamp
N_FUTURE = 21


# ----------------------------------------------------------------------
# I/O
# ----------------------------------------------------------------------
def load_sensor_data(filename):
    with open(filename, "r", newline="") as f:
        rows = list(csv.DictReader(f))
    return {k: np.array([float(r[k]) for r in rows], dtype=float) for k in rows[0]}


def receiver_channels(data, axis):
    """All GPS receiver columns of an axis: gps_x, gps2_x, gps3_x, ... (original name first)."""
    pat = re.compile(rf"^gps\d*_{axis}$")
    return sorted([c for c in data if pat.match(c)], key=lambda c: (c != f"gps_{axis}", c))


# ----------------------------------------------------------------------
# Person C maths
# ----------------------------------------------------------------------
def project_onto_basis(Q, x):
    """p = Q Q^T x, computed as Q @ (Q.T @ x) (identical to forming P, but cheaper)."""
    return Q @ (Q.T @ x)


def build_time_design_matrix(t):
    """Quadratic model position(t) = a0 + a1 t + a2 t^2  ->  rows [1, t, t^2]."""
    t = np.asarray(t, dtype=float)
    return np.column_stack((np.ones_like(t), t, t ** 2))


def fit_least_squares(t, b):
    """Solve the overdetermined A x = b in the least-squares sense. Returns (x_hat, e, ||e||)."""
    A = build_time_design_matrix(t)
    coeffs, *_ = np.linalg.lstsq(A, b, rcond=None)
    e = b - A @ coeffs
    return coeffs, e, float(np.linalg.norm(e))


def evaluate_poly(c, t):
    t = np.asarray(t, dtype=float)
    return c[0] + c[1] * t + c[2] * t ** 2


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def run_stages(data, Q):
    """
    Run projection + least squares for the three axes.
    Returns dict axis -> dict with projections, fused projection, coefficients, residuals,
    and (if ground truth is present) RMSE for each processing variant.
    """
    t = data["time"]
    out = {}
    for a in "xyz":
        rx = receiver_channels(data, a)
        raw = [data[c] for c in rx]
        proj = [project_onto_basis(Q, v) for v in raw]
        fused = np.mean(proj, axis=0)
        coeffs, e, enorm = fit_least_squares(t, fused)
        res = {"receivers": rx, "raw": raw, "proj": proj, "fused": fused,
               "coeffs": coeffs, "residual": e, "residual_norm": enorm}
        if f"true_{a}" in data:
            tr = data[f"true_{a}"]
            c_raw, _, _ = fit_least_squares(t, raw[0])
            c_proj1, _, _ = fit_least_squares(t, proj[0])
            c_rawavg, _, _ = fit_least_squares(t, np.mean(raw, axis=0))
            res["rmse"] = {
                "raw (1 receiver)": rmse(raw[0], tr),
                "projected only (1 receiver)": rmse(proj[0], tr),
                "least squares only (1 raw receiver)": rmse(evaluate_poly(c_raw, t), tr),
                "projection + least squares (1 receiver)": rmse(evaluate_poly(c_proj1, t), tr),
                f"average of {len(raw)} raw receivers + least squares": rmse(evaluate_poly(c_rawavg, t), tr),
                f"projection + average of {len(raw)} + least squares  [pipeline]": rmse(evaluate_poly(coeffs, t), tr),
            }
        out[a] = res
    return out


def main():
    if hasattr(sys.stdout, "reconfigure"):   # Windows consoles/pipes may default to a legacy codepage
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description="Person C: projection + least squares + prediction")
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--basis", default=DEFAULT_BASIS)
    ap.add_argument("--out-dir", default=HERE)
    args = ap.parse_args()

    for p, hint in ((args.data, "person-a/generate_dataset.py"), (args.basis, "person-b/person_B.py")):
        if not os.path.exists(p):
            raise FileNotFoundError(f"{p} not found. Run `python {hint}` first.")

    data = load_sensor_data(args.data)
    t = data["time"]
    Q = np.loadtxt(args.basis, delimiter=",", ndmin=2)
    if Q.shape[0] != len(t):
        raise ValueError(f"Q has {Q.shape[0]} rows but the data has {len(t)} timestamps "
                         "(regenerate the data and rerun person_B.py with the same seed).")

    print("=" * 70)
    print("UAV SENSOR FUSION — PERSON C (Projection + Least Squares)")
    print("=" * 70)
    print(f"\n0) INPUTS: {len(t)} timestamps, Q shape {Q.shape} (rank r = {Q.shape[1]})")
    print("max |Q^T Q - I| =", f"{np.max(np.abs(Q.T @ Q - np.eye(Q.shape[1]))):.2e}")

    res = run_stages(data, Q)
    A = build_time_design_matrix(t)

    # ---- 1) projection ----
    print("\n1) PROJECTION  p = Q Q^T x   (each GPS receiver, then averaged)")
    for a in "xyz":
        names = res[a]["receivers"]
        removed = [np.linalg.norm(r - p) for r, p in zip(res[a]["raw"], res[a]["proj"])]
        print(f"{a.upper()}: receivers {names}; ||raw - projected|| = "
              + ", ".join(f"{v:.3f}" for v in removed))

    # ---- 2) least squares ----
    print("\n2) LEAST SQUARES  A^T A x = A^T b   (A = [1, t, t^2], b = fused projection)")
    for a in "xyz":
        r = res[a]
        print(f"\n--- {a.upper()} ---")
        print("coefficients [a0, a1, a2]:", np.round(r["coeffs"], 5))
        print(f"residual norm ||e|| = ||b - A x_hat|| = {r['residual_norm']:.5f}")
        print(f"max |A^T e| (should be ~0) = {np.max(np.abs(A.T @ r['residual'])):.2e}")

    # ---- 2b) A's 3-point calibration vs C's least squares vs the true coefficients ----
    a_file = os.path.join(HERE, "..", "person-a", "person_A_coefficients.csv")
    if os.path.exists(a_file) and all(f"true_{a}" in data for a in "xyz"):
        with open(a_file, newline="") as f:
            a_coef = {row["axis"].lower(): [float(row["c0"]), float(row["c1"]), float(row["c2"])]
                      for row in csv.DictReader(f)}
        print("\n2b) COEFFICIENTS [c0, c1, c2]: A (3-point calibration) vs C (least squares) vs truth")
        print("    A solves a 3x3 system from 3 noisy samples; C fits all points. Truth = fit of the hidden true path.")
        for a in "xyz":
            c_true, *_ = fit_least_squares(t, data[f"true_{a}"])
            print(f"  {a.upper()}  A: {np.round(a_coef[a], 4)}   C: {np.round(res[a]['coeffs'], 4)}   "
                  f"truth: {np.round(c_true, 4)}")
            print(f"     max |coef error|   A: {np.max(np.abs(np.array(a_coef[a]) - c_true)):.4f}   "
                  f"C: {np.max(np.abs(res[a]['coeffs'] - c_true)):.4f}")
    else:
        print("\n2b) (A's coefficients not found - run person-a/person_A.py to see the A vs C comparison)")

    # ---- 3) prediction ----
    t_future = np.linspace(t.max(), t.max() + T_EXTRAPOLATE, N_FUTURE)
    pred = {a: evaluate_poly(res[a]["coeffs"], t_future) for a in "xyz"}
    print("\n3) PREDICTION")
    print(f"observed t in [{t.min():.1f}, {t.max():.1f}] s; predicted t in "
          f"[{t_future.min():.1f}, {t_future.max():.1f}] s")
    print("predicted final position (x, y, z):",
          np.round([pred[a][-1] for a in "xyz"], 4))

    # ---- bonus validation against the simulation's hidden truth ----
    if all(f"true_{a}" in data for a in "xyz"):
        print("\n(Validation vs hidden ground truth, RMSE per axis x / y / z)")
        keys = list(res["x"]["rmse"])
        for k in keys:
            print(f"  {k:<58s} " + " / ".join(f"{res[a]['rmse'][k]:.4f}" for a in "xyz"))
        # extrapolation check: the true path is an exact quadratic -> fit it exactly
        ext = []
        for a in "xyz":
            c_true, *_ = fit_least_squares(t, data[f"true_{a}"])
            ext.append(rmse(pred[a], evaluate_poly(c_true, t_future)))
        print(f"  extrapolation RMSE on (t_max, t_max+{T_EXTRAPOLATE:g}] s: "
              + " / ".join(f"{v:.4f}" for v in ext))

    # ---- save for Person D ----
    with open(os.path.join(args.out_dir, "person_C_coefficients.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["axis", "a0", "a1", "a2", "residual_norm"])
        for a in "xyz":
            c = res[a]["coeffs"]
            w.writerow([a.upper(), c[0], c[1], c[2], res[a]["residual_norm"]])

    fit = {a: evaluate_poly(res[a]["coeffs"], t) for a in "xyz"}
    with open(os.path.join(args.out_dir, "person_C_output.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time", "proj_x", "proj_y", "proj_z", "fit_x", "fit_y", "fit_z"])
        for i in range(len(t)):
            w.writerow([t[i]] + [res[a]["fused"][i] for a in "xyz"] + [fit[a][i] for a in "xyz"])

    with open(os.path.join(args.out_dir, "person_C_prediction.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time", "pred_x", "pred_y", "pred_z"])
        for i in range(len(t_future)):
            w.writerow([t_future[i]] + [pred[a][i] for a in "xyz"])

    # ---- plots ----
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(*[res[a]["raw"][0] for a in "xyz"], ".", alpha=0.3, label="Raw GPS (receiver 1)")
    ax.plot(*[res[a]["fused"] for a in "xyz"], ".", alpha=0.6, label="Projected + fused")
    ax.plot(*[fit[a] for a in "xyz"], lw=2, label="Least-squares fit")
    ax.plot(*[pred[a] for a in "xyz"], "--", lw=2, label="Predicted")
    if all(f"true_{a}" in data for a in "xyz"):
        ax.plot(*[data[f"true_{a}"] for a in "xyz"], "k:", lw=1.5, label="Ground truth")
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.set_title("UAV trajectory: raw vs projected vs fitted vs predicted")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "person_C_trajectory.png"), dpi=170)
    plt.close(fig)

    fig, axs = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    for k, a in enumerate("xyz"):
        axs[k].plot(t, res[a]["raw"][0], ".", alpha=0.35, label="raw GPS (rx 1)")
        axs[k].plot(t, res[a]["fused"], ".", alpha=0.6, label="projected + fused")
        axs[k].plot(t, fit[a], lw=2, label="LS fit")
        axs[k].plot(t_future, pred[a], "--", lw=2, label="predicted")
        if f"true_{a}" in data:
            axs[k].plot(t, data[f"true_{a}"], "k:", label="truth")
        axs[k].set_ylabel(a)
    axs[0].legend(ncol=5, fontsize=8)
    axs[-1].set_xlabel("time (s)")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "person_C_axes.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
