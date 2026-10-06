"""
PERSON B — UAV Sensor Fusion
Rank + Basis + Orthonormal basis Q of the motion subspace   (v2: noise-aware)

Run (after person-a/generate_dataset.py):
    python person_B.py
    python person_B.py --data ../person-a/uav_sensor_data.csv --sigma ../person-a/noise_sigma.csv

Input : ../person-a/uav_sensor_data.csv (rows = timestamps, columns = channels)
        ../person-a/noise_sigma.csv     (noise std of each channel, i.e. the sensor datasheet value)
Output: person_B_orthonormal_basis.csv  (n_timestamps x r matrix Q, Q^T Q = I)  -> Person C
        person_B_scree.png              (singular values vs thresholds)

WHY v2 (see docs/DATASET_FIX.md)
--------------------------------
Noise makes every matrix full rank. With i.i.d. continuous noise on each channel the
19-column matrix has rank 19 under np.linalg.matrix_rank, under the old calculate_rank
(tol 1e-10) and under find_independent_columns (greedy "is this column independent?").
Those tests only detect round-off-level dependence (~1e-12), never noise. So the v1 chain
kept every column and the Gram-Schmidt basis spanned the noise too.

The fix is a NOISE-AWARE rank, then a basis taken from the signal part of the SVD:
  1. Whiten:  divide each column by its noise sigma, so every channel has unit-variance
     noise (the threshold theory assumes equal noise in all columns).
  2. SVD of the whitened matrix  M_w = U S V^T.
  3. Effective rank r = number of singular values above the Gavish-Donoho hard threshold
     tau = lambda*(beta) * sqrt(n_rows)   (sigma = 1 after whitening, beta = n_cols/n_rows).
  4. Q = first r left singular vectors U[:, :r]  (already orthonormal; Gram-Schmidt is then
     applied as a clean-up / demonstration and ||I - Q^T Q|| is reported).

References: Gavish & Donoho, "The Optimal Hard Threshold for Singular Values is 4/sqrt(3)",
IEEE Trans. Inf. Theory 60(8), 2014 (arXiv:1305.5870);  Marchenko-Pastur edge sigma(sqrt(m)+sqrt(n)).
"""

import argparse
import csv
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA = os.path.join(HERE, "..", "person-a", "uav_sensor_data.csv")
DEFAULT_SIGMA = os.path.join(HERE, "..", "person-a", "noise_sigma.csv")


# ----------------------------------------------------------------------
# I/O (same convention as Person A: rows = timestamps, columns = channels)
# ----------------------------------------------------------------------
def load_sensor_data(filename):
    with open(filename, "r", newline="") as f:
        rows = list(csv.DictReader(f))
    return {k: np.array([float(r[k]) for r in rows], dtype=float) for k in rows[0]}


def load_sigma(filename):
    with open(filename, "r", newline="") as f:
        return {r["channel"]: float(r["sigma"]) for r in csv.DictReader(f)}


def measurement_channels(data):
    """Every channel except time and the hidden ground truth."""
    return [c for c in data if c != "time" and not c.startswith("true_")]


# ----------------------------------------------------------------------
# LEGACY v1 functions (kept for comparison; they FAIL on noisy data)
# ----------------------------------------------------------------------
def calculate_rank(matrix, tolerance=1e-10):
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    threshold = tolerance * max(matrix.shape) * singular_values[0]
    return int(np.sum(singular_values > threshold))


def find_independent_columns(matrix, tolerance=1e-10):
    independent_indices = []
    current_rank = 0
    for i in range(matrix.shape[1]):
        if len(independent_indices) == 0:
            candidate = matrix[:, [i]]
        else:
            candidate = matrix[:, independent_indices + [i]]
        new_rank = calculate_rank(candidate, tolerance)
        if new_rank > current_rank:
            independent_indices.append(i)
            current_rank = new_rank
    return independent_indices


def gram_schmidt(matrix, tolerance=1e-10):
    orthogonal_vectors = []
    for i in range(matrix.shape[1]):
        v = matrix[:, i].astype(float).copy()
        for u in orthogonal_vectors:
            v = v - np.dot(u, v) * u
        norm = np.linalg.norm(v)
        if norm > tolerance:
            orthogonal_vectors.append(v / norm)
    return np.column_stack(orthogonal_vectors)


# ----------------------------------------------------------------------
# v2: noise-aware rank and signal basis
# ----------------------------------------------------------------------
def gd_threshold(n_rows, n_cols):
    """
    Gavish-Donoho optimal hard threshold for singular values, noise sigma = 1.

    For an m x n matrix with beta = min(m,n)/max(m,n):
        tau = lambda*(beta) * sqrt(max(m,n)),
        lambda*(beta) = sqrt(2(beta+1) + 8 beta / ((beta+1) + sqrt(beta^2 + 14 beta + 1))).
    NOTE: the sqrt uses the LARGER dimension (timestamps), not the number of channels.
    """
    big, small = max(n_rows, n_cols), min(n_rows, n_cols)
    beta = small / big
    lam = np.sqrt(2 * (beta + 1) + 8 * beta / ((beta + 1) + np.sqrt(beta ** 2 + 14 * beta + 1)))
    return float(lam * np.sqrt(big))


def mp_edge(n_rows, n_cols):
    """Marchenko-Pastur upper edge of the noise singular values, sigma = 1."""
    return float(np.sqrt(n_rows) + np.sqrt(n_cols))


def whiten(matrix, col_sigma):
    return matrix / np.asarray(col_sigma, dtype=float)[None, :]


def noise_aware_rank(matrix, col_sigma):
    """Effective rank: whitened singular values above the Gavish-Donoho threshold."""
    s = np.linalg.svd(whiten(matrix, col_sigma), compute_uv=False)
    return int(np.sum(s > gd_threshold(*matrix.shape)))


def signal_basis(matrix, col_sigma):
    """
    Orthonormal basis Q (n_rows x r) of the signal subspace, and the rank r.
    Q = top-r left singular vectors of the whitened matrix; Gram-Schmidt applied as a clean-up.
    """
    Mw = whiten(matrix, col_sigma)
    U, s, _ = np.linalg.svd(Mw, full_matrices=False)
    r = int(np.sum(s > gd_threshold(*matrix.shape)))
    if r == 0:
        raise ValueError("No singular value exceeds the noise threshold: no recoverable signal.")
    return gram_schmidt(U[:, :r]), r


def principal_angle_sine(Q, t):
    """
    VALIDATION ONLY (uses the known simulation model): sine of the largest principal angle
    between span(Q) and the true motion subspace span{1, t, t^2}. 0 = Q lies in the subspace.
    """
    V = np.column_stack([np.ones_like(t), t, t ** 2])
    Qv, _ = np.linalg.qr(V)
    return float(np.linalg.norm(Q - Qv @ (Qv.T @ Q), 2))


def main():
    if hasattr(sys.stdout, "reconfigure"):   # Windows consoles/pipes may default to a legacy codepage
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description="Person B: noise-aware rank + orthonormal basis")
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--sigma", default=DEFAULT_SIGMA)
    ap.add_argument("--out-dir", default=HERE)
    args = ap.parse_args()

    for p in (args.data, args.sigma):
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"{p} not found. Run `python person-a/generate_dataset.py` first.")

    data = load_sensor_data(args.data)
    sigma = load_sigma(args.sigma)
    channels = measurement_channels(data)
    M = np.column_stack([data[c] for c in channels])
    col_sigma = np.array([sigma[c] for c in channels])
    m, n = M.shape

    print("=" * 70)
    print("UAV SENSOR FUSION — PERSON B (rank, basis, Gram-Schmidt)")
    print("=" * 70)
    print(f"\n0) SENSOR MATRIX: {m} timestamps x {n} channels")
    print("Channels:", channels)

    # ---- legacy tests: why they cannot work on noisy data ----
    print("\n1) LEGACY RANK TESTS ON THE NOISY MATRIX (detect round-off, not noise)")
    print("np.linalg.matrix_rank (default tol):", np.linalg.matrix_rank(M))
    print("calculate_rank (tol 1e-10)         :", calculate_rank(M))
    print("find_independent_columns keeps     :", len(find_independent_columns(M)), "of", n, "columns")
    print("-> every channel looks 'independent' because independent noise was added to each.")

    # ---- noise-aware rank ----
    Mw = whiten(M, col_sigma)
    s = np.linalg.svd(Mw, compute_uv=False)
    tau = gd_threshold(m, n)
    edge = mp_edge(m, n)
    r = int(np.sum(s > tau))

    print("\n2) NOISE-AWARE RANK (columns divided by their noise sigma)")
    print(f"Marchenko-Pastur noise edge sqrt(m)+sqrt(n) = {edge:.2f}")
    print(f"Gavish-Donoho threshold tau                = {tau:.2f}")
    print("\n  k   singular value   > tau")
    for k, sv in enumerate(s[:min(8, n)], start=1):
        print(f"{k:3d}   {sv:14.3f}   {'YES' if sv > tau else 'no'}")
    if n > 8:
        print(f"  ... {n - 8} more, all <= {s[8]:.3f}")
    print(f"\nEffective (signal) rank r = {r}   [legacy tests said {n}]")

    # ---- basis ----
    Q, r = signal_basis(M, col_sigma)
    dev = float(np.max(np.abs(Q.T @ Q - np.eye(r))))
    print(f"\n3) ORTHONORMAL BASIS Q: shape {Q.shape}")
    print(f"max |Q^T Q - I| = {dev:.2e}")
    print(f"[validation, uses known model] sin(largest principal angle to span{{1,t,t^2}}) = "
          f"{principal_angle_sine(Q, data['time']):.4f}  (0 = perfect)")

    out_q = os.path.join(args.out_dir, "person_B_orthonormal_basis.csv")
    np.savetxt(out_q, Q, delimiter=",")
    print(f"\nSaved {out_q}  -> Person C")

    # ---- scree plot ----
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.semilogy(np.arange(1, n + 1), s, "o-", label="whitened singular values")
    ax.axhline(tau, color="r", ls="--", label=f"Gavish-Donoho tau = {tau:.1f}")
    ax.axhline(edge, color="gray", ls=":", label=f"Marchenko-Pastur edge = {edge:.1f}")
    ax.axhline(np.sqrt(np.finfo(float).eps) * s[0], color="k", ls="-.", alpha=0.4,
               label="legacy-style tolerance (~round-off)")
    ax.set_xlabel("k")
    ax.set_ylabel("singular value (log scale)")
    ax.set_title(f"Scree plot: effective rank r = {r} (matrix has {n} columns)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "person_B_scree.png"), dpi=160)


if __name__ == "__main__":
    main()
