"""
PERSON D - UAV Sensor Fusion
PCA / Eigen-analysis + Diagonalization + final comparison

Run (after A, B, C):   python person_D.py

PIPELINE OF THIS FILE (Concept -> Purpose -> Outcome)
  1. Whiten + center the 19-channel matrix  -> put all channels on the same noise scale
  2. Covariance matrix  Sigma = Xc^T Xc / (n-1)   (19 x 19, symmetric)
  3. Eigendecomposition (eigh)                    -> eigenvalues = variance along each direction
  4. Diagonalization  Sigma = V Lambda V^T        -> verify numerically
  5. Pick k from the gap in the spectrum          -> signal vs noise directions
  6. Project onto top-k eigenvectors              -> second denoising pass
  7. Compare against ground truth (RMSE)
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
DATA = os.path.join(HERE, "..", "person-a", "uav_sensor_data.csv")
SIGMA = os.path.join(HERE, "..", "person-a", "noise_sigma.csv")
C_OUTPUT = os.path.join(HERE, "..", "person-c", "person_C_output.csv")


# ----------------------------------------------------------------------
# Plumbing
# ----------------------------------------------------------------------
def load_table(filename):
    with open(filename, "r", newline="") as f:
        rows = list(csv.DictReader(f))
    return {k: np.array([float(r[k]) for r in rows], dtype=float) for k in rows[0]}


def load_sigma(filename):
    with open(filename, "r", newline="") as f:
        return {r["channel"]: float(r["sigma"]) for r in csv.DictReader(f)}


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


# ----------------------------------------------------------------------
# Linear algebra steps
# ----------------------------------------------------------------------
def center(X):
    """Subtract each column's mean so the data cloud is centred at the origin.
    PCA measures variation AROUND the mean, so this is required."""
    mean = X.mean(axis=0)
    return X - mean, mean


def covariance_matrix(Xc):
    """Sigma = Xc^T Xc / (n-1).  n-1 (not n) because the mean was estimated from the same
    data, which uses up one degree of freedom (unbiased estimate)."""
    n = Xc.shape[0]
    return (Xc.T @ Xc) / (n - 1)


def eigendecompose(Sigma):
    """eigh is for symmetric matrices: real eigenvalues, orthogonal eigenvectors.
    Returns eigenvalues sorted DESCENDING and the matching eigenvectors (as columns)."""
    lam, V = np.linalg.eigh(Sigma)
    order = np.argsort(lam)[::-1]
    return lam[order], V[:, order]


def check_diagonalization(Sigma, V, lam):
    """Verify Sigma = V diag(lam) V^T and V^T V = I."""
    recon_err = np.max(np.abs(Sigma - V @ np.diag(lam) @ V.T))
    orth_err = np.max(np.abs(V.T @ V - np.eye(V.shape[1])))
    return recon_err, orth_err


def explained_variance(lam):
    """Fraction of total variance per eigenvalue, plus cumulative."""
    lam = np.clip(lam, 0, None)
    frac = lam / lam.sum()
    return frac, np.cumsum(frac)


def choose_k(lam):
    """Pick k = number of eigenvalues clearly above the noise bulk.
    Whitened noise has variance ~1 per channel, so eigenvalues well above that are signal.
    Rule: keep eigenvalues > 5x the median eigenvalue (the median is almost pure noise)."""
    thr = 5.0 * np.median(lam)
    return int(np.sum(lam > thr)), thr


def project_top_k(Xc, V, k, mean):
    """Second-pass denoising: Xk = Xc V_k V_k^T + mean  (orthogonal projection onto top-k PCs)."""
    Vk = V[:, :k]
    return Xc @ Vk @ Vk.T + mean


def quad_fit(t, y):
    """Least-squares quadratic through y(t) (same model as Person C)."""
    A = np.column_stack([np.ones_like(t), t, t ** 2])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return A @ coef


# ----------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------
def make_figures(t, lam, k, truth, raw, pca, lsq, pca_ls, outdir):
    # 1. eigenvalue spectrum
    fig, ax = plt.subplots(figsize=(8, 4.5))
    idx = np.arange(1, len(lam) + 1)
    colors = ["tab:green" if i <= k else "tab:gray" for i in idx]
    ax.bar(idx, np.clip(lam, 1e-3, None), color=colors)
    ax.set_yscale("log")
    ax.set_xticks(idx)
    ax.set_xlabel("Principal component")
    ax.set_ylabel("Eigenvalue (log scale)")
    ax.set_title(f"Eigenvalue spectrum - green = kept (k={k}), grey = discarded as noise")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "person_D_scree.png"), dpi=150)
    plt.close(fig)

    # 2. residuals per axis: (estimate - truth). The path is ~25 units long but errors are ~0.01-0.1,
    #    so a plain overlay hides everything; plotting the DIFFERENCE from truth makes the cleaning visible.
    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    for j, name in enumerate("xyz"):
        axes[j].axhline(0, c="k", ls="--", lw=1, label="truth (zero error)")
        axes[j].plot(t, raw[:, j] - truth[:, j], c="lightgray", lw=1, label="raw GPS")
        axes[j].plot(t, pca[:, j] - truth[:, j], c="tab:orange", lw=1.3, label="PCA-filtered")
        axes[j].plot(t, lsq[:, j] - truth[:, j], c="tab:blue", lw=1.3, label="Person C least squares")
        axes[j].set_ylabel(f"{name} error")
    axes[0].legend(ncol=4, fontsize=8, loc="upper center")
    axes[0].set_title("Estimate minus truth, per axis (closer to 0 is better)")
    axes[2].set_xlabel("time (s)")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "person_D_trajectory.png"), dpi=150)
    plt.close(fig)

    # 3. error over time
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for name, p, c in (("raw GPS", raw, "lightgray"), ("PCA k=%d" % k, pca, "tab:orange"),
                       ("C least squares", lsq, "tab:blue"), ("PCA + LS", pca_ls, "tab:green")):
        ax.plot(t, np.linalg.norm(p - truth, axis=1), label=name, c=c)
    ax.set_xlabel("time (s)"); ax.set_ylabel("position error (distance to truth)")
    ax.set_title("Error vs time (lower is better)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "person_D_error.png"), dpi=150)
    plt.close(fig)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description="Person D: PCA / eigen-analysis / comparison")
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--sigma", default=SIGMA)
    ap.add_argument("--c-output", default=C_OUTPUT)
    args = ap.parse_args()

    for p, hint in ((args.data, "person-a/generate_dataset.py"),
                    (args.c_output, "person-c/person_C.py")):
        if not os.path.exists(p):
            raise FileNotFoundError(f"{p} not found. Run `python {hint}` first (or run_pipeline.py).")

    data = load_table(args.data)
    sigma = load_sigma(args.sigma)
    c_out = load_table(args.c_output)
    t = data["time"]
    channels = list(sigma)
    gps = ["gps_x", "gps_y", "gps_z"]
    gi = [channels.index(c) for c in gps]

    print("=" * 70)
    print("UAV SENSOR FUSION - PERSON D (PCA, diagonalization, comparison)")
    print("=" * 70)

    # 0) data matrix
    X = np.column_stack([data[c] for c in channels])          # T x 19
    sg = np.array([sigma[c] for c in channels])
    truth = np.column_stack([data["true_x"], data["true_y"], data["true_z"]])
    raw = X[:, gi]
    print(f"\n0) DATA MATRIX: {X.shape[0]} timestamps x {X.shape[1]} channels")
    print("   Channels have different noise levels (GPS 0.10, baro 0.20, vel 0.05, acc 0.015, gyro 0.01),")
    print("   so each column is divided by its sigma (whitening). Then every channel has noise variance 1.")

    # 1) whiten + center
    W = X / sg
    Wc, mean = center(W)
    print(f"\n1) CENTRED + WHITENED matrix: shape {Wc.shape}, max |column mean| = {np.abs(Wc.mean(axis=0)).max():.1e}")

    # 2) covariance
    Sigma = covariance_matrix(Wc)
    print(f"\n2) COVARIANCE MATRIX Sigma = Xc^T Xc/(n-1): shape {Sigma.shape}")
    print(f"   symmetric? max |Sigma - Sigma^T| = {np.max(np.abs(Sigma - Sigma.T)):.1e}")

    # 3) eigen
    lam, V = eigendecompose(Sigma)
    frac, cum = explained_variance(lam)
    print("\n3) EIGENVALUES (variance along each principal direction)")
    print("    PC    eigenvalue    % variance   cumulative %")
    for i in range(8):
        print(f"   {i + 1:3d}   {lam[i]:11.3f}   {100 * frac[i]:9.4f}   {100 * cum[i]:10.4f}")
    print(f"   ... remaining {len(lam) - 8} eigenvalues all <= {lam[8]:.3f}")

    # 4) diagonalization
    recon_err, orth_err = check_diagonalization(Sigma, V, lam)
    print("\n4) DIAGONALIZATION  Sigma = V Lambda V^T")
    print(f"   max |Sigma - V diag(lam) V^T| = {recon_err:.2e}   (relative to max eigenvalue: {recon_err / lam[0]:.1e})")
    print(f"   max |V^T V - I|              = {orth_err:.2e}   (eigenvectors are orthonormal)")
    print(f"   smallest eigenvalue = {lam[-1]:.3f} (all >= 0: Sigma is positive semi-definite)")

    # 5) choose k
    k, thr = choose_k(lam)
    print(f"\n5) CHOOSING k: noise bulk median = {np.median(lam):.2f}; keep eigenvalues > {thr:.2f}  ->  k = {k}")
    print("   True rank is 3, but centering removes the constant direction, so the centred signal")
    print("   spans {t, t^2}: only 2 large eigenvalues are expected.")
    print(f"   Found k = {k}: " + ("matches the expectation." if k == 2 else "DIFFERS from the expected 2 (check seed / data)."))
    print(f"   First {k} components explain {100 * cum[k - 1]:.3f}% of total variance.")

    # 6) project
    Wk = project_top_k(Wc, V, k, mean)
    Xk = Wk * sg                                   # back to original units
    pca = Xk[:, gi]
    print(f"\n6) TOP-k PROJECTION: rebuilt all 19 channels from {k} components; GPS columns = PCA-filtered path")

    # 7) comparison
    lsq = np.column_stack([c_out["fit_x"], c_out["fit_y"], c_out["fit_z"]])
    pca_ls = np.column_stack([quad_fit(t, pca[:, j]) for j in range(3)])
    print("\n7) COMPARISON vs HIDDEN GROUND TRUTH (RMSE over x, y, z)")
    rows = [("raw GPS (1 receiver)", raw),
            (f"PCA-filtered (k={k}, 19 channels)", pca),
            ("Person C: projection + avg + least squares", lsq),
            (f"PCA (k={k}) then least squares  [extra]", pca_ls)]
    base = rmse(raw, truth)
    for name, p in rows:
        e = rmse(p, truth)
        note = "(baseline)" if p is raw else f"({base / e:4.1f}x better than raw)"
        print(f"   {name:46s} {e:.4f}   {note}")

    print("\n   k sweep (PCA only) - why k matters:")
    for kk in range(1, 6):
        e = rmse(project_top_k(Wc, V, kk, mean)[:, gi] * sg[gi], truth)
        print(f"     k = {kk}: RMSE {e:.4f}" + ("   <- chosen" if kk == k else ""))

    # 8) 3x3 covariance of C's fitted path
    Fc, _ = center(lsq)
    lam3, V3 = eigendecompose(covariance_matrix(Fc))
    print("\n8) BONUS: eigen-analysis of C's fitted path (3x3 covariance of x, y, z)")
    print(f"   eigenvalues = {np.array2string(lam3, precision=4)}")
    print(f"   {100 * lam3[0] / lam3.sum():.2f}% of the path variance lies along ONE direction "
          f"{np.round(V3[:, 0], 3)} = the direction of travel.")

    out = os.path.dirname(os.path.abspath(__file__))
    make_figures(t, lam, k, truth, raw, pca, lsq, pca_ls, out)
    np.savetxt(os.path.join(out, "person_D_pca_path.csv"), np.column_stack([t, pca, pca_ls]),
               delimiter=",", header="time,pca_x,pca_y,pca_z,pcals_x,pcals_y,pcals_z", comments="")
    print("\nSaved person_D_scree.png, person_D_trajectory.png, person_D_error.png, person_D_pca_path.csv")

    e_raw, e_pca, e_lsq = rmse(raw, truth), rmse(pca, truth), rmse(lsq, truth)
    print("\n" + "=" * 70)
    print("SUMMARY (Concept -> Purpose -> Outcome)")
    print("=" * 70)
    print(f"  Covariance : shows how channels vary together  -> {Sigma.shape[0]}x{Sigma.shape[1]} symmetric matrix")
    print(f"  Eigenvalues: separate signal from noise        -> {k} large eigenvalues "
          f"({lam[0]:.0f}, {lam[1]:.1f}), rest near noise level (<= {lam[k]:.1f})")
    print(f"  Diagonalize: Sigma = V Lambda V^T              -> verified, error {recon_err:.1e}")
    print(f"  Top-k      : discard noise directions          -> RMSE {e_raw:.3f} -> {e_pca:.3f}")
    if e_lsq < e_pca:
        print(f"  Note       : Person C's least squares ({e_lsq:.3f}) beats PCA alone ({e_pca:.3f}) because the")
        print("               quadratic fit also forces smoothness in TIME, which PCA does not.")
    else:
        print(f"  Note       : PCA ({e_pca:.3f}) beats Person C's least squares ({e_lsq:.3f}) on this seed.")

if __name__ == "__main__":
    main()