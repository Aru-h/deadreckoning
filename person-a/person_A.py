"""
PERSON A — UAV Sensor Fusion
Matrix Representation + RREF + LU Decomposition

Run:
    python generate_dataset.py     # (re)creates uav_sensor_data.csv + noise_sigma.csv
    python person_A.py             # add --show to open the plot window

This file is intentionally self-contained. It:
1. Loads the provided UAV sensor CSV.
2. Displays the sensor matrix.
3. Builds a quadratic trajectory model x(t)=c0+c1*t+c2*t^2.
4. Builds A and b for three calibration timestamps.
5. Solves [A|b] using RREF.
6. Solves the same system using LU decomposition.
7. Compares the two solutions.
8. Plots the noisy GPS data and the calibration-model trajectory.

The later team members can use the same CSV and the coefficient outputs.

CHANGES vs v1 (original kept in legacy/person_A_v1.py) -- see docs/DATASET_FIX.md:
  * The sensor matrix is now built from every measurement channel in the CSV
    (19 columns: GPS x3 receivers, barometer, velocity, accel, gyro) instead of a
    hard-coded list of 9. Column names of the original 9 are unchanged.
  * The RREF / LU calibration (3x3 system on gps_x/y/z at t = 0, 5, 10 s) is unchanged.
  * Paths are relative to this file, and the plot only opens with --show.
"""

import csv
import os
import sys

import matplotlib

if "--show" not in sys.argv:
    matplotlib.use("Agg")          # headless-safe: save the PNG without opening a window
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import lu, solve_triangular


HERE = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(HERE, "uav_sensor_data.csv")


def load_sensor_data(filename):
    """Load the CSV into a dictionary of NumPy arrays."""
    with open(filename, "r", newline="") as f:
        rows = list(csv.DictReader(f))

    data = {}
    for key in rows[0].keys():
        data[key] = np.array([float(row[key]) for row in rows], dtype=float)
    return data


def build_design_matrix(times):
    """
    Quadratic motion model:
        position(t) = c0 + c1*t + c2*t^2

    Therefore each row of A is:
        [1, t, t^2]
    """
    times = np.asarray(times, dtype=float)
    return np.column_stack((np.ones_like(times), times, times**2))


def rref(matrix, tol=1e-12):
    """
    Compute reduced row echelon form using elementary row operations.
    Returns a new matrix; the input is not modified.
    """
    M = np.array(matrix, dtype=float, copy=True)
    rows, cols = M.shape
    pivot_row = 0

    for pivot_col in range(cols):
        if pivot_row >= rows:
            break

        # Find the row with the largest pivot candidate.
        best = pivot_row + np.argmax(np.abs(M[pivot_row:, pivot_col]))

        if abs(M[best, pivot_col]) <= tol:
            continue

        # Swap into pivot position.
        if best != pivot_row:
            M[[pivot_row, best]] = M[[best, pivot_row]]

        # Make pivot equal to 1.
        M[pivot_row] = M[pivot_row] / M[pivot_row, pivot_col]

        # Eliminate this column from every other row.
        for r in range(rows):
            if r != pivot_row:
                factor = M[r, pivot_col]
                if abs(factor) > tol:
                    M[r] = M[r] - factor * M[pivot_row]

        pivot_row += 1

    M[np.abs(M) < tol] = 0.0
    return M


def solve_using_rref(A, b):
    """Solve Ax=b by RREF of the augmented matrix [A|b]."""
    augmented = np.column_stack((A, b))
    R = rref(augmented)

    # For our 3x3 calibration system, R should be [I | c].
    coefficients = R[:, -1]
    return R, coefficients


def solve_using_lu(A, b):
    """
    scipy.linalg.lu returns A = P @ L @ U.

    Therefore:
        P L U c = b
        L U c = P.T b
    """
    P, L, U = lu(A)

    pb = P.T @ b

    # Forward substitution: L y = P.T b
    y = solve_triangular(L, pb, lower=True)

    # Back substitution: U c = y
    c = solve_triangular(U, y)

    return P, L, U, c


def print_matrix(name, M, precision=5):
    print(f"\n{name} =")
    print(np.array2string(
        np.asarray(M),
        precision=precision,
        suppress_small=True
    ))


def main():
    if hasattr(sys.stdout, "reconfigure"):   # Windows consoles/pipes may default to a legacy codepage
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(
            f"{DATA_FILE} not found. Keep person_A.py and uav_sensor_data.csv "
            "in the same folder."
        )

    data = load_sensor_data(DATA_FILE)

    # ----------------------------------------------------------
    # STAGE 1 — REAL-WORLD DATA / MATRIX REPRESENTATION
    # ----------------------------------------------------------
    # Every measurement channel in the CSV (everything except time and the hidden truth).
    sensor_columns = [c for c in data if c != "time" and not c.startswith("true_")]

    sensor_matrix = np.column_stack([data[c] for c in sensor_columns])

    print("=" * 70)
    print("UAV SENSOR FUSION — PERSON A")
    print("=" * 70)

    print("\n1) SENSOR DATA MATRIX")
    print("Rows = timestamps")
    print("Columns = sensor channels")
    print("Column order:", sensor_columns)
    print("Matrix shape:", sensor_matrix.shape)
    print("\nFirst 5 rows:")
    print(np.array2string(sensor_matrix[:5], precision=4, suppress_small=True))

    # ----------------------------------------------------------
    # STAGE 2 — MATRIX REPRESENTATION / MOTION MODEL
    # ----------------------------------------------------------
    # Three calibration timestamps make a square 3x3 system.
    calibration_indices = [0, 50, 100]
    t_cal = data["time"][calibration_indices]

    A = build_design_matrix(t_cal)

    bx = data["gps_x"][calibration_indices]
    by = data["gps_y"][calibration_indices]
    bz = data["gps_z"][calibration_indices]

    print("\n2) CALIBRATION / MOTION MODEL")
    print("Assumed model:")
    print("    position(t) = c0 + c1*t + c2*t^2")
    print("Therefore each row of A is [1, t, t^2].")

    print_matrix("Calibration times", t_cal)
    print_matrix("A", A)
    print_matrix("b_x (GPS X)", bx)
    print_matrix("b_y (GPS Y)", by)
    print_matrix("b_z (GPS Z)", bz)

    # ----------------------------------------------------------
    # STAGE 3 — RREF / MATRIX SIMPLIFICATION
    # ----------------------------------------------------------
    print("\n3) RREF — MATRIX SIMPLIFICATION")

    results = {}

    for axis, b in [("X", bx), ("Y", by), ("Z", bz)]:
        R, coeff = solve_using_rref(A, b)
        results[axis] = {"rref": R, "rref_coeff": coeff}

        print(f"\n--- {axis} direction ---")
        print_matrix("[A | b] before RREF", np.column_stack((A, b)))
        print_matrix("RREF([A | b])", R)
        print(f"RREF trajectory coefficients [c0, c1, c2] for {axis}:")
        print(np.round(coeff, 6))

    # ----------------------------------------------------------
    # STAGE 4 — LU DECOMPOSITION
    # ----------------------------------------------------------
    print("\n4) LU DECOMPOSITION")
    P, L, U, _ = solve_using_lu(A, bx)

    print_matrix("P (permutation)", P)
    print_matrix("L (lower triangular)", L)
    print_matrix("U (upper triangular)", U)

    print("\nVerification of A = P @ L @ U:")
    print("Maximum absolute reconstruction error:",
          np.max(np.abs(A - P @ L @ U)))

    # Reuse same LU factors for X, Y, Z.
    for axis, b in [("X", bx), ("Y", by), ("Z", bz)]:
        P2, L2, U2, coeff_lu = solve_using_lu(A, b)
        results[axis]["lu_coeff"] = coeff_lu

        residual = A @ coeff_lu - b
        results[axis]["residual"] = residual

        print(f"\n{axis} direction LU coefficients:")
        print(np.round(coeff_lu, 6))
        print("Residual A*c - b:")
        print(np.round(residual, 10))

    # ----------------------------------------------------------
    # STAGE 5 — RREF VS LU COMPARISON
    # ----------------------------------------------------------
    print("\n5) RREF VS LU COMPARISON")
    for axis in ["X", "Y", "Z"]:
        cr = results[axis]["rref_coeff"]
        cl = results[axis]["lu_coeff"]
        diff = cr - cl

        print(f"\n{axis}:")
        print("RREF coefficients:", np.round(cr, 6))
        print("LU coefficients:  ", np.round(cl, 6))
        print("Difference:       ", np.round(diff, 10))
        print("Max difference:    ", f"{np.max(np.abs(diff)):.3e}")

    # ----------------------------------------------------------
    # STAGE 6 — MODELLED CALIBRATION TRAJECTORY
    # ----------------------------------------------------------
    tx = data["time"]
    coeff_x = results["X"]["lu_coeff"]
    coeff_y = results["Y"]["lu_coeff"]
    coeff_z = results["Z"]["lu_coeff"]

    model_x = coeff_x[0] + coeff_x[1]*tx + coeff_x[2]*tx**2
    model_y = coeff_y[0] + coeff_y[1]*tx + coeff_y[2]*tx**2
    model_z = coeff_z[0] + coeff_z[1]*tx + coeff_z[2]*tx**2

    print("\n6) FINAL PERSON-A OUTPUT")
    print("Calibration trajectory model coefficients:")
    print("X:", np.round(coeff_x, 6))
    print("Y:", np.round(coeff_y, 6))
    print("Z:", np.round(coeff_z, 6))

    # Save coefficients for the rest of the team.
    with open(os.path.join(HERE, "person_A_coefficients.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["axis", "c0", "c1", "c2", "method"])
        for axis in ["X", "Y", "Z"]:
            c = results[axis]["lu_coeff"]
            w.writerow([axis, c[0], c[1], c[2], "LU"])

    # ----------------------------------------------------------
    # VISUAL DEMO
    # ----------------------------------------------------------
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(
        data["gps_x"], data["gps_y"], data["gps_z"],
        ".", alpha=0.35, label="Noisy GPS"
    )
    ax.plot(
        model_x, model_y, model_z,
        linewidth=2, label="Person-A calibration model"
    )

    ax.set_xlabel("X position")
    ax.set_ylabel("Y position")
    ax.set_zlabel("Z position")
    ax.set_title("UAV Trajectory: Noisy GPS vs Calibration Model")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(HERE, "person_A_trajectory.png"), dpi=180)
    if "--show" in sys.argv:
        plt.show()


if __name__ == "__main__":
    main()
