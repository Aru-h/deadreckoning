"""
PERSON A — Dataset generator (v2, low-rank signal + independent noise)

Run:
    python generate_dataset.py                 # writes the shared CSV + sigma table here
    python generate_dataset.py --seed 7 --n-rx 6

WHY THIS EXISTS (see docs/DATASET_FIX.md for the full story)
-----------------------------------------------------------
The v1 dataset had 9 channels that were independent noise streams around a
quadratic path, so the sensor matrix was numerically full rank (9). Rank,
Gram-Schmidt and projection P = QQ^T then had nothing to remove:
  * a basis built from all 9 channels contains GPS itself -> P is the identity on GPS
  * an IMU-only basis spans {1, t}, not the quadratic position -> projection destroys the signal

v2 fixes the data, not the maths. Every CLEAN channel is a linear function of the
three trajectory coefficients (a0, a1, a2) per axis, so the clean sensor matrix
has rank 3 (every clean column lies in span{1, t, t^2}). Independent noise is then
added to each channel. The noisy matrix is still full rank, but its singular values
show a clear gap after the 3rd one (given enough curvature) -- that gap is what
Person B's noise-aware rank step detects.

Design choices
  * 3 GPS receivers per axis (gps_*, gps2_*, gps3_*): redundant, independent noise.
  * baro_z: a second, noisier measurement of z.
  * vel_*: velocity channels, V(t) = a1 + 2 a2 t   (in span{1, t}).
  * acc_*: acceleration channels, A = 2 a2          (in span{1}).
  * gyro_x, gyro_y: level flight -> zero-mean noise only.
  * gyro_z: yaw rate of a polynomial heading psi(t) = p0 + p1 t + p2 t^2,
    i.e. p1 + 2 p2 t (in span{1, t}). The v1 gyro was pure noise, which is unphysical
    for an accelerating, curving path.
  * Curvature a2 = (0.15, 0.10, 0.08): the v1 values (0.05/0.03/0.01) made the 3rd
    clean singular value (0.34) smaller than the whitened noise bulk (~1), so rank 3
    was undetectable by any threshold rule.

Outputs (next to this script by default)
  uav_sensor_data.csv   time + measurement channels + hidden true_x/true_y/true_z
  noise_sigma.csv       channel,sigma  (the noise std used for each measurement channel)
                        -- Person B needs this to whiten the columns before the rank test.

Limitations (not tested): correlated receiver errors, GPS outliers, biased noise.
Real receivers share multipath/ionospheric errors, so real redundancy is weaker than this.
"""

import argparse
import csv
import hashlib
import os
import zlib

import numpy as np

# a0, a1, a2 per axis: position(t) = a0 + a1 t + a2 t^2
COEF = {"x": (2.0, 0.8, 0.15), "y": (1.0, 0.5, 0.10), "z": (5.0, 0.2, 0.08)}


def seed_to_int(seed):
    """
    Minecraft-style seed: an integer, or any word/phrase (hashed with CRC32, which is stable
    across machines -- Python's own hash() is randomised per run and must NOT be used).
    """
    try:
        return int(seed) % (2 ** 32)
    except (TypeError, ValueError):
        return zlib.crc32(str(seed).encode("utf-8"))


def fingerprint(path):
    """Short SHA-256 of a file: two teammates with the same fingerprint have the same dataset."""
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:12]


def generate(seed=42, n=101, dt=0.1,
             sig_gps=0.10, sig_vel=0.05, sig_acc=0.015, sig_gyro=0.01, sig_baro=0.20,
             n_rx=3, yaw=(0.0, 0.05, 0.01), coef=COEF):
    """
    Return (cols, sigma): ordered dict of channel -> array, and channel -> noise std.

    Column order: time, original v1 nine channels (gps_*, acc_*, gyro_*), then the new
    redundant channels (gps2_*, gps3_*, ..., baro_z, vel_*), then the hidden truth.
    """
    # Legacy RandomState on purpose: NumPy guarantees its stream is frozen across versions and
    # platforms, so the same seed gives the same CSV on any laptop (default_rng does not promise this).
    rng = np.random.RandomState(seed_to_int(seed))
    t = np.arange(n) * dt

    pos = {a: c[0] + c[1] * t + c[2] * t ** 2 for a, c in coef.items()}
    vel = {a: c[1] + 2 * c[2] * t for a, c in coef.items()}
    acc = {a: 2 * c[2] + 0 * t for a, c in coef.items()}

    cols, sigma = {"time": t}, {}

    def add(name, clean, s):
        cols[name] = clean + rng.normal(0.0, s, n)
        sigma[name] = s

    for a in "xyz":                                  # original v1 column names first
        add(f"gps_{a}", pos[a], sig_gps)
    for a in "xyz":
        add(f"acc_{a}", acc[a], sig_acc)
    add("gyro_x", 0 * t, sig_gyro)                   # level flight
    add("gyro_y", 0 * t, sig_gyro)
    add("gyro_z", yaw[1] + 2 * yaw[2] * t, sig_gyro)  # yaw rate of polynomial heading

    for k in range(2, n_rx + 1):                     # extra redundant GPS receivers
        for a in "xyz":
            add(f"gps{k}_{a}", pos[a], sig_gps)
    add("baro_z", pos["z"], sig_baro)
    for a in "xyz":
        add(f"vel_{a}", vel[a], sig_vel)

    for a in "xyz":                                   # hidden ground truth (not a sensor)
        cols[f"true_{a}"] = pos[a]
    return cols, sigma


def write_dataset(cols, sigma, csv_path, sigma_path):
    names = list(cols)
    n = len(cols["time"])
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(names)
        for i in range(n):
            w.writerow([repr(float(cols[k][i])) for k in names])
    with open(sigma_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["channel", "sigma"])
        for k, s in sigma.items():
            w.writerow([k, s])


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--seed", default="42",
                    help="the dataset seed, like a Minecraft world seed: an integer or any word. "
                         "Same seed (+ same --n-rx) => identical dataset on any laptop. "
                         "Default 42. Use 'random' for a fresh seed (it is printed, so the run "
                         "can be reproduced).")
    ap.add_argument("--n-rx", type=int, default=3, help="GPS receivers per axis (default 3)")
    ap.add_argument("--out-dir", default=here)
    args = ap.parse_args()

    if args.seed == "random":
        seed = int(np.random.SeedSequence().entropy % (2 ** 31))
    else:
        seed = args.seed
    print(f"Seed: {seed}   (share it: `python generate_dataset.py --seed {seed} --n-rx {args.n_rx}` "
          "recreates exactly this dataset)")
    cols, sigma = generate(seed=seed, n_rx=args.n_rx)
    csv_path = os.path.join(args.out_dir, "uav_sensor_data.csv")
    sigma_path = os.path.join(args.out_dir, "noise_sigma.csv")
    write_dataset(cols, sigma, csv_path, sigma_path)

    n_meas = len(sigma)
    print(f"Wrote {csv_path}  ({len(cols['time'])} rows, {n_meas} measurement channels)")
    print(f"Wrote {sigma_path}")
    print(f"Dataset fingerprint: {fingerprint(csv_path)}  (teammates with the same seed and "
          "--n-rx must see this exact value)")
    print("Clean matrix rank is 3 by construction; the noisy matrix is full rank "
          f"({n_meas}) -- see person-b for the noise-aware rank.")


if __name__ == "__main__":
    main()
