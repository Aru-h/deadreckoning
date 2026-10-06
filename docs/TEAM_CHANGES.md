# Team changes for the dataset fix — who does what

Background and numbers: [DATASET_FIX.md](DATASET_FIX.md). Everything below runs with one command:
`python run_pipeline.py` (add `--seed <int|word>` / `--n-rx <k>`; default seed 42, 3 receivers).

## Agree first (whole team, ~1 hr)
* Everyone uses the **same seed and `--n-rx`** (default `--seed 42 --n-rx 3`). No CSV is committed;
  compare the printed **dataset fingerprint** to confirm you have identical data.
* The contract between stages is two files made by Person A: `uav_sensor_data.csv` (columns keep the v1 names
  `gps_*`, `acc_*`, `gyro_*`, plus new `gps2_*`, `gps3_*`, `baro_z`, `vel_*`, and hidden `true_*`) and
  `noise_sigma.csv` (`channel,sigma`).
* Curvature `a₂ = (0.15, 0.10, 0.08)` is part of the design — the v1 values are too weak for rank 3 to be detectable.

## Person A — data (≈ +1–2 h, easy)
* Run `python person-a/generate_dataset.py` and keep `person_A.py` working on the 19-channel matrix.
* The 3×3 RREF/LU calibration is unchanged (it only uses `gps_x/y/z`); re-run once and check RREF = LU.
* Be ready to explain: why the clean matrix has rank 3, why noise makes the noisy one full rank, why gyro was changed.

## Person B — rank, basis (≈ +3–4 h, hardest)
* `person-b/person_B.py` replaces "greedy independent columns at 1e-10" with: **divide columns by σ → SVD → keep
  singular values above the Gavish–Donoho threshold → Q = top-r left singular vectors** (Gram-Schmidt as clean-up).
* Output `person_B_orthonormal_basis.csv` (101 × r) and `person_B_scree.png` — show the singular-value table and scree plot in the demo.
* Be ready to explain: why `matrix_rank` returns 19, what Marchenko–Pastur / Gavish–Donoho mean, why the threshold uses
  √(number of timestamps) not √(number of channels), where σ comes from (datasheet), and why Q is not "channels that survived".

## Person C — projection + least squares (≈ +1 h, easy)
* `person-c/person_C.py`: maths unchanged (`P = QQᵀ`, `lstsq` on `[1, t, t²]`, extrapolate, `Aᵀe = 0`). New: it projects
  each GPS receiver, averages them, and prints the comparison table.
* Never project onto an IMU-only Q (RMSE 0.84 in the Monte-Carlo run).
* Viva: projection ≈ halves the error alone, but **projection + least squares ties least squares alone**; fusion of receivers is the real gain.

## Person D — PCA (≈ +1–2 h, moderate; skeleton provided)
* `person-d/person_D.py` is a **skeleton**: inputs + a TODO list + notes. The maths is yours to write.
* Use the **covariance of σ-whitened channels**, not the correlation matrix; expect **two** dominant eigenvalues (centering removes the constant).
* Compare raw vs C's least-squares path vs PCA-filtered path against `true_*`, and compile the Concept → Purpose → Outcome report for all four stages.

## Message you can send
> Team — our 9 sensor channels are independent noise streams, so the matrix is always numerically full rank (9). B's rank step
> finds nothing to remove, and C's projection leaves GPS unchanged (or destroys it if the basis is IMU-only).
> Fix (on branch `fix/low-rank-dataset`, details in `docs/DATASET_FIX.md`): A's generator now makes redundant channels
> (3 GPS receivers, barometer, velocity) with stronger curvature; B uses a noise-aware rank (divide by σ, SVD, Gavish–Donoho
> threshold, Q = top singular vectors); C's code is unchanged; D has a skeleton. No CSV is stored — everyone runs
> `python run_pipeline.py --seed 42` and checks the dataset fingerprint matches.
