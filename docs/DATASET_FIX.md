# Dataset fix: why rank/projection were vacuous, and what changed

Status: implemented on branch `fix/low-rank-dataset`. Applies to Persons A, B, C (D has a skeleton).
Per-person actions: [TEAM_CHANGES.md](TEAM_CHANGES.md). Full literature/experiment write-up:
[research/low-rank-dataset-fix-report.md](research/low-rank-dataset-fix-report.md).

## 1. The problem (v1 dataset)

The v1 sensor matrix had 9 channels (GPS x,y,z, accel x,y,z, gyro x,y,z) over 101 timestamps.
Each channel was an independent noise stream: GPS = exact quadratic path + white noise, accel =
constant + noise, gyro = pure noise. Consequences:

| Step | What happened on v1 |
|---|---|
| Person B rank | `np.linalg.matrix_rank`, `calculate_rank(tol=1e-10)` and `find_independent_columns` all return **9 of 9**. |
| Projection onto a basis of all 9 channels | GPS is one of the spanning vectors, so `P = QQᵀ` is the **identity on GPS** (`‖raw − projected‖ = 0`, RMSE unchanged). |
| Projection onto an IMU-only basis | Accel/gyro span at most `{1, t}`, but position needs `t²`: signal destroyed (x-axis RMSE 0.09 → 3.97 on v1). |
| Thresholding alone | The 3rd *clean* singular value was ≈ 0.054 vs a noise floor ≈ 0.1: even a noise-aware rule reports rank 2, not 3. |

## 2. Root cause

Not a bug in anyone's code. Two separate things:

1. **Noise makes any matrix full rank.** Continuous independent noise on every column lifts the
   rank to `min(rows, cols)` with probability 1. numpy's default tolerance (`S.max()·max(M,N)·ε`)
   and the old `1e-10` only detect round-off-level dependence (~1e-12 × s_max), never noise.
2. **The data had no redundancy and too little curvature.** The clean matrix does have rank 3 (every
   clean channel is a linear function of the three coefficients per axis, i.e. lies in
   `span{1, t, t²}`), but nothing in v1 made that structure *visible* above the noise.

## 3. What changed

| Stage | Change |
|---|---|
| **A** `person-a/generate_dataset.py` (new) | Regenerates the data with **redundant channels** (3 GPS receivers per axis, barometer, velocity), physically consistent gyro (`gyro_z` = yaw rate of a polynomial heading; `gyro_x/y` level-flight noise), and **stronger curvature** `a₂ = 0.15 / 0.10 / 0.08` (v1: 0.05 / 0.03 / 0.01). Writes `noise_sigma.csv` (noise std per channel). 19 measurement channels. |
| **A** `person_A.py` | Builds the matrix from every measurement channel in the CSV (was a hard-coded 9). RREF/LU calibration on `gps_x/y/z` at t = 0, 5, 10 s is unchanged. Original kept in `person-a/legacy/`. |
| **B** `person_B.py` | **Noise-aware rank + SVD basis** replaces the greedy "is this column independent at 1e-10" test. Divide each column by its noise σ → SVD → keep singular values above the **Gavish–Donoho threshold** → `Q` = top-r left singular vectors, then Gram-Schmidt as clean-up and `‖I − QᵀQ‖` check. Legacy functions are kept in the file and the script prints how they fail. |
| **C** `person_C.py` | Maths unchanged: `p = Q(Qᵀx)`, quadratic `lstsq`, extrapolation, `Aᵀe = 0` check. Now projects each GPS receiver, **fuses** (averages) them, and prints an honest comparison table. Q must come from B's all-channel SVD (never IMU-only). |
| **D** `person_D.py` | **Skeleton only** (inputs, task list, notes). Use the covariance of σ-whitened channels, not the correlation matrix. |
| Tooling | `run_pipeline.py` (A→B→C→D), `tools/monte_carlo_validation.py` (reproduces the numbers below). |

### Noise-aware rank in one paragraph
After dividing columns by σ the noise in every column has unit variance, so the largest *noise*
singular value of an `m × n` matrix concentrates near the Marchenko–Pastur edge `√m + √n`.
Gavish–Donoho give the MSE-optimal hard cutoff `τ = λ*(β)·√m` with `β = n/m` (m = the larger
dimension, here 101 timestamps, **not** 9 or 19 channels):
`λ*(β) = √(2(β+1) + 8β / ((β+1) + √(β²+14β+1)))`. The effective rank is the number of whitened singular
values above `τ`. For 101 × 19 this gives `τ ≈ 17.0` (MP edge ≈ 14.4); the seed-42 spectrum is
3198, 355, 46 | 12.9, 12.6, … so r = 3.

## 4. Reproducible dataset without storing a CSV

The CSV is **not** committed (`*.csv` is git-ignored). `python run_pipeline.py --seed 42` recreates it.
Like a Minecraft world seed: same `--seed` (an integer or any word) **and** same `--n-rx` ⇒ identical
dataset. The generator uses NumPy's legacy `RandomState`, whose stream NumPy guarantees to be frozen
across versions/platforms, and CRC32 (not Python's randomised `hash()`) for word seeds. Every run prints a
**dataset fingerprint** (SHA-256 prefix of the CSV); teammates must see the same value.
`--seed random` draws a fresh seed and prints it.

Verified: same seed in separate processes (different `PYTHONHASHSEED`) gives a byte-identical file.
*Not* verified: a different NumPy version or OS (the install of a second NumPy timed out in the sandbox);
that part rests on NumPy's documented `RandomState` stability guarantee.

## 5. Results (computed, reproducible)

`python tools/monte_carlo_validation.py` — 200 seeds (0–199), 3 receivers/axis, RMSE vs the hidden true
path averaged over x, y, z:

| Variant | RMSE (mean ± std) |
|---|---|
| Noise-aware rank equals the true rank 3 | 100 % of seeds |
| Legacy `find_independent_columns` keeps | 100 % of columns (removes nothing) |
| Raw GPS (1 receiver) | 0.0995 ± 0.0039 |
| Legacy all-channel basis, projected only | 0.0995 (identity — no change) |
| IMU-only basis, projected only | 0.8405 (signal destroyed) |
| **Projection only** (1 receiver) | **0.0467** (≈ 2.1× better than raw) |
| Least squares only (1 raw receiver) | 0.0161 |
| Projection + least squares (1 receiver) | 0.0162 |
| Average of 3 raw receivers + least squares | 0.0093 |
| **Projection + average of 3 + least squares (pipeline)** | **0.0094** |

### What to say honestly in the viva
* Projection roughly halves the error on its own (removes noise outside the 3-D signal subspace),
  so it is doing real denoising now.
* **Projection + least squares ties least squares alone** (0.0162 vs 0.0161): the quadratic fit already
  enforces the motion model. Do not claim projection is what gives the final accuracy.
* **Redundancy is what helps most:** fusing receivers cuts the error ~42 % (0.0161 → 0.0094).
* The true rank (3), the clean-matrix structure and the noise σ are *known here because the data is
  simulated*; with real data σ would be estimated (e.g. from the residual of a per-column quadratic fit).

## 6. Limitations / not tested

* Noise is independent white Gaussian. Real GPS receivers share multipath/ionospheric error and have
  outliers and bias; none of that is tested.
* `a₂` was chosen from a coarse scan, not a minimal-sufficient search. Tripling the v1 curvature already
  gave 100 % detection in the research runs.
* The Gram-Schmidt error-bound exponent for the *modified* variant is unresolved in the sources read
  (Moler: κ; one set of lecture notes: κ²). Check Björck (1967), Golub & Van Loan §5.2.8 or Higham before quoting it.
* Person D's stage is a skeleton; the "two dominant eigenvalues" statement was checked on the whitened
  19-channel covariance only (seed 42: 23445, 22.2, then a bulk ≤ 1.8; the correlation matrix hides the second one).
* Public datasets (INSANE, Zurich MAV, ALFA) were reviewed: real noise is also full rank, so they do not remove the
  need for a noise-aware rank. They could serve as a separate real-data demo; INSANE's page and any column layout were not read.

## 7. Sources

* Gavish & Donoho, *The Optimal Hard Threshold for Singular Values is 4/√3*, IEEE Trans. Inf. Theory 2014 — [arXiv:1305.5870](https://arxiv.org/pdf/1305.5870)
* [numpy.linalg.matrix_rank](https://numpy.org/doc/stable/reference/generated/numpy.linalg.matrix_rank.html) (default tolerance)
* [Marchenko–Pastur distribution](https://en.wikipedia.org/wiki/Marchenko%E2%80%93Pastur_distribution)
* MIT 18.065 Lecture 7, Eckart–Young — [OCW](https://live.ocw.mit.edu/courses/18-065-matrix-methods-in-data-analysis-signal-processing-and-machine-learning-spring-2018/resources/lecture-7-eckart-young-the-closest-rank-k-matrix-to-a/)
* Moler, [Apologies to Gram-Schmidt](https://blogs.mathworks.com/cleve/2016/10/27/apologies-to-gram-schmidt/)
* Full list: [research report](research/low-rank-dataset-fix-report.md)
