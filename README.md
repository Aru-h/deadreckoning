# Dead Reckoning

**UAV Sensor Fusion — Noisy Trajectory Reconstruction & Prediction**

Linear algebra mini-project for **UE25MA242A — Mathematical Foundation for AI & Data Science** (PES University, Dept. of CSE).

A drone's onboard sensors (GPS x,y,z position + IMU accelerometer/gyro orientation) produce noisy, redundant readings over time. This project reconstructs and predicts the true flight trajectory from that noisy data using a linear-algebra pipeline:

1. **Data + Matrix Representation + RREF/LU** — raw sensor data as a matrix; motion/calibration model solved via Gaussian elimination and LU decomposition
2. **Rank, Basis, Independence + Gram-Schmidt** — true degrees of freedom (noise-aware rank), orthonormal basis of the motion subspace
3. **Projection + Least Squares** — noisy readings projected onto the orthonormal basis (denoising), then a least-squares trajectory model fit and extrapolated
4. **PCA / Eigen-analysis + Diagonalization** — covariance eigendecomposition for a second denoising pass, final trajectory comparison and report

## Quick start
```bash
pip install -r requirements.txt
python run_pipeline.py                 # seed 42: every teammate gets the identical dataset
python run_pipeline.py --seed hello    # any integer or word is a seed (like a Minecraft world seed)
python run_pipeline.py --seed random   # fresh draw; the seed is printed so it can be shared
python tools/monte_carlo_validation.py # 200-seed validation of the numbers in docs/DATASET_FIX.md
```
**No dataset is stored in the repo.** `person-a/generate_dataset.py` recreates it from the seed; each run prints a
*dataset fingerprint* — if two teammates use the same `--seed` and `--n-rx` the fingerprint must match.

## Team
4 members, strict dependency chain A → B → C → D. See `docs/mini-project-overview.md` for the full role breakdown, evaluation scheme, and design rationale.

## Structure
```
docs/           Overview, scope, DATASET_FIX.md (why/what changed), TEAM_CHANGES.md (who does what), research/
person-a/       Data generator + matrix representation + RREF/LU        (generate_dataset.py, person_A.py)
person-b/       Noise-aware rank, basis, Gram-Schmidt                     (person_B.py)
person-c/       Projection + least squares + prediction                   (person_C.py)
person-d/       PCA / eigen-analysis + report                             (person_D.py — SKELETON)
tools/          Monte-Carlo validation
run_pipeline.py Runs A → B → C → D
```
Each stage reads the previous stage's files and prints what it did; generated CSV/PNG outputs are git-ignored.

## Status
| Stage | State |
|---|---|
| A — data, RREF, LU | working (v2 dataset; v1 kept in `person-a/legacy/`) |
| B — rank, basis | working (noise-aware rank + SVD basis) |
| C — projection, least squares | working |
| D — PCA | skeleton, to be implemented |

## Important: dataset v2
The v1 dataset made every sensor channel independent noise, so the matrix was always full rank and the rank/projection steps
did nothing. v2 adds redundant channels and a noise-aware rank. Read [docs/DATASET_FIX.md](docs/DATASET_FIX.md) before the viva.
