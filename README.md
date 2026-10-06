# Dead Reckoning

**UAV Sensor Fusion — Noisy Trajectory Reconstruction & Prediction**

Linear algebra mini-project for **UE25MA242A — Mathematical Foundation for AI & Data Science** (PES University, Dept. of CSE).

A drone's onboard sensors (GPS x,y,z position + IMU accelerometer/gyro orientation) produce noisy, redundant readings over time. This project reconstructs and predicts the true flight trajectory from that noisy data using a linear-algebra pipeline:

1. **Data + Matrix Representation + RREF/LU** — raw sensor data as a matrix; motion/calibration model solved via Gaussian elimination and LU decomposition
2. **Rank, Basis, Independence + Gram-Schmidt** — true degrees of freedom, redundant channels dropped, orthonormal basis of the motion subspace
3. **Projection + Least Squares** — noisy readings projected onto the orthonormal basis (denoising), then a least-squares trajectory model fit and extrapolated
4. **PCA / Eigen-analysis + Diagonalization** — covariance eigendecomposition for a second denoising pass, final trajectory comparison and report

## Team
4 members, strict dependency chain A → B → C → D. See `docs/mini-project-overview.md` for the full role breakdown, evaluation scheme, and design rationale.

## Structure
```
docs/           Project overview and scope documents
person-a/       Data + matrix representation + RREF/LU
person-b/       Rank, basis, independence + Gram-Schmidt
person-c/       Projection + least squares (docs/person-c-scope.md)
person-d/       PCA / eigen-analysis + diagonalization + report
```
