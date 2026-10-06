# UE25MA242A — Mathematical Foundation for AI & Data Science: Mini-Project

## Context
Course: UE25MA242A, Mathematical Foundation for AI & Data Science, PES University (Dept. of CSE). This is a Linear Algebra mini-project. The university handout gave 14 official problem statements (basic matrix ops, image filters, linear systems, football rankings, convolution, norms/movies, interpolation/climate, 3D graphics, Chaos Game, PCA/face recognition, PageRank, clustering, SVD/image compression). **This team built a custom problem statement instead**, not picked from the list — confirm with the professor/TA that custom statements are allowed before final submission, since the handout doesn't explicitly forbid it but doesn't explicitly confirm it either.

Team: 4 members (student using this project is Aru-h, doing Person C).

## Why this problem statement (design rationale)
The university's official workflow diagram requires the project to touch (as applicable) ALL of these pipeline stages:
1. Real-World Data
2. Matrix Representation (System of Linear Equations / Linear Transformations)
3. Matrix Simplification (Gaussian Elimination / RREF / LU Decomposition)
4. Structure of the Space (Vector Spaces → Subspaces → Basis → Rank & Nullity)
5. Remove Redundancy (Linear Independence → Basis Selection)
6. Orthogonalization (Orthogonal Vectors → Gram-Schmidt → Orthogonal Bases)
7. Projection (Orthogonal Projections → Projection onto Subspaces)
8. Prediction/Approximation (Least Squares Solution)
9. Pattern Discovery (Eigenvalues & Eigenvectors)
10. System Simplification (Diagonalization of Matrix / Symmetric Matrix)
11. Final Application Output (Predictions / Compression / Trends / Noise Reduction / Modeling)

The chosen UAV sensor fusion problem was specifically designed so each of the 4 team members' work maps onto consecutive pipeline stages, with no stage skipped — this directly satisfies the official evaluation rubric (see below), which explicitly checks "how one step connects to the next in your workflow."

## Problem Statement
**UAV Sensor Fusion — Noisy Trajectory Reconstruction & Prediction via Least Squares and PCA**

A drone's onboard sensors (GPS x,y,z position + IMU accelerometer/gyro orientation) produce noisy, redundant position/orientation readings over time. The project pipeline:
1. Represents raw sensor data as a matrix (rows = timestamps, columns = sensor channels)
2. Simplifies/reduces it to its true degrees of freedom (rank, basis)
3. Produces a clean orthonormal basis for the true motion subspace (Gram-Schmidt)
4. Projects noisy readings onto that basis (first-pass denoising)
5. Fits a least-squares trajectory model to predict/smooth the path
6. Uses PCA (eigenvalue analysis of the covariance matrix) to identify and remove dominant noise/motion patterns (second-pass denoising)
7. Produces a cleaned, predicted flight trajectory, compared against the raw and least-squares-only versions

Can be built with **simulated data** (recommended — simplest, full control over ground truth for error metrics) or **public IMU/GPS datasets** if available.

## Evaluation Scheme (official, applies to the whole team)

**Demo Evaluation — 5 marks**, assessed on:
1. Execution — program runs successfully end to end
2. Concepts Covered — correct use of selected linear algebra components from the workflow
3. Output of Each Component — must clearly explain the result at each stage: matrix representation of data, RREF/matrix simplification, basis and orthogonal basis formation, projection-based prediction, least squares estimation, eigenvalue/eigenvector analysis, final reduced model/application output

**Viva Evaluation — 5 marks**, tests understanding not memorization. For every linear algebra concept used, be ready to explain:
- What concept was used
- Why it was needed in the project
- What result it produced on the data

**Answer structure for viva: Concept → Purpose → Outcome**

**Examiners focus on:**
- What your data matrix represents
- Why each mathematical step was applied
- How each step improved prediction or revealed patterns
- How one step connects to the next in your workflow

**Explicitly NOT expected:** long derivations, theorem proofs, heavy manual calculation.

## Full Role Breakdown (all 4 people, strict dependency chain A → B → C → D)

### Person A — Data + Matrix Representation + RREF/LU
**Task:**
- Source or simulate noisy UAV sensor data: GPS (x, y, z position) + IMU (accelerometer/gyro readings) across multiple timestamps. Arrange as a matrix (rows = timestamps, columns = sensor/position channels).
- Define a motion/calibration model as a linear system Ax = b (e.g., solving for unknown motion coefficients or sensor calibration constants from the raw readings).
- Solve this system via **RREF** (Gaussian elimination), showing the row-reduction steps explicitly.
- Re-solve via **LU decomposition** to demonstrate efficient repeated solving (useful if the same system needs solving at every timestep with different b vectors).
- **Deliverable:** calibration/motion-model system solved, shown step by step, handed forward as the working data matrix for Person B.
- **Viva:** Concept = RREF/LU → Purpose = solve the linear system for motion/calibration coefficients → Outcome = fitted baseline model / solved coefficients.
- **Pipeline stages covered:** Real-World Data, Matrix Representation, Matrix Simplification.

### Person B — Rank, Basis, Linear Independence + Gram-Schmidt
**Task:**
- Compute the **rank** of the sensor data matrix from Person A — this reveals the true degrees of freedom in the system (e.g., are all sensor channels actually independent, or is the real motion constrained to fewer dimensions?).
- Check **linear independence** of the sensor channels (columns); identify and drop any redundant/correlated channels.
- Select a clean **basis** for the motion subspace from the surviving independent vectors.
- Apply **Gram-Schmidt orthogonalization** to convert that basis into an **orthonormal basis**.
- **Deliverable:** orthonormal basis Q of the motion subspace, handed to Person C for projection.
- **Viva:** Concept = rank/independence/Gram-Schmidt → Purpose = remove redundancy and obtain a clean orthogonal basis → Outcome = orthonormal basis of the true motion directions.
- **Pipeline stages covered:** Structure of the Space, Remove Redundancy, Orthogonalization.

### Person C — Projection + Least Squares (Aru-h's role — see person-c-scope.md for full detail)
**Task:**
- **Project** raw noisy sensor readings onto Person B's orthonormal basis Q using the projection matrix P = QQᵀ (valid specifically because Q has orthonormal columns — no matrix inversion needed). This is the first-pass denoising step, removing signal components outside the "true motion" directions.
- Define a **trajectory model** (e.g., degree-2 polynomial position-vs-time: x(t) = a + bt + ct²) and set up the overdetermined linear system Ax = b, where A's columns are time-basis functions (1, t, t²...) and b is the projected position data.
- Solve via **least squares**: x̂ = (AᵀA)⁻¹Aᵀb (normal equations), or numerically via `numpy.linalg.lstsq` for stability.
- Use the fitted model to **predict/extrapolate** future position beyond the observed data range.
- Quantify fit quality via the **residual error** (b − Ax̂), which is guaranteed orthogonal to the column space of A.
- **Deliverable:** smoothed, best-fit predicted flight path with quantified residual error — handed to Person D.
- **Viva:** Concept = Projection + Least Squares → Purpose = remove noise outside the true motion subspace, then fit the best-approximating trajectory to what remains → Outcome = smoothed, predicted flight path with quantified fit error.
- **Pipeline stages covered:** Projection, Prediction/Approximation.

### Person D — PCA / Eigen-analysis + Diagonalization + Report
**Task:**
- Build the **covariance matrix** Σ = (1/n)XᵀX from Person C's (projected, least-squares-fitted) trajectory data.
- **Eigendecompose** Σ: find eigenvalues/eigenvectors using `numpy.linalg.eigh` (preferred over `eig` since covariance matrices are symmetric — guarantees real eigenvalues and orthogonal eigenvectors).
- **Diagonalize**: Σ = VΛVᵀ. Interpret large eigenvalues as directions of real signal/motion variance, small eigenvalues as noise directions.
- **Project onto top-k eigenvectors** for final noise filtering — a second-pass denoising beyond what projection+least-squares (Person C) already achieved.
- **Compare** raw noisy path vs. least-squares-fitted path (Person C's output) vs. PCA-filtered path; compute error metrics (e.g., RMSE against ground truth if using simulated data).
- Produce final visualizations: trajectory overlay plot, eigenvalue spectrum bar chart, before/after noise comparison.
- **Compile the full team report**, structured as Concept → Purpose → Outcome for all 4 pipeline stages (not just D's own stage) — D is best positioned for this since their stage sits last and has visibility into every prior stage's output.
- **Viva:** Concept = covariance + eigendecomposition + diagonalization → Purpose = find directions of maximum variance (signal) vs. minimum variance (noise), reduce dimensionality while preserving the real motion pattern → Outcome = top-k eigenvectors captured X% of variance; filtering along these removed further noise and improved trajectory accuracy.
- **Pipeline stages covered:** Pattern Discovery, System Simplification, Final Application Output.
- Be ready to explain the **spectral theorem** (symmetric matrices → real eigenvalues, orthogonal eigenvectors) as a likely viva follow-up — no proof needed, just state it and explain why it guarantees a clean, interpretable basis here.

## Dependency Chain & Parallelization
Strict dependency: **A → B → C → D**. Each stage genuinely needs the previous stage's real output to produce its own real output.

**Recommended approach to avoid full serialization:** everyone builds their code against **placeholder/dummy data** in parallel for the first 1–2 days (e.g., C builds against a fake orthonormal basis, D builds against a fake fitted trajectory), then swaps in real upstream outputs once available. This compresses total wall-clock time significantly.

## Time Estimate
- Combined: ~29–37 hrs across all 4 people
- Per person: ~7–9 hrs (A, B, C roughly even; D slightly more at ~8–10 hrs due to the report-compilation work)
- Wall-clock: 4–5 days with the parallel/dummy-data approach; 6–7 days if done as a strict sequential handoff

## Integration Risks / Things to Settle as a Team Early
1. **Shared data format convention** — agree up front on matrix orientation (rows = timestamps vs. columns = timestamps) and variable naming, or A→B→C→D handoffs will break on format mismatches, not math errors.
2. **Simulated vs. real data decision** — simulated data is strongly recommended since it gives a known ground truth, which makes Person D's final error-metric comparison (RMSE etc.) meaningful and demo-able.
3. **Floating-point edge cases** — eigenvalues/singular values that should be exactly 0 or 1 will come out as very small nonzero floats (e.g., 0.999999 or 1e-16) — round/filter appropriately rather than treating as a bug.
