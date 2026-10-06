# Person C — Projection + Least Squares (Aru-h's role)

See mini-project-overview.md for full project context, the other 3 roles, and the evaluation scheme. This doc is the complete, self-contained scope for Person C — a new chat should need nothing else to pick this up.

## Task Summary
Take Person B's orthonormal basis Q → project noisy sensor readings onto it (denoise) → fit a least-squares trajectory to the projected data → predict/extrapolate → quantify error. Hand the result to Person D.

## Full Pipeline

### Step 1 — Projection (denoising)
Given Person B's orthonormal basis Q (columns are orthonormal vectors q₁...qₖ spanning the true motion subspace):

**Projection matrix:** P = QQᵀ

This formula is valid *specifically* because Q has orthonormal columns. The general projection matrix onto the column space of any matrix A is P = A(AᵀA)⁻¹Aᵀ — but when A = Q has orthonormal columns, QᵀQ = I, so the inversion drops out entirely: P = Q(QᵀQ)⁻¹Qᵀ = Q·I·Qᵀ = QQᵀ. This is why Gram-Schmidt (Person B's output) is worth doing — it turns projection into a trivial matrix multiply.

**Projecting raw data:** for each raw noisy reading vector x, the projection is:
p = Px = QQᵀx = (q₁ᵀx)q₁ + (q₂ᵀx)q₂ + ... + (qₖᵀx)qₖ

This removes any component of x that falls outside the span of Q — i.e., anything that isn't in the "true motion" directions gets discarded. This is the first-pass denoising step.

### Step 2 — Least Squares Trajectory Fitting
Define a trajectory model. For UAV position data (t_i, x_i, y_i, z_i), a degree-2 polynomial fit per axis looks like:

x(t) = a₀ + a₁t + a₂t²

Build matrix A with columns = powers of time:
```
A = [1  t₁  t₁²]
    [1  t₂  t₂²]
    [⋮   ⋮    ⋮ ]
    [1  tₙ  tₙ²]
```
and b = projected position values (output of Step 1) at each time.

Since there are more data points (timestamps) than unknowns (polynomial coefficients), this system Ax = b is **overdetermined** — generally no exact solution exists, so solve via least squares:

**Normal equations:** AᵀA x̂ = Aᵀb → x̂ = (AᵀA)⁻¹Aᵀb

In practice, use `numpy.linalg.lstsq(A, b)` rather than manually inverting AᵀA — numerically more stable, avoids conditioning issues.

**Why this works (the projection connection):** the least-squares solution x̂ is exactly the one that makes Ax̂ the orthogonal projection of b onto the column space of A. The **residual** e = b − Ax̂ is guaranteed orthogonal to every column of A, i.e., Aᵀe = 0. This is literally the same orthogonal-projection idea as Step 1, just applied to a different subspace (column space of the time-basis matrix A, instead of the sensor-basis Q).

### Step 3 — Prediction / Extrapolation
Once x̂ (the polynomial coefficients) is found, evaluate the fitted polynomial at future time values beyond the observed range to predict future position. Quantify fit quality via the residual norm ‖e‖ = ‖b − Ax̂‖, or per-point residuals for a more detailed error breakdown.

## Deliverable
A smoothed, best-fit predicted flight path derived from noisy sensor data, with quantified residual error — handed to Person D, who applies PCA/eigen-analysis on top of this for a second (final) pass of noise filtering.

## Viva Structure (Concept → Purpose → Outcome)
- **Concept:** Projection + Least Squares
- **Purpose:** Remove noise outside the true motion subspace (via projection onto B's orthonormal basis), then fit the best-approximating trajectory curve to what remains (via least squares)
- **Outcome:** Smoothed, predicted flight path with quantified fit error (residuals)

**Likely viva follow-up questions and how to answer:**
1. *Why is the projection matrix for orthonormal Q just QQᵀ and not Q(QᵀQ)⁻¹Qᵀ?* — Because QᵀQ = I when columns are orthonormal, so the inverse term disappears; the full formula reduces to QQᵀ.
2. *Why is the least-squares residual always orthogonal to the column space of A?* — Because x̂ is chosen to minimize ‖b − Ax̂‖, and the minimum occurs exactly where the error vector is perpendicular to every column of A (otherwise you could reduce the error further by moving along a column direction). Formally: Aᵀ(b − Ax̂) = 0.
3. *What does matrix A look like for a degree-2 polynomial fit on (t_i, x_i, y_i, z_i) data?* — A has one row per timestamp and columns [1, t, t²]; this A is reused for fitting x(t), y(t), z(t) separately (same time-basis, different b vector for each coordinate).

## Theory Prerequisites — Unit-2 Course Notes (by page)
Study these pages; skip the rest of Unit-2 (that's Person D's scope: eigenvalues, diagonalization, positive-definiteness, SVD, PCA are pages 25 onward plus the separate SVD/PCA handout).

| Pages | Topic | Why Person C needs it |
|---|---|---|
| 1–3 | Norm, dot product, orthogonality | Foundation — $\|x\|^2 = x^Tx$, $x^Ty = \|x\|\|y\|\cos\theta$ |
| 4–5 | Orthogonal subspaces, orthogonal complement | Conceptual grounding for what "projecting onto a subspace" means structurally |
| 10–12 | Orthonormal bases, orthogonal matrices, properties | **Core** — $P=QQ^T$ comes directly from property 6 here: $b=(q_1^Tb)q_1+\cdots+(q_k^Tb)q_k$ |
| 13–14 | Gram-Schmidt | Context only — Person B implements this; C consumes the orthonormal basis it produces |
| 15–18 | QR factorization | Useful — note that $R\hat{x}=Q^Tb$ solves least squares when $Ax=b$ is inconsistent; same underlying method as normal equations |
| 19–21 | Projection onto a line, rank-1 projection matrix | **Core** — $P=\frac{aa^T}{a^Ta}$, the single-vector case before generalizing to QQᵀ |
| 22–24 | Projections & Least Squares | **Core** — normal equations $A^TA\hat{x}=A^Tb$, error-orthogonality, worked least-squares-line-fit examples |

## Reading Sources (external)
- QuantEcon — Orthogonal Projections and Their Applications: https://python-advanced.quantecon.org/orth_proj.html (theory + NumPy code side by side)
- ITU Projection & Least Squares Tutorial: https://iml.itu.dk/04-tutorials/04-projection_least_sq.html (hands-on: construct projection matrix, project points, derive least-squares fitting from projection)
- Duke/Connected Curriculum — Least Squares via Orthogonal Projection: https://amser.org/r13794/least_squares (develops least-squares fitting directly from projection-onto-subspace concept)
- WVU Least Squares Problem Set: https://community.wvu.edu/~kciesiel/ProfessionalStuff/teach/Spring2011/441Spr2011/leastsquares.pdf (worked examples, viva-style "what vector is projected onto what subspace" reasoning)
- Charles University — Orthogonal Projection & Method of Least Squares slides: https://iuuk.mff.cuni.cz/~ipenev/LA2S2025Lecture17slides.pdf
- numpy.linalg.lstsq docs: https://numpy.org/doc/stable/reference/generated/numpy.linalg.lstsq.html
- numpy.polyfit docs: https://numpy.org/doc/stable/reference/generated/numpy.polyfit.html (convenience wrapper for polynomial least-squares)
- scipy.linalg.lstsq docs: https://docs.scipy.org/doc/scipy-1.16.1/reference/generated/scipy.linalg.lstsq.html (alternative solver, includes worked quadratic-fit example)

## Time Estimate
~7–9 hrs total: projection (2–3 hrs) + least-squares setup and solve (3–4 hrs) + prediction/error analysis (2 hrs)

## Dependency
Needs Person B's orthonormal basis Q before the real projection step can run. Build and test the projection + least-squares code against a placeholder/dummy orthonormal basis in parallel if B isn't finished yet, then swap in B's real basis once available — avoids idle time from strict serialization.

## Tools
`numpy.linalg.lstsq` (preferred over manually inverting AᵀA — numerically more stable), direct matrix multiplication `Q @ Q.T @ x` for projection (no inversion needed since Q is orthonormal), `matplotlib` for visualizing raw vs. projected vs. fitted trajectory.
