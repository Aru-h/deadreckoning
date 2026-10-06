# Viva notes: Person B (rank / basis) and Person C (projection / least squares)

## B: the noise threshold in plain words
1. **Divide each channel by its noise level** (whitening), so every channel has noise of size 1.
2. **Take the SVD** of the whitened matrix. Signal gives a few large singular values; noise gives many small ones.
3. **Keep the singular values above the noise ceiling** tau. Count them: that count is the effective rank r (= 3).
   Nullity = 19 - r = 16 directions that carry only noise.

Why not `matrix_rank`? Noise makes every matrix full rank; numpy only detects round-off (~1e-12), never noise.

- **Marchenko-Pastur edge** sqrt(m) + sqrt(n): the largest singular value pure unit noise of this shape can reach.
- **Gavish-Donoho tau** = lambda*(beta) * sqrt(larger dimension): the cut that minimises reconstruction error. Slightly above the edge.
- If asked "why does it work here": seed 42: three singular values (3200, 358, 47) are above tau = 17; the rest (13, 12, ...) sit at noise level. The third is the closest call.

## B: Gram-Schmidt
SVD vectors are already orthonormal, so Gram-Schmidt changes Q by about 1e-16 (printed by `person_B.py`). It is kept as the explicit orthonormalisation step. The numerical value that mattered was the threshold, not Gram-Schmidt.

## C: honest numbers
- Projection alone about halves the GPS error.
- **Projection + least squares ties least squares alone** (200-seed means 0.0162 vs 0.0161): the quadratic fit already removes the same noise.
- The real gain is **fusing the 3 GPS receivers** (about 0.0094).
- A's 3-point calibration is noisier than C's fit because it solves a 3x3 system from 3 noisy samples (`person_C.py` section 2b prints A vs C vs truth).

## D: questions B/C should be able to hand over
- Spectral theorem: a real symmetric matrix has real eigenvalues and orthogonal eigenvectors, so Sigma = V Lambda V^T.
- Why k = 2, not 3: centring removes the constant term, so the centred signal has rank 2.
- Why n-1: the mean was estimated from the same data.
- Why PCA loses to C: PCA ignores time ordering; the quadratic fit enforces smoothness in time.
