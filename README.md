# Dead Reckoning

**UAV Sensor Fusion — Noisy Trajectory Reconstruction & Prediction**

Linear algebra mini-project for **UE25MA242A — Mathematical Foundation for AI & Data Science** (PES University, Dept. of CSE).

A drone's onboard sensors (GPS x,y,z position + IMU accelerometer/gyro orientation) produce noisy, redundant readings over time. This project reconstructs and predicts the true flight trajectory from that noisy data using a linear-algebra pipeline:

1. **Data + Matrix Representation + RREF/LU** — raw sensor data as a matrix; motion/calibration model solved via Gaussian elimination and LU decomposition
2. **Rank, Basis, Independence + Gram-Schmidt** — true degrees of freedom (noise-aware rank), orthonormal basis of the motion subspace
3. **Projection + Least Squares** — noisy readings projected onto the orthonormal basis (denoising), then a least-squares trajectory model fit and extrapolated
4. **PCA / Eigen-analysis + Diagonalization** — covariance eigendecomposition for a second denoising pass, final trajectory comparison and report

## Setup and run (Linux, macOS, Windows)

**Needs:** Python 3.10 or newer and git. Everything else is installed from `requirements.txt` (numpy, scipy, matplotlib).
No GPU, no display and no other tools are needed — plots are saved as PNG files, no window opens.
Tested and passing on Linux (x86), macOS (Apple Silicon) and Windows, all printing the same dataset fingerprint; Python 3.13 and 3.14, NumPy 2.3.5 and 2.5.3.

**1. Get the code**
```bash
git clone https://github.com/Aru-h/deadreckoning.git
cd deadreckoning
```

**2. Create an environment and install the packages** (pick your OS)

| | Linux / macOS (Terminal) | Windows (PowerShell) |
|---|---|---|
| Create | `python3 -m venv .venv` | `py -m venv .venv` |
| Activate | `source .venv/bin/activate` | `.venv\Scripts\Activate.ps1` |
| Install | `pip install -r requirements.txt` | `pip install -r requirements.txt` |

Windows `cmd.exe`: activate with `.venv\Scripts\activate.bat`. If PowerShell refuses to run the activate script, run
`Set-ExecutionPolicy -Scope Process RemoteSigned` once in that window and activate again.
On macOS/Linux, if `python3` is missing install it from python.org or your package manager (`brew install python`, `sudo apt install python3 python3-venv`).
Once the environment is active, `python` works on all three systems.

**3. Run the whole project**
```bash
python run_pipeline.py                 # seed 42: every teammate gets the identical dataset
python run_pipeline.py --seed hello    # any integer or word is a seed (like a Minecraft world seed)
python run_pipeline.py --seed random   # fresh draw; the seed is printed so it can be shared
python run_pipeline.py --n-rx 6        # 6 GPS receivers per axis instead of 3
python tools/monte_carlo_validation.py # 200-seed validation of the numbers in docs/DATASET_FIX.md
```
It runs A → B → C → D in order (D does PCA / eigen-analysis). Generated files (`*.csv`, `*.png`, `person_*_output.txt`)
appear next to each script and are git-ignored.

**What a good run looks like:** `Seed: 42`, `Dataset fingerprint: cfb83d778fae` (seed 42, 3 receivers — must match across laptops),
`Effective (signal) rank r = 3   [legacy tests said 19]`, `max |Q^T Q - I|` around `1e-16`, and a validation table from Person C
ending in `projection + average of 3 + least squares  [pipeline]` with errors around 0.01.

**Run one stage on its own** (after the stages before it): `python person-a/generate_dataset.py`, `python person-a/person_A.py`,
`python person-b/person_B.py`, `python person-c/person_C.py`, `python person-d/person_D.py`. Each script finds its inputs relative to its own
location, so it works from any folder. A missing-input error tells you which earlier stage to run.

**Troubleshooting:** `python` not found → use `python3` (Linux/macOS) or `py` (Windows), or activate the environment first.
`ModuleNotFoundError` → the environment is not active or `pip install -r requirements.txt` was skipped.
Different fingerprint than a teammate → different `--seed` or `--n-rx`.

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
person-d/       PCA / eigen-analysis + report                             (person_D.py)
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
| D — PCA | working (covariance, eigendecomposition, top-k, comparison) |

## Important: dataset v2
The v1 dataset made every sensor channel independent noise, so the matrix was always full rank and the rank/projection steps
did nothing. v2 adds redundant channels and a noise-aware rank. Read [docs/DATASET_FIX.md](docs/DATASET_FIX.md) before the viva.
