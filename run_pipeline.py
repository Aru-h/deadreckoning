"""
Run the whole chain A -> B -> C -> D from one command. Nothing needs to be stored in git:
the dataset is regenerated from a seed (like a Minecraft world seed).

    python run_pipeline.py                  # seed 42 (default): identical data on every laptop
    python run_pipeline.py --seed 7         # a different, still reproducible, dataset
    python run_pipeline.py --seed hello     # any word works as a seed
    python run_pipeline.py --seed random    # fresh draw; the seed is printed so you can share it
    python run_pipeline.py --n-rx 6         # 6 GPS receivers per axis instead of 3

Teammates comparing results: make sure the "Dataset fingerprint" line matches.
"""

import argparse
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))


def run(label, script, *args):
    print(f"\n{'#' * 70}\n# {label}\n{'#' * 70}")
    cmd = [sys.executable, os.path.join(ROOT, script), *args]
    env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")   # Windows-safe output
    if subprocess.call(cmd, cwd=os.path.dirname(os.path.join(ROOT, script)), env=env) != 0:
        sys.exit(f"{script} failed -- fix it before running the next stage.")


def main():
    ap = argparse.ArgumentParser(description="Run Person A -> B -> C -> D")
    ap.add_argument("--seed", default="42")
    ap.add_argument("--n-rx", default="3")
    args = ap.parse_args()

    run("Person A: generate dataset", "person-a/generate_dataset.py",
        "--seed", args.seed, "--n-rx", args.n_rx)
    run("Person A: matrix, RREF, LU", "person-a/person_A.py")
    run("Person B: rank, basis, Gram-Schmidt", "person-b/person_B.py")
    run("Person C: projection, least squares, prediction", "person-c/person_C.py")
    run("Person D: PCA / eigen-analysis (skeleton until implemented)", "person-d/person_D.py")


if __name__ == "__main__":
    main()
