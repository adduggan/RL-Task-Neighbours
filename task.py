# task.py
# Entry point: loads/generates data, then calls solution.main to write outputs/topk.parquet

from pathlib import Path
import numpy as np
import importlib
import sys
import os

DATA_DIR = Path("data")
OUT_DIR = Path("outputs")
EMB_PATH = DATA_DIR / "embeddings.npy"
N, D = 1500, 128  # tune alongside grader thresholds

# Allow seeding via environment variable for varied trials
SEED = int(os.getenv("TASK_SEED", "42"))


def ensure_data():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not EMB_PATH.exists():
        rng = np.random.default_rng(SEED)
        X = rng.normal(size=(N, D)).astype(np.float32)
        # inject a few exact zero vectors (edge case)
        zero_rows = rng.choice(N, size=max(3, N // 100), replace=False)
        X[zero_rows] = 0.0
        np.save(EMB_PATH, X)
        print(f"Generated embeddings with seed {SEED}")
    else:
        print(f"Using existing embeddings at {EMB_PATH}")


def main():
    ensure_data()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    try:
        solution = importlib.import_module("solution")
    except Exception as e:
        print("ERROR: Could not import solution.py:", e)
        sys.exit(1)

    try:
        solution.main(str(EMB_PATH), str(OUT_DIR / "topk.parquet"))
    except Exception as e:
        print("ERROR: solution.main failed:", e)
        sys.exit(1)


if __name__ == "__main__":
    main()