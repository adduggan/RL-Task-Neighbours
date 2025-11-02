# grader.py
# Behavioral grader: correctness (subset), determinism, edge-cases, performance, resource.

from __future__ import annotations
from pathlib import Path
import time
import tempfile
import importlib
import numpy as np
import pandas as pd
import tracemalloc

from tools import time_block, bytes_to_mb  # time_block optional; bytes_to_mb used

DATA_DIR = Path("data")
OUT_DIR = Path("outputs")
EMB_PATH = DATA_DIR / "embeddings.npy"
OUT_PATH = OUT_DIR / "topk.parquet"

# Tunables (adjust to hit 10–40% pass-rate on your model):
X_SECONDS = 5.0       # max wall time for full dataset run in this env
Y_MB = 800.0          # rough peak alloc over baseline (tracemalloc)
K = 5                 # top-k
SAMPLE_FOR_REF = 60   # how many rows to fully re-check
DET_ROWS = 50         # rows used for determinism micro-run
ATOL = 1e-4
TIE_EPS = 1e-6


def _safe_norms(X: np.ndarray) -> np.ndarray:
    # L2 norms with float32 stability
    norms = np.linalg.norm(X, axis=1)
    return norms


def _cosine_sim_block(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    # Computes cosine between rows of A and rows of B; handles zero norms by returning 0
    a_norm = _safe_norms(A)
    b_norm = _safe_norms(B)
    denom = (a_norm[:, None] * b_norm[None, :])
    S = A @ B.T
    np.divide(S, denom, out=S, where=denom != 0)  # Where denom==0, leave as 0
    return S


def _reference_topk(X: np.ndarray, rows: np.ndarray, k: int):
    # For each row index in `rows`, compute reference neighbors among all rows
    A = X[rows]
    S = _cosine_sim_block(A, X)
    # exclude self by setting -inf on diagonal positions for the chosen rows
    S[np.arange(len(rows)), rows] = -np.inf
    # get similarities per row
    idx_part = np.argpartition(S, -k, axis=1)[:, -k:]
    sims_part = np.take_along_axis(S, idx_part, axis=1)
    order = np.argsort(-sims_part, axis=1)
    top_idx = np.take_along_axis(idx_part, order, axis=1)
    top_sim = np.take_along_axis(sims_part, order, axis=1)
    return top_idx, top_sim


def _load_output(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing output file: {path}")
    df = pd.read_parquet(path)
    expected_cols = {"row_id", "nbr_idx", "nbr_sim"}
    if set(df.columns) != expected_cols:
        raise ValueError(f"Output schema must be {expected_cols}, got {set(df.columns)}")
    if len(df) == 0:
        raise ValueError("Empty output")
    # row_id must be 0..N-1 in order
    if not np.array_equal(df["row_id"].values, np.arange(len(df))):
        raise ValueError("row_id must be 0..N-1 in order")
    # list lengths == K
    if any(len(x) != K for x in df["nbr_idx"]):
        raise ValueError("nbr_idx lists must have length K")
    if any(len(x) != K for x in df["nbr_sim"]):
        raise ValueError("nbr_sim lists must have length K")
    # finite sims & non-increasing order per row
    for i, sims in enumerate(df["nbr_sim"]):
        arr = np.array(sims, dtype=np.float32)
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"Row {i}: non-finite similarities")
        if np.any(arr[1:] - arr[:-1] > 1e-7):
            raise ValueError(f"Row {i}: nbr_sim must be sorted desc (non-increasing)")
    return df


def _determinism_check(X: np.ndarray, k: int) -> None:
    # create a temporary small dataset and call solution.main twice
    import solution
    rows = np.arange(min(len(X), DET_ROWS))
    X_small = X[rows]
    with tempfile.TemporaryDirectory() as td:
        tdir = Path(td)
        data_dir = tdir / "data"
        out_dir = tdir / "outputs"
        data_dir.mkdir()
        out_dir.mkdir()
        small_path = data_dir / "embeddings.npy"
        np.save(small_path, X_small)
        out_path = out_dir / "topk.parquet"
        solution.main(str(small_path), str(out_path))
        df1 = pd.read_parquet(out_path)
        # run again
        solution.main(str(small_path), str(out_path))
        df2 = pd.read_parquet(out_path)
        if not df1.equals(df2):
            raise AssertionError("Determinism check failed: two runs differ on same data")


def _zero_vector_rule(X: np.ndarray, df: pd.DataFrame) -> None:
    norms = _safe_norms(X)
    zero_rows = np.where(norms == 0)[0]
    for i in zero_rows:
        sims = np.array(df.loc[i, "nbr_sim"], dtype=np.float32)
        if not np.allclose(sims, 0.0, atol=0.0):
            raise AssertionError(f"Row {i}: zero-vector must yield 0.0 sims to all neighbors")


def _self_exclusion_check(df: pd.DataFrame) -> None:
    for i in range(len(df)):
        if i in set(df.loc[i, "nbr_idx"]):
            raise AssertionError(f"Row {i}: self index present in neighbors")


def _subset_correctness(X: np.ndarray, df: pd.DataFrame) -> None:
    rng = np.random.default_rng(123)
    rows = rng.choice(len(X), size=min(SAMPLE_FOR_REF, len(X)), replace=False)
    ref_idx, ref_sim = _reference_topk(X, rows, K)
    for j, i in enumerate(rows):
        pred_idx = np.array(df.loc[i, "nbr_idx"], dtype=np.int64)
        pred_sim = np.array(df.loc[i, "nbr_sim"], dtype=np.float32)
        # similarities within tolerance
        if not np.allclose(pred_sim, ref_sim[j], atol=ATOL):
            raise AssertionError(f"Row {i}: similarities deviate beyond atol")
        # set-wise acceptance under ties: accept any indices with sim >= kth - TIE_EPS
        kth = ref_sim[j][-1]
        Srow = _cosine_sim_block(X[i:i+1], X)[0]
        Srow[i] = -np.inf
        allowed = set(np.where(Srow >= (kth - TIE_EPS))[0])
        if not set(pred_idx).issubset(allowed) or len(pred_idx) != K:
            raise AssertionError(f"Row {i}: neighbor indices not within allowed tie set")


def _performance_check(X_path: Path) -> None:
    import solution
    start = time.perf_counter()
    tracemalloc.start()
    with tempfile.TemporaryDirectory() as td:
        tdir = Path(td)
        out_path = tdir / "topk.parquet"
        solution.main(str(X_path), str(out_path))
        current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    elapsed = time.perf_counter() - start
    if elapsed > X_SECONDS:
        raise AssertionError(f"Runtime {elapsed:.2f}s exceeds limit {X_SECONDS:.2f}s")
    if bytes_to_mb(peak) > Y_MB:
        raise AssertionError(f"Peak alloc {bytes_to_mb(peak):.1f}MB exceeds limit {Y_MB:.1f}MB")


def main():
    # 1) Load embeddings
    if not EMB_PATH.exists():
        raise FileNotFoundError("Run `python task.py` first to generate data and output.")
    X = np.load(EMB_PATH)

    # 2) Determinism micro-check (before reading user's big output)
    _determinism_check(X, K)

    # 3) Load candidate output from OUT_PATH
    df = _load_output(OUT_PATH)

    # 4) Edge cases & structural checks
    _self_exclusion_check(df)
    _zero_vector_rule(X, df)

    # 5) Correctness on sample
    _subset_correctness(X, df)

    # 6) Performance on full data (fresh run)
    _performance_check(EMB_PATH)

    print("PASS: All checks succeeded.")


if __name__ == "__main__":
    main()