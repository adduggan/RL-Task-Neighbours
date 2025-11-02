# solution.py
# Compute top-5 cosine neighbors per row, excluding self, deterministic, NumPy-only.

from __future__ import annotations
import numpy as np
import pandas as pd

K = 5  # top-k


def _safe_norms(X: np.ndarray) -> np.ndarray:
    """Row-wise L2 norms (float32)."""
    return np.linalg.norm(X, axis=1)


def main(input_path: str, output_path: str) -> None:
    # Load embeddings [N, D]
    X = np.load(input_path)
    X = np.asarray(X, dtype=np.float32)
    N = X.shape[0]

    # Normalize rows where norm > 0 (zero rows remain zeros)
    norms = _safe_norms(X)
    Xn = X.copy()
    nz = norms > 0
    Xn[nz] /= norms[nz, None]

    # Cosine similarity via dot of normalized rows
    # Zero rows stay all-zeros against others
    S = Xn @ Xn.T

    # Exclude self matches
    np.fill_diagonal(S, -np.inf)

    # Top-K via argpartition + argsort within the K block (descending)
    idx_part = np.argpartition(S, -K, axis=1)[:, -K:]
    sims_part = np.take_along_axis(S, idx_part, axis=1)

    order = np.argsort(-sims_part, axis=1)
    top_idx = np.take_along_axis(idx_part, order, axis=1).astype(np.int64, copy=False)
    top_sim = np.take_along_axis(sims_part, order, axis=1).astype(np.float32, copy=False)

    # For true zero-norm rows, enforce all neighbor sims == 0.0
    if np.any(~nz):
        zero_rows = np.where(~nz)[0]
        top_sim[zero_rows, :] = 0.0

    # Build required Parquet schema
    df = pd.DataFrame(
        {
            "row_id": np.arange(N, dtype=np.int64),
            "nbr_idx": [row.tolist() for row in top_idx],
            "nbr_sim": [row.tolist() for row in top_sim],
        }
    )

    df.to_parquet(output_path, index=False)