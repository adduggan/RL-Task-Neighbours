# tools.py
# Small helper utilities for timing and memory conversions

from contextlib import contextmanager
import time


def bytes_to_mb(n_bytes: int) -> float:
    """Convert bytes to megabytes for readability."""
    return n_bytes / (1024 * 1024)


@contextmanager
def time_block(label: str = "block"):
    """Context manager to measure execution time."""
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        print(f"[{label}] Elapsed: {elapsed:.3f}s")