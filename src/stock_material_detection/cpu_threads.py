"""CPU thread count for inference.

OpenMP, MKL, and OpenBLAS read their thread limits when a native library
first loads. ``configure_cpu_threads`` only resolves ``STOCK_NUM_THREADS``
and sets those variables, so this module must not import torch, cv2, or
paddle at module level. Call it before any of those are imported.
"""

from __future__ import annotations

import os

_BLAS_THREAD_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
)


def resolve_num_threads() -> int:
    """Positive thread count from ``STOCK_NUM_THREADS``, or 1 when unset."""
    raw = os.environ.get("STOCK_NUM_THREADS")
    if raw is None or not raw.strip():
        return 1
    try:
        count = int(raw.strip())
    except ValueError as exc:
        raise ValueError(
            f"STOCK_NUM_THREADS must be a positive integer, got {raw!r}"
        ) from exc
    if count < 1:
        raise ValueError(
            f"STOCK_NUM_THREADS must be a positive integer, got {raw!r}"
        )
    return count


def configure_cpu_threads() -> int:
    """Set any missing BLAS thread variable to the resolved count.

    A variable that is already set is left as-is. Returns the count.
    """
    count = resolve_num_threads()
    value = str(count)
    for key in _BLAS_THREAD_VARS:
        os.environ.setdefault(key, value)
    return count


def apply_torch_thread_limit() -> None:
    """Intra-op and inter-op pools. Inter-op throws once parallel work has started."""
    import torch

    count = resolve_num_threads()
    torch.set_num_threads(count)
    try:
        torch.set_num_interop_threads(count)
    except RuntimeError as exc:
        if "cannot set number of interop threads" not in str(exc):
            raise


def apply_opencv_thread_limit() -> None:
    import cv2

    cv2.setNumThreads(resolve_num_threads())
