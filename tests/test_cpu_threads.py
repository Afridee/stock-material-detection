"""CPU thread default. Does not load torch, OpenCV, or Paddle."""

from __future__ import annotations

import os

import pytest

from stock_material_detection.cpu_threads import configure_cpu_threads, resolve_num_threads

_BLAS_THREAD_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
)


def test_unset_resolves_to_one(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("STOCK_NUM_THREADS", raising=False)
    for key in _BLAS_THREAD_VARS:
        monkeypatch.delenv(key, raising=False)

    assert configure_cpu_threads() == 1
    for key in _BLAS_THREAD_VARS:
        assert os.environ[key] == "1"


def test_preset_omp_is_left_alone(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("STOCK_NUM_THREADS", raising=False)
    monkeypatch.setenv("OMP_NUM_THREADS", "8")
    monkeypatch.delenv("MKL_NUM_THREADS", raising=False)
    monkeypatch.delenv("OPENBLAS_NUM_THREADS", raising=False)

    assert configure_cpu_threads() == 1
    assert os.environ["OMP_NUM_THREADS"] == "8"
    assert os.environ["MKL_NUM_THREADS"] == "1"
    assert os.environ["OPENBLAS_NUM_THREADS"] == "1"


def test_override_fills_missing_vars(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STOCK_NUM_THREADS", "3")
    for key in _BLAS_THREAD_VARS:
        monkeypatch.delenv(key, raising=False)

    assert configure_cpu_threads() == 3
    assert os.environ["MKL_NUM_THREADS"] == "3"


@pytest.mark.parametrize("raw", ["0", "-2", "two"])
def test_invalid_count_raises(monkeypatch: pytest.MonkeyPatch, raw: str) -> None:
    monkeypatch.setenv("STOCK_NUM_THREADS", raw)
    with pytest.raises(ValueError, match="STOCK_NUM_THREADS"):
        resolve_num_threads()
