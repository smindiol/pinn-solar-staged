"""Level 2 invariants of data.py added for Kaggle: where the root is, and when something is downloaded."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from Staged_PINN_Solar import data


def _flat(base: Path) -> Path:
    d = base / "dpv-pinn-canonical"
    d.mkdir(parents=True)
    for rel in data.SOURCES.values():
        (d / Path(rel).name).write_bytes(b"")
    return d


def test_default_root_prefers_env_then_kaggle_input_then_workspace(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("DPV_SALIDAS", raising=False)
    monkeypatch.setattr(data, "KAGGLE_INPUT", tmp_path / "absent")
    assert data.default_root() == data.local_root()
    flat = _flat(tmp_path / "input")
    monkeypatch.setattr(data, "KAGGLE_INPUT", tmp_path / "input")
    assert data.default_root() == flat                       # flat layout found under /kaggle/input
    monkeypatch.setenv("DPV_SALIDAS", str(tmp_path / "chosen"))
    assert data.default_root() == tmp_path / "chosen"        # the environment variable wins


def test_incomplete_kaggle_input_is_not_taken(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("DPV_SALIDAS", raising=False)
    d = tmp_path / "input" / "x"
    d.mkdir(parents=True)
    (d / "dpv_E1A_b000.h5").write_bytes(b"")                # one of four files only
    monkeypatch.setattr(data, "KAGGLE_INPUT", tmp_path / "input")
    assert data.default_root() == data.local_root()


def test_ensure_dataset_downloads_only_the_missing_files(tmp_path, monkeypatch) -> None:
    got = []
    dl = tmp_path / "dl"
    dl.mkdir()

    def fake_download(slug, path):
        got.append((slug, path))
        (dl / path).write_bytes(b"")
        return str(dl / path)

    monkeypatch.setitem(sys.modules, "kagglehub", types.SimpleNamespace(dataset_download=fake_download))
    for var in ("DPV_SALIDAS", "DISABLE_KAGGLE_CACHE"):      # setenv first so teardown restores "absent"
        monkeypatch.setenv(var, "x")
        monkeypatch.delenv(var)
    full = _flat(tmp_path / "full")
    assert data.ensure_dataset(target=full) == full and got == []        # nothing missing: no network
    (full / "dpv_E1B_b001.h5").unlink()
    root = data.ensure_dataset(target=full)
    assert got == [("sgarizado/dpv-pinn-canonical", "dpv_E1B_b001.h5")] and root == dl


needs_dataset = pytest.mark.skipif(not (data.local_root() / "s1").exists(), reason="dPV set not available")


@needs_dataset
def test_device_carries_terminal_current_and_standard_power() -> None:
    dev = data.load_device(data.inventory()[0])
    assert dev.J is not None and dev.J.shape == dev.V.shape
    assert 899.0 < dev.p_in_std < 901.0
