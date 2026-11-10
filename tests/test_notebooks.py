"""The step notebooks: structure, the byte-identical first cell, and one executed in rehearsal mode."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from Staged_PINN_Solar import __steps__, data

REPO = Path(__file__).resolve().parents[1]
NB = REPO / "Staged-PINN-Solar" / "Notebooks"
needs_dataset = pytest.mark.skipif(not (data.local_root() / "s1").exists(), reason="dPV set not available")


def _cells(name):
    return json.loads((NB / name).read_text())["cells"]


def test_every_declared_step_has_its_notebook_and_remote_placement() -> None:
    assert set(__steps__) == {"pce_check", "reference", "e1", "e2", "e5", "e4", "analysis"}
    for name, step in __steps__.items():
        assert step["placement"] == "remote" and step["service"] == "kaggle" and step["job"] == name
        assert f"Notebooks/{name}.ipynb" in step["produces"] and f"Results/{name}" in step["produces"]
        assert (NB / f"{name}.ipynb").exists()
        # the first code cell is the repository binding, copied byte for byte
        assert _cells(f"{name}.ipynb")[0]["source"] == _cells("verification.ipynb")[0]["source"]


@needs_dataset
def test_a_step_notebook_runs_in_rehearsal_mode(tmp_path, monkeypatch) -> None:
    """Cells executed in order, cwd = the notebooks folder, FORGE_RUN_MODE=smoke: the unit result lands under _smoke."""
    monkeypatch.setenv("STAGED_PINN_RESULTS", str(tmp_path))
    monkeypatch.setenv("FORGE_RUN_MODE", "smoke")
    monkeypatch.delenv("FORGE_RUN_UNITS", raising=False)
    monkeypatch.delenv("FORGE_CLONE_ROOT", raising=False)
    monkeypatch.delenv("FORGE_CLONE_COMMIT", raising=False)
    monkeypatch.chdir(NB)
    scope: dict = {}
    for cell in _cells("e5.ipynb"):
        if cell["cell_type"] == "code":
            exec(compile("".join(cell["source"]), "e5.ipynb", "exec"), scope)
    assert scope["ROOT"] == REPO
    assert len(list((tmp_path / "e5" / "_smoke").glob("e5-*.json"))) == 2
