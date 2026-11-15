"""Invariants of diag_m.py: the two diagnostic arms of network M on the dpvB2-only base (operator 2026-11-15)."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from Staged_PINN_Solar import config as C
from Staged_PINN_Solar import data_m, diag_m, e6, scratch_m

needs_base_b = pytest.mark.skipif(not (data_m.local_root() / data_m.SPLIT_FILE).exists(),
                                  reason="base_b not available")


def test_pcgrad_removes_every_pairwise_conflict_of_two_terms() -> None:
    """Two conflicting gradients: each projected one is orthogonal to the other's original direction."""
    a, b = torch.tensor([1.0, 0.0], dtype=torch.float64), torch.tensor([-1.0, 1.0], dtype=torch.float64)
    out, n = diag_m.pcgrad({"a": a, "b": b}, np.random.default_rng(0))
    pa = a - torch.dot(a, b) / torch.dot(b, b) * b
    pb = b - torch.dot(b, a) / torch.dot(a, a) * a
    assert n == 2
    assert torch.allclose(out, pa + pb)
    assert float(torch.dot(pa, b)) == pytest.approx(0.0, abs=1e-12)
    assert float(torch.dot(pb, a)) == pytest.approx(0.0, abs=1e-12)
    assert float(torch.dot(out, a)) >= 0.0 and float(torch.dot(out, b)) >= 0.0


def test_pcgrad_leaves_agreeing_gradients_unchanged() -> None:
    """No negative inner product: PCGrad is the plain sum, no conflict counted."""
    g = {"a": torch.tensor([1.0, 2.0, 0.0], dtype=torch.float64), "b": torch.tensor([0.5, 0.0, 3.0], dtype=torch.float64),
         "c": torch.tensor([0.0, 1.0, 1.0], dtype=torch.float64)}
    out, n = diag_m.pcgrad(g, np.random.default_rng(1))
    assert n == 0 and torch.allclose(out, g["a"] + g["b"] + g["c"])


def test_adaptive_lambda_keeps_the_data_weight_at_one() -> None:
    """Eq. 12 rescaled to lambda_data = 1 then Eq. 13: data stays 1, a term with a small gradient gains weight."""
    cfg = SimpleNamespace(eps_g=1e-8, lambda_min=1e-2, lambda_max=1e4)
    grads = {"data": torch.full((4,), 1.0, dtype=torch.float64), "n": torch.full((4,), 0.01, dtype=torch.float64)}
    lam = {"data": 1.0, "n": 1.0}
    new = diag_m.adaptive_update(lam, grads, cfg, alpha=0.5)
    assert new["data"] == pytest.approx(1.0)
    assert new["n"] == pytest.approx(0.5 * 1.0 + 0.5 * 100.0, rel=1e-6)


@needs_base_b
@pytest.mark.parametrize("variant", diag_m.VARIANTS)
def test_diag_arm_runs_on_a_tiny_cpu_context(tmp_path, monkeypatch, variant) -> None:
    """Each arm pretrains on the dpvB2 base, writes the centre, one record per grid stage and a summary."""
    monkeypatch.setenv("STAGED_PINN_RESULTS", str(tmp_path))
    monkeypatch.setenv("STAGED_PINN_DEVICE", "cpu")
    cfg = C.network_m().with_(width=16, depth=2, n_devices_per_batch=3, n_nodes_per_device=8, max_stages=2)
    real = scratch_m.build_ctx
    monkeypatch.setattr(scratch_m, "build_ctx", lambda: real(cfg, limit=3, root=data_m.local_root()))
    monkeypatch.setattr(e6, "PRE_STEPS", 3)
    monkeypatch.setattr(diag_m, "UPDATE_EVERY", 2)
    s = diag_m.run_diag(variant, n_steps=5, parts=1)
    out = diag_m.results_dir()
    assert s["stages"] == [0] and s["diverged"] == []
    rec = json.loads((out / f"e7md-{variant}-j00.json").read_text())
    assert rec["steps"] == 5 and set(rec["end"]["terms"]) >= {"n", "p", "data", "J"}
    assert rec["lambda_start"]["data"] == 1.0
    if variant == "adaptive":
        assert len(rec["lambda_trajectory"]) == 3 and rec["lambda_end"]["data"] == pytest.approx(1.0)
        assert rec["lambda_end"] != rec["lambda_start"]
    else:
        assert rec["lambda_end"] == rec["lambda_start"] and rec["mean_conflicts"] is not None
    assert (out / f"e7md-{variant}.summary.json").exists() and (out / diag_m.PRETRAINED).exists()
