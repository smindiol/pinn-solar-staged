"""Invariants of m6 / m6_data (r14 Section 8, experiments v15 E9)."""
from pathlib import Path

import numpy as np
import pytest
import torch

from Staged_PINN_Solar import m6, m6_data

BASE_B = Path(__file__).resolve().parents[3] / "dpv-dataset" / "base_b"
needs_base = pytest.mark.skipif(not (BASE_B / "split_m.csv").is_file(), reason="base B not present")


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    out = tmp_path_factory.mktemp("m6")
    info = m6_data.build(out, root=BASE_B, limit=3, report=None)
    return out, info


@needs_base
def test_m6_dataset_has_250_fixed_nodes_per_device(tiny):
    out, info = tiny
    raw = m6_data.load(out)
    assert info["M"] == 250 and raw["idx"].shape[1] == 250
    assert all(np.unique(r).size == 250 for r in raw["idx"])


@needs_base
def test_m6_ohmic_contact_potentials_match_dpv(tiny):
    assert tiny[1]["max_phi_eq_gap"] < 1e-9


@needs_base
def test_m6_dataset_keeps_the_device_split(tiny):
    raw = m6_data.load(tiny[0])
    assert np.bincount(raw["split"]).tolist() == [3, 3, 3]
    assert len(set(raw["uid"])) == 9


@needs_base
def test_m6_output_transform_imposes_the_six_contact_conditions(tiny):
    ctx = m6.Ctx.load(tiny[0], torch.device("cpu"))
    model = m6.new_model(ctx, 0)
    b = m6.batch(ctx, "train", np.random.default_rng(0), 2, 4)
    q = b["q"]
    one = torch.ones_like(q["s"])
    phi0, pn0, _, Jn0, Jp0 = model(one * 0, q)
    phi1, _, pp1, Jn1, Jp1 = model(one, q)
    m = m6._material(q)
    from Staged_PINN_Solar import carriers
    n0 = carriers.electron_density(phi0, pn0, m)
    p1 = carriers.hole_density(phi1, pp1, m)
    model.net[-1].bias.data.normal_(0, 0.3)
    phi0, pn0, _, Jn0, Jp0 = model(one * 0, q)
    phi1, _, pp1, Jn1, Jp1 = model(one, q)
    n0 = carriers.electron_density(phi0, pn0, m)
    p1 = carriers.hole_density(phi1, pp1, m)
    n_eq0 = q["A_n"] / ctx.attrs["S_n0"]
    p_eq1 = q["A_p"] / ctx.attrs["S_p1"]
    assert torch.allclose(phi0, q["phi_eq0"]) and torch.allclose(phi1, q["phi_eq1"] + q["Vt"])
    assert torch.all(Jp0 == 0) and torch.all(Jn1 == 0)
    assert torch.allclose(Jn0, ctx.attrs["S_n0"] * (n0 - n_eq0), rtol=1e-8)
    assert torch.allclose(Jp1, ctx.attrs["S_p1"] * (p1 - p_eq1), rtol=1e-8)


def test_m6_lambdas_follow_the_order_of_magnitude_rule():
    vals = {"phi": 3e-5, "phi_n": 1e-5, "phi_p": 1e-5, "J_n": 0, "J_p": 0, "P": 250.0, "Kn": 0.04}
    lam = m6.lambdas(vals)
    assert lam["data"] == pytest.approx(1e6) and lam["P"] == pytest.approx(1e-2) and lam["Kn"] == pytest.approx(1e2)


@needs_base
def test_m6_run_completes_on_a_tiny_cpu_context(tiny, monkeypatch, tmp_path):
    monkeypatch.setenv("STAGED_PINN_RESULTS", str(tmp_path))
    monkeypatch.setattr(m6, "N_DEV", 2)
    monkeypatch.setattr(m6, "NODES_DATA", 4)
    monkeypatch.setattr(m6, "NODES_PHYS", 4)
    r = m6.run_seed(0, root=tiny[0], pre_epochs=2, adam_steps=2, lbfgs_iters=2, hours=0.1)
    assert set(r["arms"]) == set(m6.ARMS) and all(a["status"] == "ok" for a in r["arms"].values())


@needs_base
def test_m6_residuals_close_on_dpv_fields(tiny):
    """The data term is zero-able: dPV's own potentials satisfy the transform's contact values."""
    raw = m6_data.load(tiny[0])
    d = raw["dev"]
    assert np.allclose(d["phi_eq0"], d["phi_eq0_dpv"]) and np.allclose(d["phi_eq1"], d["phi_eq1_dpv"])
