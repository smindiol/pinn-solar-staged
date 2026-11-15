"""Invariants of m5.py: network M5 and the scaled mixed system of r13 (Eqs. 28-33), and the E8 smoke."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from Staged_PINN_Solar import config as C
from Staged_PINN_Solar import carriers, data, data_m, e6, m5

needs_base_b = pytest.mark.skipif(not (data_m.local_root() / data_m.SPLIT_FILE).exists(),
                                  reason="base_b not available")


@pytest.fixture(scope="module")
def ctx():
    cfg = C.network_m().with_(width=16, depth=2)
    return m5.build_ctx(limit=6, cfg=cfg, root=data_m.local_root())


def test_m5_has_267781_parameters_and_five_outputs() -> None:
    """r13 Sec. 6: (12+1) 256 + 4 (257) 256 + 257 * 5 = 267781; outputs tanh, sigmoid, sigmoid, linear, linear."""
    arch = m5.Arch5(11, 256, 5, seed=0)
    assert sum(p.numel() for p in arch.parameters()) == 267781
    u = arch(torch.rand(7, 1, dtype=torch.float64), torch.rand(7, 11, dtype=torch.float64))
    assert u.shape == (7, 5) and u.dtype == torch.float64
    assert torch.all(u[:, 0].abs() <= 1) and torch.all((u[:, 1:3] > 0) & (u[:, 1:3] < 1))


@needs_base_b
def test_minority_contact_currents_vanish_by_construction(ctx) -> None:
    """Eq. 28: jn(1) = 0 and jp(0) = 0 for any parameters; at the equilibrium start every current is zero."""
    model = m5.new_model(ctx, 0)
    torch.nn.init.normal_(model.arch.net[-1].weight)                  # leave the equilibrium start
    xi = torch.rand(4, ctx.prep.n_xi, dtype=torch.float64)
    one, zero = torch.ones(4, 1, dtype=torch.float64), torch.zeros(4, 1, dtype=torch.float64)
    peq = torch.zeros(4, 1, dtype=torch.float64)
    _, _, _, jn1, _ = model(one, xi, peq, peq, 0.0)
    _, _, _, jn0, jp0 = model(zero, xi, peq, peq, 0.0)
    assert torch.all(jn1 == 0) and torch.all(jp0 == 0) and torch.any(jn0 != 0)


@needs_base_b
def test_scaled_system_closes_on_dpv_references(ctx) -> None:
    """The per-device scales and scaled interface currents of m5 close the scaled continuity (Eq. 29) on dPV
    within the mesh floor, and the majority contacts (Eq. 30) at round-off; the minority currents are ~0."""
    bank, sc = ctx.prep.train, ctx.r13["train"]
    for d in range(len(bank)):
        x, G = bank.x_hat[d], bank.G[d]
        mat = data.device_material(data_m.load_device(ctx.prep.refs["train"][d]), ctx.prep.scales)
        j0 = 0
        phi, pn, pp = (ctx.fields["train"][d, j0, c].astype(np.float64) for c in range(3))
        n, p = carriers.electron_density(phi, pn, mat), carriers.hole_density(phi, pp, mat)
        src = (carriers.recombination(n, p, mat) - G)[1:-1] / sc["Gbar"][d]
        jn, jp = sc["jref"][d, j0, 0].astype(np.float64), sc["jref"][d, j0, 1].astype(np.float64)
        half = 0.5 * (np.diff(x)[1:] + np.diff(x)[:-1])
        assert np.mean((np.diff(jn) / half - src) ** 2) < 1e-6
        assert np.mean((np.diff(jp) / half + src) ** 2) < 1e-6
        Bn = pn[0] - np.log1p(jn[0] / sc["beta_n0"][d])
        Bp = -pp[-1] - np.log1p(jp[-1] / sc["beta_p1"][d])              # V = 0 at stage 0
        assert abs(Bn) < 1e-3 and abs(Bp) < 1e-3
        assert abs(jn[-1]) < 1e-2 and abs(jp[0]) < 1e-2
    assert 1.0 < ctx.sK < 1e4


@needs_base_b
def test_terminal_current_matches_the_gr_identity_on_dpv(ctx) -> None:
    """Eq. 31 on the stored currents, J0 J_gen jn(first interface), equals dPV's terminal current within 4e-3 J_sc."""
    sc, ref = ctx.r13["train"], ctx.currents["train"]
    J = ctx.prep.scales.J0 * sc["Jgen"][:, None] * sc["jref"][:, :, 0, 0]
    ok = np.isfinite(ref)
    rel = np.abs(J[ok] - ref[ok]) / np.abs(np.broadcast_to(ref[:, :1], ref.shape)[ok])
    assert np.max(rel) < 4e-3


@needs_base_b
def test_collocation_windows_hold_every_junction_and_contact_node(ctx) -> None:
    """Eq. 33: every node within W_C of each contact and within W_J of the junction is in the batch, plus K_U."""
    b = m5.batch(ctx, "train", 0, [0, 0, 17], n_dev=3)
    bank = ctx.prep.train
    x = b["x"].cpu().numpy().reshape(b["n_dev"], b["k"])
    for r, d in enumerate(b["devices"]):
        xs = bank.x_hat[d]
        xj = e6.junction_position(bank.x_hat[d:d + 1], bank.Ndop[d:d + 1])[0]
        inner = xs[1:-1]
        must = inner[(inner < m5.W_C) | (inner > 1 - m5.W_C) | (np.abs(inner - xj) <= m5.W_J - 1e-9)]
        assert set(np.round(must, 12)) <= set(np.round(x[r], 12))
        assert x[r].size >= must.size + min(m5.K_U, inner.size - must.size) - 2


@needs_base_b
def test_smoke_runs_on_a_tiny_cpu_context(ctx, tmp_path, monkeypatch) -> None:
    """Pretraining then Adam on every stage, one record per stage with every Eq. 32 term, and the summary."""
    monkeypatch.setenv("STAGED_PINN_RESULTS", str(tmp_path))
    monkeypatch.setattr(m5, "N_DEV", 3)
    ctx.K = 2
    s = m5.run_smoke(pre_steps=2, adam_steps=3, parts=1, ctx=ctx)
    rec = json.loads((m5.results_dir() / "e8s-j00.json").read_text())
    assert set(rec["end"]["terms"]) == set(m5.TERMS) and rec["steps"] == 3
    assert {"current_rel", "pce_bound_pp"} <= set(rec["end"])
    assert s["stages"] == [0, 1] and "val_pce_final" in s and "val_pce_pretrained_control" in s


@needs_base_b
def test_gpu_workers_receive_the_r13_scales(ctx) -> None:
    """Regression (e8smoke 2026-11-15): the context handed to each GPU worker carries ctx.r13 and ctx.sK."""
    back = e6._load_ctx(e6._save_ctx(ctx))
    assert back.sK == ctx.sK and set(back.r13) == set(ctx.r13)
    assert np.array_equal(back.r13["val"]["Jgen"], ctx.r13["val"]["Jgen"])
